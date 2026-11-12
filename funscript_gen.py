#!/usr/bin/env python3
"""
funscript_gen.py - FunScript (Hismith) generator from a video.

Thin CLI entry point: argparse + the generation pipeline (the core logic
lives in the `core` package next to this file).

Mode: STREAMING (1 frame in memory) + preview window (OpenCV, 960x540,
centered on screen).  Generation runs until the end of the video, until
q/space is pressed in the preview window, or until the stroke state
machine's direction defender fires (a graceful stop: the collected track
is saved and the post-pass + RDP run on it - the script does NOT die).

Usage:
    python funscript_gen.py video.mp4 --start 5:27
    python funscript_gen.py video.mp4 --start 5:27 --calibrate \
        --tracker yolo --yolo-model yolo26x-pose --point mid-hips
"""

import argparse
import json
import os
import sys
import traceback

# Disable any tqdm progress bar the libraries may spawn (transformers
# "Loading weights", huggingface-hub model downloads, ...).  Under OFS
# Process.new the stderr pipe has a closed parent end, so a
# sys.stderr.flush() in tqdm.status_printer throws OSError [Errno 22]
# and kills the model build.  The hard fix lives in
# core/detectors/groundingdino.py (transformers.utils.logging
# .disable_progress_bar() before Model(...)); this env var additionally
# covers huggingface_hub's download bars (any first-time model pull).
# Set BEFORE the heavy tracker imports (they all happen inside _run,
# i.e. after this point).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Never-throwing sys.stderr shim ─────────────────────────────────────────
# OFS launches us via Process.new(python, args...) - the STDERR pipe has its
# parent (OFS) end in an indeterminate state.  Any third-party that does
# `sys.stderr.flush()` (tqdm.status_printer, numpy warnings, torch, cv2,
# huggingface-hub download bars, SAM 2's load_video_frames_from_jpg_images,
# ...) can therefore raise `OSError(22) Invalid argument` and KILL the
# process before the log line is even flushed.
#
# The HF_HUB_DISABLE_PROGRESS_BARS env above covers huggingface, but does
# NOT cover sam2's own tqdm(img_paths, ...).  The clean, one-line fix is
# to wrap sys.stderr with a shim whose write/flush/close all swallow
# exceptions.  In a headless console the shim transparently forwards to
# the real stderr (all the tqdm bars etc. still appear).  Under OFS the
# writes are best-effort, so the pipeline never dies on a broken pipe.
_orig_stderr = sys.stderr
_orig_stdout = sys.stdout


class _SafeStream:
    """Never-raising wrapper around a text stream.

    Every write/flush/close is best-effort: errors are swallowed silently.
    `writable()` returns True so stdlib libraries (tqdm, logging) still pick
    the stream.  `fileno()` is best-effort too - libraries that only use
    `os.fstat` / `isatty` will fall back to sensible defaults.
    """
    encoding = "utf-8"

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def writable(self):
        return True

    def isatty(self):
        try:
            return bool(self._inner and self._inner.isatty())
        except Exception:
            return False

    def write(self, s):
        try:
            r = self._inner.write(s)
            return r if r is not None else (len(s) if isinstance(s, str) else 0)
        except Exception:
            return len(s) if isinstance(s, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._inner.close()
        except Exception:
            pass

    def fileno(self):
        try:
            return self._inner.fileno()
        except Exception:
            return 2

    def __getattr__(self, name):
        # Delegate unknown attributes (e.g. .name, .mode) to the inner
        # stream - that lets libraries that probe the stream behave
        # normally when the underlying object supports them.
        try:
            return getattr(object.__getattribute__(self, "_inner"), name)
        except Exception:
            return None


sys.stderr = _SafeStream(_orig_stderr)
# stdout is the "primary" OFS log channel - keep it, but still never throw.
sys.stdout = _SafeStream(_orig_stdout)

from core.logging import log, log_file

# ── Dependency check: the minimum set of Python modules required for
# ── any tracker (the default sot-csrt).  On a clean venv (py -m venv)
# ── these are missing; catch the ImportError and write a clear install
# ── hint to the log file + stderr before exiting with code 1.  The log
# ── file is opened FIRST (core.logging is stdlib-only: os/sys/pathlib),
# ── so the "Open stderr log" button in the OFS GUI shows the hint even
# ── when the stdout/stderr pipes are broken (OFS closes the parent end).
_missing = []
cv2 = None
_np = None
try:
    import cv2
except ImportError:
    _missing.append("cv2 (pip: opencv-contrib-python)")
try:
    import numpy as _np
except ImportError:
    _missing.append("numpy")
try:
    from simplification.cutil import simplify_coords_idx  # noqa: F401
except ImportError:
    _missing.append("simplification")

if _missing:
    _py = sys.executable
    _hint = (
        "ERROR: Required Python module(s) not installed: "
        + ", ".join(_missing) + ".\n"
        "Minimum set for the default sot-csrt tracker:\n"
        f"    {_py} -m pip install opencv-contrib-python numpy simplification\n"
        "Full set (all trackers):\n"
        f"    {_py} -m pip install -r requirements.txt"
    )
    log_file(_hint)
    for _t in (None, sys.stderr):
        try:
            print(_hint, file=_t, flush=True)
        except Exception:
            pass
    sys.exit(1)

# startup marker: the very first line that can appear in the stderr log
# (if OFS reports an error but the log is empty - this line is missing,
# which means the crash happened inside the imports themselves).  Written
# via log() so it lands in the log FILE (the reliable channel) as well as
# the stdout/stderr pipes - this keeps the log non-empty on a successful
# import too, so "Open stderr log" always shows something useful.
try:
    log("[boot] funscript_gen.py imported ok | "
          f"python={sys.version.split()[0]} cv2={cv2.__version__} numpy={_np.__version__}")
except Exception:
    pass


def _log_identify_note(tracker) -> None:
    kind = getattr(tracker, "kind", "pose")
    if kind == "sot":
        log("[main] identify - an OpenCV window will open: "
            "drag a box around the point to track (SOT), Enter = confirm ...")
    elif kind == "dino":
        log("[main] identify - an OpenCV window will open: "
            "GroundingDINO detections will be shown (box + point p); "
            "LMB inside the active box moves point p (release = fix), "
            "R = next frame, Enter = confirm ...")
    else:
        log("[main] identify_partners - an OpenCV window will open "
            "(A=active, P=passive; --auto-partners skips it) ...")


def _run(args) -> None:
    from pathlib import Path
    from typing import Optional

    from core.config import Config
    from core.detectors import create_tracker
    from core.interface.rdp_ui import interactive_rdp
    from core.interface import calibrate as _calib_mod
    from core.interface import identify as _ident_mod
    from core.postpass import funscript as _funscript
    from core.postpass.positions import compute_positions
    from core.postpass.vlm import vlm_ollama
    from core.tracking.flow_refiner import FlowRefiner
    from core.tracking.streaming import tracking_stream
    from core.utilities import format_time, parse_time, read_existing_points
    import numpy as np

    log("=" * 60)
    log("[main] funscript_gen.py started")
    log(f"[main] python {sys.version.split()[0]} at {sys.executable}")
    log(f"[main] video     = {args.video}")
    log(f"[main] tracker   = {args.tracker}"
        + (f" ({args.yolo_model})" if args.tracker == "yolo" else "")
        + (f" ({args.dino_model}, target={args.dino_target})" if args.tracker == "groundingdino" else "")
        + ((f", prompt='{args.dino_prompt}'") if (args.tracker == "groundingdino" and args.dino_prompt) else "")
        + (f" ({args.ostrack_model})" if args.tracker == "ostrack" else "")
        + (f" ({args.sam2_model})" if args.tracker == "gsam2" else "")
        + (f" ({args.sot_model})" if (args.tracker in ("sot-vit", "sot-mil")) else ""))
    point_log = args.dino_target if args.tracker == "groundingdino" else args.point
    log(f"[main] point     = {point_log}")
    log(f"[main] motion={args.motion} normalize={args.normalize} "
        f"calibrate={args.calibrate} auto_partners={args.auto_partners} "
        f"vlm={args.vlm} no_rdp={args.no_rdp}")
    log(f"[main] output    = {args.output}")
    log("=" * 60)

    start_s = parse_time(args.start)
    log(f"[main] start time = {start_s:.3f}s")

    # For dino: --dino-target is the tracked zone name (stored in cfg.point);
    # for all other trackers: --point is that name.  --point is IGNORED by
    # the dino branch below (its target comes from --dino-target).
    cfg_point = args.dino_target if args.tracker == "groundingdino" else args.point

    cfg = Config(
        keyframe_interval_ms=args.min_interval_ms,
        motion=args.motion,
        normalize=args.normalize,
        inverted=args.inverted,
        latency_ms=args.latency_ms,
        point=cfg_point,
    )

    # 1. FPS
    log("[main] reading video FPS ...")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()
    cfg.fps = fps
    log(f"[main] FPS = {fps}")

    # ── LLM (Ollama) MODE - EARLY RETURN BRANCH ─────────────────────
    # When the user picked 'ollama' as the tracker, the tracking pipeline
    # (identify / calibrate / streaming / compute_positions) is SKIPPED:
    # instead the user selects a single fixed ROI, we sample one frame per
    # clip_s of the span, crop that ROI out of every sampled frame, shrink
    # to thumbnail, encode base64 and send ALL images in ONE multimodal
    # Ollama generate() call.  The model responds with a STRICT JSON array
    # of {"frame": N, "depth_pct": X} objects; we map N to the real
    # presentation time of the Nth sampled frame and X to pos (direct:
    # 0% inside = pos 0, 100% outside = pos 100).
    #
    # The result REPLACES the per-frame track entirely: after the LLM call
    # we go straight to the (optional) interactive RDP window and then to
    # build_funscript - exactly like the tracking pipeline's tail, so the
    # OFS GUI's import path is identical.
    if args.tracker == "ollama":
        from core.interface.llm_roi import select_llm_roi
        from core.interface.llm_thinking import ThinkingWindow
        from core.interface.rdp_ui import interactive_rdp
        from core.interface.preview import fit
        from core.postpass.funscript import build_funscript
        from core.postpass.llm_gen import (LLMGenParams, generate_track,
                                           MAX_IMAGES_PER_CALL)

        log("[llm] LLM (Ollama) generation mode - select ROI, sample "
            "frames, one multimodal generate() call")
        if not getattr(args, "ollama_model", None):
            raise RuntimeError(
                "[llm] --ollama-model is REQUIRED for tracker=ollama. "
                "List available models:  `ollama list`   or   "
                "`python funscript_gen.py --list ollama`")
        log(f"[llm] model={args.ollama_model!r}  clip_s={args.llm_clip_s}  "
            f"num_ctx={args.num_ctx}  num_predict={args.num_predict}  "
            f"num_batch={args.num_batch}  thumb={args.llm_thumb}  "
            f"prompt={'custom' if args.llm_prompt else 'default'}")
        # 1. ROI selection (single fixed crop in original frame pixels)
        roi = select_llm_roi(args.video, start_s, fps)
        if roi is None:
            log("[llm] user aborted the ROI selection - nothing to generate")
            sys.exit(2)
        x, y, w, h = roi
        log(f"[llm] fixed ROI crop = ({x},{y}) size {w}x{h} px "
            f"(original frame coordinates, applied to every sampled frame)")
        # 2. build the LLM parameter bundle
        llm_params = LLMGenParams(
            model=args.ollama_model,
            clip_s=float(args.llm_clip_s),
            num_ctx=int(args.num_ctx),
            num_predict=int(args.num_predict),
            num_batch=int(args.num_batch),
            thumb=int(args.llm_thumb),
        )
        # The ROI the user just drew in select_llm_roi() MUST be forwarded
        # to every LLMGenParams so _encode_frame crops the frame before
        # sending it as a base64 jpeg chunk — otherwise the WHOLE sampled
        # frame goes to the VLM.  (The user's ROI window and the "as sent
        # to LLM" preview both depend on this.)
        llm_params.roi = roi
        if args.llm_prompt:
            llm_params.prompt = args.llm_prompt
        if getattr(args, "llm_stop", None):
            llm_params.stop_s = parse_time(args.llm_stop)
            log(f"[llm] stop_s = {llm_params.stop_s!r} "
                f"({format_time(llm_params.stop_s)})")
        # 3. generate the (t_sec, pos) track - INSIDE the live "LLM
        #    thinking" window (Q=abandon, ticking timer, ETA from prior runs)

        # 3a. display frame: the exact first sampled frame + the ROI box
        #     (what the model will see), letterboxed + green overlay.
        import cv2 as _cv2
        from core.tracking.seek import seek_to_time_exact as _stse_llm
        _stop_s = (llm_params.stop_s if llm_params.stop_s is not None
                   else None)
        # compute the expected pipeline size (same formula as inside
        # generate_track) so the window can show chunk/frame progress from
        # the start.  clip_s = chunk duration; every frame inside a chunk is
        # sent (e.g. 0.5s x 30fps = 15), one LLM call per chunk.
        _cap = _cv2.VideoCapture(args.video)
        _total = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0) if _cap.isOpened() else 0
        if _total > 0:
            _end_s = _total / max(1e-6, fps)
            _hi = min(_end_s, _stop_s) if _stop_s is not None else _end_s
            _span = max(0.0, _hi - start_s)
            _n_chunks = max(1, int(_span / max(1e-3, llm_params.clip_s)))
            _fpc = max(1, min(int(llm_params.clip_s * fps), llm_params.max_images))
            _n_send = _n_chunks * _fpc
        else:
            _n_chunks = 1
            _fpc = 1
            _n_send = 1
            _hi = start_s
        _cap_pos = start_s
        _disp_frame = None
        if _cap.isOpened():
            try:
                _fr, _t, _ok = _stse_llm(_cap, start_s, fps=fps)
                if _ok and _fr is not None:
                    _disp_frame = _fr
                else:
                    _cap.set(_cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
                    _ret, _fr = _cap.read()
                    if _ret:
                        _disp_frame = _fr
            except Exception:
                pass
        _cap.release()
        if _disp_frame is None:
            raise RuntimeError("[llm] could not read the display frame "
                               f"at start_s={start_s!r}")
        # draw the ROI (letterbox coordinates) so the user can review it
        view, scale, x0, y0 = fit(_disp_frame)
        if roi is not None:
            rx, ry, rw, rh = roi
            p1 = (int(rx * scale) + x0, int(ry * scale) + y0)
            p2 = (int((rx + rw) * scale) + x0, int((ry + rh) * scale) + y0)
            _cv2.rectangle(view, p1, p2, (0, 220, 0), 2)

        log(f"[llm] opening the 'LLM thinking' window "
            f"(images={_n_send}, Q=abandon) ...")
        tw = ThinkingWindow(
            frame_bgr=view,
            model=llm_params.model,
            n_images_target=_n_send,
            video_t_start=start_s,
            video_t_end=float(_hi),
            task_callable=generate_track,
            task_kwargs={"video_path": args.video, "start_s": start_s,
                         "fps": fps, "p": llm_params},
        )
        aborted, _err, track = tw.run()
        if aborted and (not track or len(track) == 0):
            log("[llm] user pressed Q with zero completed chunks - aborting")
            sys.exit(2)
        if aborted:
            log(f"[llm] user pressed Q - generating funscript from "
                f"{len(track)} partial track point(s) (completed chunks only)")
        if _err:
            raise RuntimeError(f"[llm] the LLM call failed: {_err}")
        log(f"[llm] LLM produced {len(track)} (t_sec, pos) point(s)")
        if not track:
            raise RuntimeError("[llm] the LLM answered no points - "
                               "nothing to generate")
        # 4. (optional) interactive RDP / range window
        if not args.no_rdp and len(track) >= 10:
            log("[llm] RDP / range window opens (same semantics as "
                "tracking mode: A/Z start, S/X stop, wheel eps)")
            mode, approx = interactive_rdp(track)
            if mode == "ok" and approx:
                track = list(approx)
                log(f"[llm] RDP mode='ok' -> {len(track)} selected points")
            else:
                log(f"[llm] RDP mode={mode!r} - keeping the raw LLM track")
        # 5. build_funscript (SAME function as the tracking pipeline)
        script = build_funscript(track, fps, cfg, args.title)
        n_actions = len(script["actions"])
        out_path = Path(args.output)
        out_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
        log(f"[done] -> {out_path.resolve()}  "
            f"({n_actions} actions, {script['metadata']['duration']}s)")
        log(f"PASSED: {n_actions} actions generated -> {out_path.name}")
        return

    # 2. (optional) VLM - advisory only, does not gate the run
    if args.vlm:
        log("[main] VLM scene analysis - this may take a while ...")
        vlm_ollama(args.video, start_s, fps, args.vlm_model,
                   zone=f"{cfg_point} zone")

    # 3. Tracker
    log(f"[main] initializing tracker '{args.tracker}' ...")
    if args.tracker == "yolo":
        tracker = create_tracker("yolo", args.point,
                                 model=args.yolo_model, imgsz=args.imgsz)
    elif args.tracker == "mediapipe":
        tracker = create_tracker("mediapipe", args.point, fps=fps)
    elif args.tracker == "groundingdino":
        # dino target comes from --dino-target (its OWN list, independent of
        # --point); --dino-prompt optionally overrides the named target's
        # built-in phrase.
        tracker = create_tracker("groundingdino", args.dino_target,
                                 prompt=args.dino_prompt,
                                 variant=args.dino_model)
    elif args.tracker == "ostrack":
        tracker = create_tracker("ostrack", args.point,
                                 model=args.ostrack_model)
    elif args.tracker == "gsam2":
        # Grounded SAM 2 - needs the video path (init_state) + the start
        # frame index so the mask is seeded on the correct source frame.
        tracker = create_tracker("gsam2", args.point,
                                 variant=args.sam2_model,
                                 video_path=args.video,
                                 start_frame=int(start_s * fps),
                                 axis_mode=args.sam2_axis,
                                 frames_backend=args.sam2_frames,
                                 ema_alpha=args.sam2_ema)
    else:  # sot-*
        # sot-vit honours args.sot_model (explicit ONNX weights via
        # cv2.TrackerVit_Params().net); sot-mil/sot-csrt/sot-kcf do not
        # take an external model file (mil builds its classifier online,
        # csrt/kcf are self-contained) - the tracker silently ignores it.
        tracker = create_tracker(args.tracker, args.point,
                                 onnx_file=args.sot_model)
    log("[main] tracker ready")

    # 4. Identify the partners (active / passive) - or the SOT ROI
    #    - or the DINO ROI window (pick target box + place point p inside)
    _log_identify_note(tracker)
    active, passive = _ident_mod.identify_partners(
        args.video, start_s, fps, tracker,
        auto=(args.auto_partners and tracker.kind not in ("sot", "dino")),
        no_scene_roi=getattr(args, "no_scene_roi", False) and tracker.kind == "sot")
    log("[main] partners identified")

    # 4b. OPTIONAL second SOT tracker (the "static part" ps) - cloned from
    #     the primary tracker (same algo/weights) and initialized on the
    #     ps ROI the user drew with Ctrl+LMB in the identify window.
    #     Only SOT-class trackers support it (pose / dino trackers have no
    #     box-based ROI; their point p is a keypoint or a percent position
    #     and a separate box tracker would not be coherent with the flow).
    ps_tracker = None
    ps_roi = getattr(tracker, "ps_roi", None)
    if ps_roi is not None:
        if tracker.kind != "sot":
            log(f"[main] warning: tracker '{args.tracker}' has kind="
                f"{tracker.kind!r} - the ps static-part ROI is ignored "
                f"(SOT trackers only)")
            ps_roi = None
    if ps_roi is not None:
        log(f"[main] static-part ps ROI = {ps_roi} - building a second "
            f"'{args.tracker}' tracker for camera-drift compensation")
        # Clone the tracker with the SAME kwargs as the primary one.
        # (The primary was just initialized in identify; we now re-init
        # the clone with the ps ROI on the first (scene-warp) frame.)
        if args.tracker == "gsam2":
            ps_tracker = create_tracker(
                "gsam2", args.point,
                variant=args.sam2_model,
                video_path=args.video,
                start_frame=int(start_s * fps),
                axis_mode=args.sam2_axis,
                frames_backend=args.sam2_frames,
                ema_alpha=args.sam2_ema)
            ps_tracker.scene = getattr(tracker, "scene", None)
        elif args.tracker == "ostrack":
            ps_tracker = create_tracker("ostrack", args.point,
                                        model=args.ostrack_model)
            ps_tracker.scene = getattr(tracker, "scene", None)
        else:  # sot-* (sot-vit / sot-mil / sot-csrt / sot-kcf)
            ps_tracker = create_tracker(args.tracker, args.point,
                                        onnx_file=args.sot_model)
            ps_tracker.scene = getattr(tracker, "scene", None)
        # Seed on the SAME (scene-warp-applied) first frame as the primary
        # tracker so both trackers share one coordinate space.
        _cap_ps = cv2.VideoCapture(args.video)
        from core.tracking.seek import seek_to_time_exact as _stse_ps
        _fr_ps, _t_ps, _s_ok = _stse_ps(_cap_ps, start_s, fps=fps)
        _cap_ps.release()
        if not _s_ok or _fr_ps is None:
            raise RuntimeError(
                "Failed to re-open the first frame to initialize the "
                "static-part (ps) tracker.")
        _fr_ps = (ps_tracker.scene.apply(_fr_ps)
                  if ps_tracker.scene is not None else _fr_ps)
        ps_tracker.init(_fr_ps, tuple(int(v) for v in ps_roi))
        log(f"[main] static-part tracker ready: {type(ps_tracker).__name__}")
        del _fr_ps, _t_ps, _s_ok

    # 5. Axis calibration (f = 0% in, b = 100% out)
    #    --calibrate OFF: no marks, no axis (f=b=None, initial_axis=None) —
    #    the StrokeStateMachine derives bf from the first stroke's own
    #    displacement, and the auto-calibration pass no longer leaves a
    #    dangling empty calibration window on screen.
    if args.calibrate:
        log("[main] calibrate_thrust - an OpenCV window will open "
            "(mark f and b, Enter=confirm, Q=abort) ...")
        cal = _calib_mod.calibrate_thrust(
            args.video, start_s, fps, tracker, active, auto=False, passive=passive)
        log(f"[main] calibration done, initial axis = {cal.initial_axis}")
    else:
        log("[main] calibration skipped - no f/b marks, no axis "
            "(StrokeStateMachine will derive bf from the first stroke)")
        cal = _calib_mod.ThrustCalibration()

    # 6. Streaming: raw tracking of the active partner point p
    if args.flow == "off":
        flow_refiner = None
        log("[main] flow refinement = off (raw pose point as-is)")
    else:
        flow_refiner = FlowRefiner(mode=args.flow, win_size=args.flow_win,
                                   max_jump_px=args.flow_jump)
        log(f"[main] flow refinement = {args.flow} "
            f"(win auto/forced={args.flow_win or 'auto'}, "
            f"jump auto/forced={args.flow_jump or 'auto'})")
    # ── 7.5 existing-points plumbing (requirement #2 + #3) ─────────────
    # main.lua wrote <ext_dir>/hf_existing_points.txt with one line per
    # pre-existing action: "<time_seconds> <pos>".  From it we derive:
    #   * stop_at    : the FIRST existing time at-or-after start_s (the
    #                  streaming loop will break exactly there);
    #   * anchor_begin_pos / anchor_end_pos: the anchor pos values used by
    #                  compute_positions() to rescale the boundary stroke(s)
    #                  so the new segment lands exactly on the old ones.
    existing_files = []
    if hasattr(args, "existing_points_file") and args.existing_points_file:
        existing_files = [args.existing_points_file]
    stop_at: Optional[float] = None
    anchor_begin_pos: Optional[float] = None
    anchor_end_pos: Optional[float] = None
    for path in existing_files:
        if stop_at is not None:
            # first file already provided the boundary -- skip
            continue
        ext, ab, ae = read_existing_points(path, start_s)
        if not ext:
            continue
        # Boundary selection (the "hole" we are generating inside):
        #   * anchor_begin = the LAST keyframe at-or-BEFORE start_s
        #     (or the one AT start_s: in that case it is the LEFT endpoint
        #     of the new segment -- the stroke START).  This value is the
        #     known physical starting position; the stream begins here.
        #   * anchor_end   = the FIRST keyframe STRICTLY after start_s.
        #                    If there is NO such keyframe, the new segment
        #                    extends to END OF VIDEO (anchor_end stays None
        #                    AND stop_at stays None).
        #
        # Semantics of "playhead on a keyframe":
        #   - "playhead at the START of the hole" (leftmost keyframe in
        #     the existing dump) -> the new segment goes FORWARD (to the
        #     next keyframe, or to end-of-video if none);
        #   - "playhead INSIDE the hole" (a < start < b; both a and b exist)
        #     -> the new segment ends at b (b is the RIGHT boundary);
        #   - "playhead at the END of the hole" (rightmost keyframe in the
        #     dump) -> same as end-of-hole case: stream to END OF VIDEO.
        #
        # (This is the "1 got 1 point" scenario: the old rule
        # `stop_at = first t >= start_s` made T0 itself the stop, so the
        # stream collapsed to a single frame at start -- even though there
        # were more than 2 frames to the actual end of the stroke.)
        if stop_at is None:
            left_t = max([t for t in ext if t <= start_s], default=None)
            right_t = next((t for t in ext if t > start_s), None)
            if right_t is not None:
                stop_at = right_t
                log(f"[main] boundary anchors: LEFT  (start) = {left_t:.3f}s "
                    f"({format_time(left_t)}), pos={ab if ab is not None else 'N/A'}")
                log(f"[main] boundary anchors: RIGHT / stop = {right_t:.3f}s "
                    f"({format_time(right_t)}), pos={ae if ae is not None else 'N/A'}")
            else:
                # no keyframe after start_s: the new segment extends
                # FORWARD to end of video.
                stop_at = None
                log(f"[main] boundary anchors: LEFT  (start) = {left_t:.3f}s "
                    f"({format_time(left_t)}), pos={ab if ab is not None else 'N/A'}")
                log(f"[main] boundary anchors: RIGHT / stop = None "
                    f"(no existing keyframe after start_s) - "
                    f"the new segment extends to the END OF THE VIDEO.")
        if ab is not None:
            anchor_begin_pos = ab
        if ae is not None:
            anchor_end_pos = ae
        print(f"[main] existing points from {path!r}: n={len(ext)} "
              f"anchor_begin={anchor_begin_pos} anchor_end={anchor_end_pos}")
        if stop_at is not None:
            print(f"[main] stop_at: {stop_at:.3f}s = {format_time(stop_at)}")
    # Log ALL parsed CLI options (not just the ones already above) so the
    # stderr log is self-contained when debugging a crash: this is the
    # EXACT argv the process was launched with, independent of any
    # mid-run state.  vars(args) covers every add_argument in main().
    log("[main] launch options (full, as parsed by argparse):")
    _sorted_keys = sorted(vars(args).keys())
    for _k in _sorted_keys:
        _v = vars(args)[_k]
        if isinstance(_v, str) and len(_v) > 64:
            _v = _v[:64] + "..."
        log(f"[main]   {_k} = {_v!r}")

    log("[main] tracking stream - preview window opens; stop with q/space "
        "(saves the processed part) ...")
    _pos, raw_tracks = tracking_stream(
        args.video, start_s, fps, tracker, active, cal, cfg,
        flow=flow_refiner,
        passive=passive,
        ps_tracker=ps_tracker,
        stop_after_ms=args.stop_after_ms or None,
        hold_tol=args.hold_tol,
        hold_max=args.hold_max,
        stop_at=stop_at)
    # New pipeline fields (2026-09): REAL presentation time per track slot
    # (from cap.get(CAP_PROP_POS_MSEC)) + the number of duplicate frames
    # that were skipped during the stream.
    stream_times = getattr(raw_tracks, "times", None)
    if stream_times is not None:
        log(f"[main] stream: {len(stream_times)} real-time slots "
            f"(ghost slots excluded by compute_positions)")
    else:
        stream_times = None  # legacy: keep all slots with frame_index/fps
    tracker.close()
    if flow_refiner is not None:
        log(f"[main] flow: {flow_refiner.rebinds} rebind(s) to the pose anchor, "
            f"final winSize={flow_refiner.win_size}")

    if not raw_tracks:
        raise SystemExit("Not enough frames to generate.")
    valid_count = sum(1 for t in raw_tracks if t is not None)
    n_ghost = len(raw_tracks) - valid_count
    log(f"[main] tracked {valid_count} valid frame(s) "
        f"({n_ghost} ghost slot(s) excluded from the result)")

    # 7. Post-pass: stroke segmentation + pos per mode + normalization
    log("[main] post-pass: stroke segmentation + positions ...")
    points, strokes = compute_positions(
        raw_tracks, start_s, fps, cal.initial_axis,
        motion=cfg.motion, normalize=cfg.normalize,
        cal_f=cal.f, cal_b=cal.b,
        anchor_begin_pos=anchor_begin_pos,
        anchor_end_pos=anchor_end_pos,
        times=stream_times)
    log(f"[main] {len(points)} points (one per frame), {len(strokes)} strokes "
        f"(motion={cfg.motion}, normalize={cfg.normalize})")
    print(f"[points] {len(points)} points (one per frame), "
          f"strokes: {len(strokes)} (motion={cfg.motion}, normalize={cfg.normalize})")
    for i, s in enumerate(strokes):
        print(f"          #{i}: frames {s.frame_start}-{s.frame_end} dir={s.direction} "
              f"path={s.total_len:.0f}px axis=({s.axis[0]:.3f},{s.axis[1]:.3f})")

    # 6b. Raw CSV (for a repeat RDP run without re-tracking)
    raw_csv = Path(args.output).with_name(Path(args.output).stem + "_raw.csv")
    np.savetxt(raw_csv, np.asarray(points, dtype=np.float64),
               delimiter=",", header="time_s,pos", comments="", fmt="%.4f")
    log(f"[raw] -> {raw_csv}")

    # 8. RDP + range selection (interactive, single window with 3 sliders)
    if args.no_rdp:
        approx = points
        log("[rdp] skipped (--no-rdp), keeping all points")
    else:
        log("[main] RDP/range window opens - trim with Start/Stop sliders, "
            "tune Epsilon with wheel/drag, Enter = OK, Esc = keep originals ...")
        mode, approx = interactive_rdp(points)
        if mode == "all" or approx is None:
            # Esc or track-too-short: use the ORIGINAL full track, unscaled.
            approx = points
            log("[rdp] Esc / short track: keeping all points (unscaled)")
        elif mode == "closed_ext":
            # user closed the RDP window with X (Win32 WM_CLOSE) - keep the
            # full unscaled track (same outcome as Esc, distinct log line
            # so the GUI can explain which one happened)
            approx = points
            log("[rdp] RDP window was closed by the user: keeping all points (unscaled)")
        else:
            # "ok": approx is already a list of (t_sec, pos_scaled 0..100)
            # tuples (or an ndarray if a future edit reverts to it) --
            # normalise to a plain list for build_funscript.
            if not isinstance(approx, (list, tuple)):
                approx = np.asarray(approx, dtype=np.float64).tolist()
            log(f"[rdp] OK: {len(approx)} frames, pos 0..100 (rescaled)")

    # 9. FunScript JSON
    # approx (or `points` when no_rdp) is a sequence of (t_sec, pos) where
    # t_sec ALREADY carries the start_s offset (compute_positions).  RDP may
    # return a SUBSET of the track - times remain absolute, so no rel-frame
    # re-derivation is needed.  build_funscript does the final seconds->ms
    # conversion and adds the user-configured latency.
    script = _funscript.build_funscript(approx, fps, cfg, args.title)
    n_actions = len(script["actions"])
    out_path = Path(args.output)
    out_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
    log(f"[done] -> {out_path.resolve()}  "
        f"({n_actions} actions, {script['metadata']['duration']}s)")
    # single canonical success line: the OFS GUI (main.lua) scans the last
    # log line for this marker to show a friendly status without dumping
    # the whole traceback into the window.
    log(f"PASSED: {n_actions} actions generated -> {out_path.name}")


YoloPoseModels = ("yolov8n-pose", "yolov8s-pose", "yolov8m-pose",
                  "yolov8l-pose", "yolov8x-pose",
                  "yolo11n-pose", "yolo11s-pose", "yolo11m-pose",
                  "yolo11l-pose", "yolo11x-pose",
                  "yolo26n-pose", "yolo26s-pose", "yolo26m-pose",
                  "yolo26l-pose", "yolo26x-pose")

from core.config import MOTION_MODES, NORMALIZE_MODES, TARGETS
from core.detectors import TRACKERS


def main():
    # --list [ostrack | vittrack]: machine-readable listing of ONNX files,
    # one bare filename per line, taken from ONE family folder only:
    #   --list ostrack   -> models/ostrack/*.onnx   (OSTrack weights)
    #   --list vittrack  -> models/vittrack/*.onnx  (sot-vit weights)
    # without an argument both sections are printed (diagnostic use only;
    # main.lua reads the two family folders directly, not this list).
    if "--list" in sys.argv:
        from core.detectors.models import (DEFAULT_SAM2_VARIANT,
                                           list_ostrack_models,
                                           list_sam2_models,
                                           list_vittrack_models,
                                           SAM2_VARIANTS)
        i = sys.argv.index("--list")
        arg = sys.argv[i + 1] if (i + 1 < len(sys.argv)
                                  and not sys.argv[i + 1].startswith("-")) else None
        # Robust the family token against shell/quoting artefacts.  The OFS
        # GUI builds the io.popen command line by string concatenation, and
        # some shells (io.popen -> /c cmd.exe) leak a trailing ';' or '"'
        # into the argv token (e.g. `ollama;` instead of `ollama`), which
        # made `--list ollama` fall through to "unknown family".  Strip any
        # non-letter tail and normalise to lowercase before matching.
        if arg:
            arg_c = arg.strip().strip('";,')
            if arg_c:
                arg = arg_c.lower()
        else:
            arg = None
        if arg in (None, "all", "ollama"):
            # Ollama VLM model list - comes from a RUNNING `ollama serve`
            # instance (the same server used by the generate() call).
            # Deliberately opt-in via `--list ollama` (or `--list all`) so
            # a missing Python client does not kill the other listings.
            from core.postpass.llm_gen import _list_ollama_models
            try:
                for name in _list_ollama_models():
                    print(name, flush=True)
            except Exception as e:
                sys.stderr.write(f"[list-ollama] {e!r}\n")
                sys.exit(1)
        if arg in (None, "all", "ostrack"):
            for name in list_ostrack_models():
                print(name, flush=True)
        if arg in (None, "all", "vittrack"):
            for name in list_vittrack_models():
                print(name, flush=True)
        if arg in (None, "all", "sam2"):
            # SAM 2 checkpoint picker (gsam2 tracker) - the fixed set of
            # official variants, DEFAULT printed FIRST (GUI combo index 1),
            # then the rest in table order.
            names = [DEFAULT_SAM2_VARIANT]
            names += [v for v in SAM2_VARIANTS if v != DEFAULT_SAM2_VARIANT]
            # append any user-placed *.pt in models/sam2/ that is not an
            # official variant name
            local = [f for f in list_sam2_models() if f not in names]
            for name in names + local:
                print(name, flush=True)
        if arg in (None, "all", "dino"):
            # DINO target list - straight from core/detectors/groundingdino.py
            # (DINO_TARGETS: target name -> prompt).  The OFS GUI (main.lua)
            # reads this list and passes the pick back as --dino-target.
            # Deliberately NOT imported in the module top-level: keeps the
            # `--list` path cheap (no tracking imports needed).
            from core.detectors.groundingdino import (DEFAULT_DINO_TARGET,
                                                      DINO_TARGETS)
            # default FIRST (GUI combo index 1), then the rest in table order
            for name in ([DEFAULT_DINO_TARGET]
                         + [t for t in DINO_TARGETS if t != DEFAULT_DINO_TARGET]):
                print(name, flush=True)
        if arg not in (None, "all", "ostrack", "vittrack", "dino", "sam2",
                       "ollama"):
            sys.stderr.write(f"[list] unknown family {arg!r}; "
                             f"expected 'ostrack', 'vittrack', 'dino', "
                             f"'sam2', or 'all'\n")
            sys.exit(1)
        sys.exit(0)

    ap = argparse.ArgumentParser(
        description="FunScript generator for Hismith - streaming + preview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python funscript_gen.py video.mp4 --start 5:27
  python funscript_gen.py video.mp4 --start 5:27 --calibrate --tracker yolo --yolo-model yolo26x-pose
  python funscript_gen.py video.mp4 --tracker sot-vit   # SOT: draw an ROI, tracked point = its center
        """)
    ap.add_argument("video", help="Path to the video")
    ap.add_argument("--start", type=str, default="0:00",
                    help="Start time, MM:SS (default 0:00)")
    ap.add_argument("--tracker", choices=list(TRACKERS), default="yolo")
    ap.add_argument("--yolo-model", default="yolo26x-pose",
                    choices=list(YoloPoseModels),
                    help="default yolo26x-pose (v26 family; v11/v8 still available)")
    ap.add_argument("--point", default="mid-hips", choices=list(TARGETS),
                    help="Tracked body zone = point p (left/right = the person's own "
                         "side). IGNORED by SOT trackers (sot-*/ostrack) - they "
                         "track the CENTER of the ROI you draw")
    ap.add_argument("--flow", default="off", choices=["off", "pure", "fusion"],
                    help="Anti-jitter LK optical-flow refinement of point p: "
                         "off (raw point, default) | pure (flow primary, rebind "
                         "on loss) | fusion (anchor + flow particles, best wins)")
    ap.add_argument("--flow-win", type=int, default=0,
                    help="LK window size in px (odd, 15..71). 0 = auto "
                         "(from person scale; box trackers: 31)")
    ap.add_argument("--flow-jump", type=float, default=0.0,
                    help="Max plausible per-frame motion in px. "
                         "0 = auto (0.06 * min(person box w, h))")
    ap.add_argument("--hold-tol", type=float, default=0.0,
                    help="Identity hold-over: max distance (px) from the last "
                         "accepted point to re-adopt the anchor. "
                         "0 = auto (max(80, 3*median of recent steps))")
    ap.add_argument("--hold-max", type=int, default=60,
                    help="Identity hold-over: max consecutive ghost frames "
                         "before resetting (default 60, about 2.5 s at 24 fps)")
    ap.add_argument("--stop-after-ms", type=float, default=0.0,
                    help="Auto-stop the tracking stream after this many ms "
                         "(0 = run to the end; quick A/B/A tests)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO input size (default 1280: more accurate on 1080p)")
    ap.add_argument("--dino-model", default="swinb",
                    choices=["swinb", "swint"],
                    help="GroundingDINO checkpoint (default swinb = Swin-B, "
                         "56.7 zero-shot AP, the most accurate; "
                         "swint = Swin-T, lighter but less precise). "
                         "Weights auto-download into models/groundingdino/ on first run.")
    # GroundingDINO target / prompt - INDEPENDENT of the pose-tracker
    # --point list (DINO targets are free-text detection prompts, defined
    # in core/detectors/groundingdino.py DINO_TARGETS - the single source
    # of truth; the list is printed by `--list dino` and read by the GUI).
    from core.detectors.groundingdino import (DEFAULT_DINO_TARGET, DINO_TARGETS)
    ap.add_argument("--dino-target", default=DEFAULT_DINO_TARGET,
                    choices=list(DINO_TARGETS),
                    help=f"GroundingDINO detection target (a named prompt, "
                         f"default {DEFAULT_DINO_TARGET}). IGNORED by non-dino "
                         f"trackers (--point covers those). Point p (where "
                         f"inside the box) is chosen interactively in the ROI "
                         f"window - default = box center. "
                         f"List the targets: --list dino")
    ap.add_argument("--dino-prompt", default=None,
                    help="GroundingDINO: override the --dino-target's "
                         "built-in phrase with an EXACT custom text prompt "
                         "(e.g. 'the waistband line of a pair of shorts'). "
                         "Only used when --tracker groundingdino.")
    ap.add_argument("--ostrack-model", default=None,
                    help="OSTrack ONNX file (a name from models/ostrack/ or a "
                         "full path). Default: the first *.onnx in "
                         "models/ostrack/ (list them with --list ostrack). "
                         "Draw an ROI around the body zone - the identify "
                         "window uses the SOT ROI mode, R re-selects it "
                         "in the preview.")
    ap.add_argument("--sot-model", default=None,
                    help="sot-vit: ONNX weights file (a name from "
                         "models/vittrack/ or a full path), applied via "
                         "cv2.TrackerVit_Params().net. "
                         "sot-mil / sot-csrt / sot-kcf: IGNORED (MIL trains "
                         "online; csrt/kcf are self-contained). "
                         "Default for sot-vit = auto-download vitTracker.onnx "
                         "into models/vittrack/ then the CWD. "
                         "List available files with --list vittrack.")
    ap.add_argument("--sam2-model", default="sam2.1-hiera-base-plus",
                    help="gsam2: SAM 2 checkpoint variant (default "
                         "sam2.1-hiera-base-plus; sam2.1-hiera-large = "
                         "max accuracy, sam2.1-hiera-tiny = fastest). "
                         "Weights auto-download into models/sam2/ on first "
                         "run. List options: --list sam2.")
    ap.add_argument("--sam2-axis", default="cos",
                    choices=["cos", "principal", "bbox"],
                    help="gsam2: how the tracked point p is derived from the "
                         "silhouette mask. cos|principal = subpixel center "
                         "of mass (O(N), recommended); bbox = mask's "
                         "bounding-box center (debug knob).")
    ap.add_argument("--sam2-frames", default="lazy",
                    choices=["lazy", "jpeg"],
                    help="gsam2: how SAM 2 receives video frames. "
                         "lazy (default, RECOMMENDED) = decode each frame "
                         "on demand via decord - constant RAM, this removes "
                         "the upfront whole-clip RAM spike. "
                         "jpeg = pre-export the video to a JPEG folder and "
                         "let SAM 2 stream from it (faster decode for some "
                         "codecs, but RAM grows with clip length).")
    ap.add_argument("--sam2-ema", type=float, default=0.5,
                    help="gsam2: EMA blend factor over the raw SAM 2 logit "
                         "map, 0.0..1.0.  Dampens the per-frame "
                         "silhouette-boundary jitter that shows up as a "
                         "10-30 px centroid oscillation around the ROI "
                         "center (visible in funscript_gen.stderr.log). "
                         "1.0 = OFF (legacy). Default 0.5 halves the "
                         "weight of any single noisy frame; 0.7/0.8 = "
                         "smoother but slightly lagged; 0.2 = sharper "
                         "tracking. Only applied when frames_backend is "
                         "steady (same predictor run) - the tracker "
                         "resets it in init() / reset().")
    ap.add_argument("--calibrate", action="store_true",
                    help="Interactive f/b calibration (default = auto)")
    ap.add_argument("--auto-partners", action="store_true",
                    help="Skip the identify window (take the first two detected persons)")
    ap.add_argument("--no-scene-roi", action="store_true",
                    help="SOT only: skip the scene ROI setup window and run "
                         "on the full frame as-is (identity transform: no "
                         "zoom, no rotation, no pan, no crop) - fastest "
                         "path, used when the video already has the right "
                         "framing")
    ap.add_argument("--vlm", action="store_true",
                    help="VLM scene analysis (Ollama)")
    ap.add_argument("--vlm-model", default="qhuihui-qwen3.5-abliterated-q6_k:27b")
    # ── LLM (Ollama) generator mode - tracker=ollama ─────────────
    # These knobs are ONLY consumed when --tracker ollama is selected.
    # Defaults mirror core/config.py LLM_DEFAULT_* (single source of
    # truth; see plans/llm-ollama-generation.md).
    ap.add_argument("--ollama-model", dest="ollama_model", default=None,
                    help="Ollama model name to use for tracker=ollama "
                         "(e.g. 'huihui_ai/qwen3-vl-abliterated:32b'). "
                         "List available models: `ollama list` or "
                         "`python funscript_gen.py --list ollama`. "
                         "REQUIRED for tracker=ollama.")
    ap.add_argument("--llm-clip-s", dest="llm_clip_s",
                    type=float, default=0.5,
                    help="LLM mode: how many seconds of video = ONE sampled "
                         "image (default 0.5 => one frame every 0.5 s). "
                         "Smaller = finer temporal resolution but more "
                         "images (and more GPU time).  Larger = coarser "
                         "but fits longer spans in one request.")
    ap.add_argument("--llm-stop", dest="llm_stop", type=str, default=None,
                    help="LLM mode: end of the sampled span, MM:SS "
                         "(default = end of video).  The START is the "
                         "`--start` value; the sampled span is "
                         "[start_s .. llm_stop].")
    ap.add_argument("--num-ctx", dest="num_ctx", type=int, default=200000,
                    help="LLM mode: Ollama explicit context budget "
                         "(tokens, default 200000).  Bigger = more images "
                         "fit in one call + more response room.  Costs "
                         "VRAM.")
    ap.add_argument("--num-predict", dest="num_predict", type=int,
                    default=32000,
                    help="LLM mode: max tokens of the model response "
                         "(default 32000).  Must be larger than the JSON "
                         "answer; smaller = faster (stops on hitting the cap).")
    ap.add_argument("--num-batch", dest="num_batch", type=int,
                    default=2048,
                    help="LLM mode: Ollama prompt-processing batch size "
                         "(default 2048).  Larger is faster on GPU "
                         "(especially for many 1920px images).")
    ap.add_argument("--llm-thumb", dest="llm_thumb", type=int,
                    default=1920,
                    help="LLM mode: PIL thumbnail max side in pixels "
                         "(default 1920, aspect preserved, downscale only). "
                         "Smaller = smaller base64 payload + faster "
                         "encoding; too small = less detail for the VLM.")
    ap.add_argument("--llm-prompt", dest="llm_prompt", type=str,
                    default=None,
                    help="LLM mode: override the default depth-of-penetration "
                         "prompt with a custom string (default: see "
                         "core.config.LLM_DEFAULT_PROMPT).  The prompt MUST "
                         "demand the strict JSON shape "
                         "[{\"frame\": N, \"depth_pct\": X}, ...].")
    ap.add_argument("--motion", default="complex", choices=list(MOTION_MODES),
                    help="Motion interpretation: straight | arc | parabola | complex (raw path, default)")
    ap.add_argument("--normalize", default="global-max", choices=list(NORMALIZE_MODES),
                    help="per-stroke (each stroke = 100%%) | global-max "
                         "(normalize by the biggest stroke, default)")
    ap.add_argument("--output", default="funscript.json")
    ap.add_argument("--title", default="Hismith Funscript")
    ap.add_argument("--inverted", action="store_true")
    ap.add_argument("--min-interval-ms", type=int, default=42)
    ap.add_argument("--latency-ms", type=int, default=0)
    ap.add_argument("--no-rdp", action="store_true",
                    help="Skip the RDP window (save all per-frame points)")
    ap.add_argument("--existing-points-file", dest="existing_points_file",
                    default=None,
                    help="Path to a text file with one 'time pos' line per "
                         "pre-existing OFS action (written by main.lua). "
                         "The streaming loop STOPS at the first time at/after "
                         "start_s, and the boundary stroke(s) are rescaled so "
                         "the new segment lands exactly on the anchor values "
                         "(see main.lua 'hf_existing_points.txt').")
    args = ap.parse_args()

    try:
        _run(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # user closed a window / pressed Ctrl+C - same convention as abort:
        # exit code 2, OFS must not import a partial result as an error
        msg = "[main] interrupted by the user"
        log_file(msg)
        for _t in (None, sys.stderr):
            try:
                print(msg, file=_t, flush=True)
            except Exception:
                pass
        # final line for the OFS GUI status (single-line, no traceback)
        log_file("ERROR: interrupted by the user (window closed / Ctrl+C)")
        for _t in (None, sys.stderr):
            try:
                print("ERROR: interrupted by the user (window closed / Ctrl+C)",
                      file=_t, flush=True)
            except Exception:
                pass
        sys.exit(2)
    except Exception:
        # The traceback MUST reach the fallback log file first - in the
        # worst case both stdout and stderr are broken pipes and this is
        # the only channel that still works (this is how "exit code 1,
        # empty log" becomes debuggable on any setup).
        tb = traceback.format_exc()
        log_file(tb)
        for _t in (None, sys.stderr):
            try:
                print(tb, file=_t, flush=True)
            except Exception:
                pass
        # canonical ERROR line (LAST line of the log): the OFS GUI displays
        # this single line in the window; the full traceback is still
        # available via "Open stderr log" (Notepad) above.
        err = tb.strip().splitlines()
        last = err[-1].strip() if err else "unknown error"
        err_line = f"ERROR: {last}"
        log_file(err_line)
        for _t in (None, sys.stderr):
            try:
                print(err_line, file=_t, flush=True)
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
