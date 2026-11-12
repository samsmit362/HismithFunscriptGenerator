"""
Streaming collect: for every frame track the ACTIVE partner's point p.

Person identity is re-matched per frame: the candidate whose point is
CLOSEST to the previously tracked point wins (match_active) - survives
tracker re-numbering and its max(score) fallback no longer teleports the
track onto the passive partner.

Identity hold-over (HoldoverPoint): when the active partner is missing for
a few frames (or the only detector hit is the passive partner, whose point
is far from the last accepted one), the tracker continues from the
LK-predicted "ghost" of the active point instead of snapping to the
passive; when the active re-appears inside the accept tolerance the anchor
is adopted again.

A per-frame jump budget (relative to the running median step) flags likely
re-bindings: an orange ring is drawn, a one-shot log line is emitted and a
running [rebinds=N] counter is shown, so you can audit which frame the track
switched partner.

No position calculation here (that is the post-pass) - only raw tracks, the
calibration overlay and the preview window.
In --motion complex a LIVE StrokeStateMachine is fed on every accepted point
and the DYNAMIC f/b endpoint (the origin of the current stroke: 'b' when the
current motion is advance, 'f' when retract) is drawn instead of the static
user marks - exactly the state the post-pass will use.
Runs until the end of the video OR q/space in the preview, OR the state
machine's direction defender raises Exception("stop") - that is a GRACEFUL
STOP (not an error): the collected track so far is returned, the post-pass
keeps the strokes committed before the conflict, and RDP then runs on that
result.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.config import Config, TARGETS
from core.detectors.person_tracker import PersonTracker
from core.interface.preview import (PREVIEW_H, PREVIEW_W, center_window,
                                    draw_skeleton, fit, setup_window)
from core.interface.win32 import key_held
from core.logging import log
from core.matching import box_iou, match_active
from core.tracking.flow_refiner import FlowRefiner
from core.tracking.holdover import HoldoverPoint, StreamTracks
from core.tracking.seek import (frame_timestamp_ms, is_duplicate_frame,
                                seek_to_time_exact)
from core.types import Person, ThrustCalibration
from core.postpass.strokes import StrokeStateMachine
from core.utilities import format_time
from time import sleep


def tracking_stream(video_path: str, start_s: float, fps: float,
                    tracker: PersonTracker,
                    active: Person,
                    cal: "ThrustCalibration",
                    cfg: Config,
                    flow: Optional[FlowRefiner] = None,
                    passive: Optional[Person] = None,
                    ps_tracker: Optional[PersonTracker] = None,
                    stop_after_ms: Optional[float] = None,
                    hold_tol: float = 0.0,
                    hold_max: int = 60,
                    stop_at: Optional[float] = None) -> Tuple[np.ndarray, StreamTracks]:
    """...  (see module docstring for the full behavioural contract.)

    stop_at: ABSOLUTE video time in seconds.  When set, the streaming loop
        finishes the frame whose time first reaches (>=) stop_at and then
        breaks - the new generated points stop exactly at the first pre-
        existing OFS action, per requirement #2 (main.lua forwards this from
        the existing-points dump so the new segment does not run into the
        old one).

    New invariants (2026-09):
      * the start frame is located with core.tracking.seek.seek_to_time_exact
        (POS_MSEC pre-seek + per-frame walk), not CAP_PROP_POS_FRAMES;
      * every tracked frame carries its REAL timestamp
        (cap.get(CAP_PROP_POS_MSEC)) in tracks.times - NEVER frame_index/fps;
      * DUPLICATE frames (same POS_MSEC as the accepted predecessor, or
        pixel-identical: cv2.absdiff max == 0) are SKIPPED: they are neither
        tracked, stored in tracks, nor fed to the stroke state machine;
      * GHOST frames (identity hold-over) are stored as None slots in tracks
        and are NOT fed to the live StrokeStateMachine - the post-pass
        (compute_positions) drops the None slots, so ghosts reach neither RDP
        nor OpenFunScripter.

      * OPTIONAL second ROI (`ps_tracker`, "static part" of the scene):
        when a second SOT tracker is supplied, BOTH the primary point p and
        the static reference ps are tracked on every non-duplicate frame.
        The ps sequence is preserved 1:1 with `tracks` in `tracks.ps` (a
        separate list, per project requirement #2), and the COMPENSATED
        point  p_eff = p - (ps - ps_baseline)  (ps_baseline = the first
        accepted ps value in the stream, kept FROZEN afterwards) is what
        gets appended to `tracks` and fed to the StrokeStateMachine, so:
          - req 1: the SM's anchor drifts by -(ps_cur - ps_prev) per frame
                    (it is the accumulated p_eff of the stroke extreme);
          - req 3: the raw complex-motion step becomes
                    |(p - p_prev) - (ps - ps_prev)|  - exactly the formula
                    required by the user (ps0 cancels, only the deltas count).
        Ghost frames (p-side hold-over) keep their None-slot semantics in
        BOTH `tracks[slot]` and `tracks.ps[slot]`, so the parallel lists
        stay 1:1 through dedup / ghost filtering.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / fps
    # ── reliable start seek (NOT CAP_PROP_POS_FRAMES) ──────────────────
    frame, first_t, seeked = seek_to_time_exact(cap, start_s, fps=fps)
    if not seeked or frame is None:
        cap.release()
        raise RuntimeError(
            f"Failed to read the first frame at start={start_s:.3f}s "
            "(video too short or unseekable).")
    # REAL presentation time of the first decoded frame (POS_MSEC based):
    # used for stop_after_ms / preview % instead of the nominal start_s.
    first_t = max(0.0, first_t)

    pos_list: List[float] = []
    tracks = StreamTracks()
    last_track: Optional[Tuple[float, float]] = None
    last_point: Optional[Tuple[float, float]] = tuple(active.point)
    last_box: Optional[Tuple[int, int, int, int]] = active.box
    step_buf: List[float] = []
    n_rebinds = 0
    hold = HoldoverPoint(accept_tol=hold_tol, max_lost=hold_max)
    # hold_tol=inf = disabled (baseline: missing anchor -> empty slot)
    hold.enabled = not (hold_tol == float("inf")) and hold_max > 0
    last_accepted: Optional[Tuple[float, float]] = None
    n_hold = 0
    frame_idx = int(start_s * fps)          # nominal counter for % progress
    # ── per-frame bookkeeping (REAL time, duplicate/ghost filtering) ──
    time_list: List[float] = []             # parallel to `tracks`
    n_dup = 0                               # duplicate frames skipped
    n_ghost_out = 0                         # ghost frames stored as None
    prev_ms: Optional[float] = None         # last ACCEPTED timestamp (ms)
    prev_frame: Optional[np.ndarray] = None  # last ACCEPTED raw frame
    cur_ms = frame_timestamp_ms(cap)        # current (first) frame timestamp
    cur_t = first_t                         # current frame REAL time (sec)
    # first frame: initialize the duplicate baseline + process it inline
    prev_ms = cur_ms
    prev_frame = frame.copy()
    tracks.times = time_list

    # ── OPTIONAL "static part" (ps) tracking ─────────────────────────────
    # A second SOT tracker (ps_tracker) can be supplied to compensate the
    # primary point p against the drift of a STATIC reference element in the
    # frame (per user requirement: fixed-camera assumption).  Two parallel
    # lists are kept 1:1 with `tracks`:
    #   * tracks.ps   - the RAW ps sequence (ps_k per accepted frame, per
    #                    req #2: stored in a separate list named `tracks_ps`);
    #   * `self`/tracks - the COMPENSATED point p_eff = p - (ps - ps_base).
    # ps_baseline = the FIRST accepted ps value, kept FROZEN afterwards; this
    # makes the per-frame step reduce to |(p-p_prev)-(ps-ps_prev)| (ps_base
    # cancels) - exactly the user's formula, while the SM anchor still
    # accumulates -(ps_cur - ps_prev) per stroke (req #1).
    ps_list: List[Optional[Tuple[float, float]]] = []   # tracks.ps (raw)
    ps_base: Optional[Tuple[float, float]] = None        # first accepted ps
    tracks.ps = ps_list
    if ps_tracker is not None:
        log("[stream] static-part (ps) compensation ENABLED - the primary "
            "point p is corrected against the drift of the ps reference; "
            "step = |(p - p_prev) - (ps - ps_prev)| in --motion complex")

    cal_f = cal.f
    cal_b = cal.b
    cal_axis = cal.initial_axis

    # ── DYNAMIC f/b (complex mode only) ──
    # StrokeStateMachine runs LIVE on every accepted track point, so in the
    # preview we can draw ONLY the "relevant" end-point for the current stroke:
    #   current stroke is 'advance' (toward f)   -> dynamic b (pos=100 origin)
    #   current stroke is 'retract' (toward b)   -> dynamic f (pos=0  origin)
    # The marker is re-anchored at every reversal commit - this is the
    # "f/b as seen by the post-pass right now", not the user's original marks.
    # In 'complex' the static cal_f / cal_b / axis overlay is skipped.
    complex_mode = (getattr(cfg, "motion", "complex") == "complex")
    sm_stream = StrokeStateMachine(
        bf_axis=cal_axis,
        markers=(cal_f, cal_b),
    ) if complex_mode else None
    n_rev_commits = 0
    dyn_hint = ""               # appended right before the initial hint = "..."

    def _jump_budget() -> float:
        if len(step_buf) >= 5:
            med = float(np.median(step_buf))
            return max(40.0, 3.0 * med)
        return 60.0

    log(f"[stream] Streaming from {format_time(start_s)} to end of video ({format_time(total_dur)})")
    print("[stream] q / space = stop & save")

    win = "Hismith Funscript Preview (active partner point p)"
    # Same window setup as the identify / calibrate windows (setup_window):
    # 960x540, aspect-ratio locked, centered - so every window in the SOT flow
    # is exactly the same size and the ROI drawn there lines up 1:1 with the
    # point / calibration marks shown here.
    setup_window(win)

    # direction-defender stop signal (complex mode): set when the SM raises
    # Exception("stop") - the loop then exits, keeping the collected track;
    # the post-pass segments up to the last committed extreme and RDP
    # (unless --no-rdp) runs on exactly that result.
    stop_stream = False

    # ── per-frame reader (REAL timestamp + duplicate baseline) ─────────
    # The loop above initialized `frame`/`cur_ms` (first frame after the
    # seek).  Each subsequent iteration pulls the NEXT frame and updates
    # cur_ms / cur_t to its REAL presentation time (POS_MSEC) - NEVER
    # computed from a frame index.
    def _pull_next():
        nonlocal frame, cur_ms, cur_t
        if not cap.grab():
            return False
        new_ms = frame_timestamp_ms(cap)
        ret2, new_frame = cap.retrieve()
        if not ret2:
            return False
        frame = new_frame
        cur_ms = new_ms
        cur_t = new_ms / 1000.0
        return True

    ps_cur = None

    while True:
        if stop_stream:
            print(f"[stream] Stopped at {format_time(cur_t)}")
            break
        # first iteration: `frame`/`cur_ms`/`cur_t` already hold the first
        # post-seek frame (see initialization block above)
        if not (time_list):
            pass  # first frame is in place
        else:
            if not _pull_next():
                break

        # Scene transform (SOT only): the same SceneTransform seeded in
        # _identify_sot (stored on tracker.scene) is applied here so the
        # SOT tracker sees the SAME frame coordinates in init (step 2 ROI
        # drag) and in every update() call below.  Identity scene -> the
        # same ndarray object is returned (no copy, no warpAffine).
        sc = getattr(tracker, "scene", None)
        if sc is not None and not sc.is_identity(shape=frame.shape[:2]):
            frame = sc.apply(frame)

        # ── duplicate-frame filter (project rules 3-1 / 3-2) ────────────
        # A duplicated frame is NOT tracked, NOT stored in tracks and NOT
        # fed to the stroke state machine: it would contribute a zero step
        # that distorts the per-stroke progress and pollutes the result
        # with repeated (t, pos) keyframes in OFS.
        # The FIRST processed frame is the baseline (never a duplicate):
        # detect "baseline not yet established" by the empty time_list.
        if time_list and is_duplicate_frame(cur_ms, prev_ms, frame, prev_frame):
            # 3-1 / 3-2: identical timestamp or bit-identical pixels
            n_dup += 1
            frame_idx += 1
            continue

        t = None
        persons = tracker.detect_all(frame)
        raw_anchor: Optional[Tuple[float, float]] = None
        rebind_now = False
        ghost_now = False
        cand_person: Optional[Person] = None
        if persons:
            i, _cost = match_active(persons, last_point, last_box)
            if i < 0:
                i = int(np.argmax([p.score for p in persons]))
            cand_person = persons[i]
            raw_anchor = persons[i].point
            if last_point is not None:
                d = float(np.hypot(persons[i].point[0] - last_point[0],
                                   persons[i].point[1] - last_point[1]))
                if d >= _jump_budget() and d > hold.auto_tol():
                    # big gap between candidates - likely the passive
                    # partner (or a real partner swap).  Keep the audit log.
                    rebind_now = True
                    n_rebinds += 1
                    print(f"[stream] rebind @ {format_time(cur_t)}: "
                          f"jump={d:.0f}px  score={persons[i].score:.2f} "
                          f"prev={last_point} -> now={persons[i].point}")
        # ── identity hold-over decision ──
        # When a hold-over is needed (no anchor, or anchor far from the last
        # accepted point), the LK ghost of the active point is kept for
        # this frame; when the active partner re-appears inside the accept
        # tolerance the anchor is adopted again.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t, ghost_now = hold.decide(gray, raw_anchor, cand_person)
        # FlowRefiner refines the ANCHOR - only applied on accepted
        # frames; ghost frames already carry a flow prediction.
        if not ghost_now and t is not None and flow is not None and raw_anchor is not None:
            t = tuple(flow.step(gray,
                                np.asarray(raw_anchor, dtype=np.float32),
                                cand_person))
        if ghost_now:
            n_hold += 1
            n_ghost_out += 1

        # ── OPTIONAL "static part" (ps) tracking (user requirements 1-3) ─
        # Independent tracker on a SEPARATE ROI (the "static element" per
        # the spec), running on the SAME scene-warp (via tracker.scene).
        # On every non-duplicate frame we pull the raw ps position from
        # ps_tracker.detect_all(frame) (best-scored Person.point).  Two
        # derived quantities are used downstream:
        #      * ps_cur   : RAW px per frame, stored in `tracks.ps[slot]`
        #                   (requirement #2: separate list, 1:1 with tracks,
        #                    parallel to `time_list`).
        #      * p_eff    : the COMPENSATED primary point fed to `tracks`
        #                   and the StrokeStateMachine (requirement #1 + #3):
        #                       p_eff = p - (ps_cur - ps_base)
        #                   where ps_base = the FIRST accepted ps value in
        #                   the whole stream, frozen afterwards.  This gives
        #                   a per-frame step of exactly
        #                       |(p - p_prev) - (ps - ps_prev)|     (req #3)
        #                   because the frozen ps_base cancels in the diff.
        #                   And the SM's anchor (A0 = stroke origin, req #1)
        #                   drifts by -(ps_k - ps_base) at every commit —
        #                   which is just the cumulative compensation the
        #                   user asked for ("anchor -= ps_cur - ps_prev").
        #
        # GHOST frames (p-side hold-over): ps_cur is STILL recorded (ps is
        # an independent tracker and its object is normally stable); the
        # primary `t` stays None so tracks[slot] is None as before and the
        # post-pass drops it.  On a ghost frame p_eff is not computed (the
        # "p - (ps - base)" expression is undefined for a None p) and the
        # SM is not fed (existing behavior).
        #
        # ps LOST this frame (ps_tracker returned []): ps_cur = None,
        # p_eff = t (raw, NO compensation) — safer than holding the old ps
        # because the LK ghost we already use on the primary side is a
        # motion-prediction and the static reference is expected to be
        # essentially stable frame-to-frame.
        ps_prev = ps_cur
        ps_cur = None
        if ps_tracker is not None:
            ps_persons = ps_tracker.detect_all(frame)
            if ps_persons:
                ps_cur = tuple(float(v) for v in ps_persons[0].point)

        if ps_cur is not None and ps_base is None:
            ps_base = ps_cur
            log(f"[stream] ps BASELINE fixed = ({ps_cur[0]:.1f}, {ps_cur[1]:.1f}) "
                f"at t={format_time(cur_t)} (frozen for the whole run)")

        if ps_cur is None:
            ps_cur = ps_prev

        t_store = t
        if (ps_tracker is not None and t_store is not None
                and ps_cur is not None and ps_base is not None):
            t_store = (t_store[0] - (ps_cur[0] - ps_base[0]),
                       t_store[1] - (ps_cur[1] - ps_base[1]))

        # ── store the frame in the result (requirement #2 + #4) ──────────
        # GHOST frames (hold-over on the PRIMARY tracker) are stored as
        # None slots: the post-pass (compute_positions) drops them so they
        # never reach RDP or OpenFunScripter.
        # ACCEPTED frames store the COMPENSATED primary point (t_store);
        # when ps was not usable on this frame, t_store == t (raw).
        # The RAW ps value is persisted 1:1 with tracks in tracks.ps
        # (requirement #2).
        tracks.append(None if ghost_now else t_store)
        time_list.append(cur_t)      # REAL time (POS_MSEC), 1:1 with tracks
        ps_list.append(None if ghost_now else ps_cur)

        # log(f"[tracking_stream] t:{t} ghost_now:{ghost_now}")

        # NB - use t_store (the ps-COMPENSATED point) not raw t: the whole
        # rest of the accepted-frame bookkeeping (step buffer, hold-over auto
        # tolerance, last_point / SM feed) operates on the COMPENSATED
        # trajectory, so rebind detection, LK ghost windows and stroke
        # amplitude all reflect what the post-pass will ACTUALLY see.
        if t is not None and not ghost_now:
            if last_track is not None:
                step = float(np.hypot(t_store[0] - last_track[0],
                                      t_store[1] - last_track[1]))
                step_buf.append(step)
                if len(step_buf) > 60:
                    step_buf = step_buf[-60:]
            if not ghost_now:
                hold.push_accepted(last_track, t_store)
            last_track = t_store
            last_point = t_store
            if cand_person is not None:
                last_box = cand_person.box
            # ── live feed the state machine (complex mode) ──
            # ACCEPTED frames ONLY - ghost / None frames are EXCLUDED from
            # the live SM so the dynamic f/b marker we draw now is EXACTLY
            # the one the post-pass will compute later (which also drops
            # the None slots).  Feeding ghosts used to drift the live anchor
            # away from the final result.
            if sm_stream is not None:
                evt = None
                try:
                    evt = sm_stream.feed(len(time_list) - 1, t_store, cur_t)
                    # sleep(0.5)
                except Exception as e:
                    if str(e) == "stop":
                        # direction defender: the calibration axis and the
                        # real motion no longer agree.  GRACEFUL STOP (this
                        # is a signal, not an error): the loop below exits
                        # at the top, the post-pass keeps the strokes
                        # committed before this frame and RDP runs on the
                        # resulting positions.
                        log(f"[tracking_stream] direction defender fired @ "
                            f"{format_time(cur_t)} - halting the stream, "
                            f"the collected track is kept for RDP")
                        print('[tracking_stream] STOP (direction defender) @ '
                              f'{format_time(cur_t)} - saving the '
                              'collected track into RDP')
                        dyn_hint = "[STOP: direction defender]  "
                        stop_stream = True
                    else:
                        raise
                if evt is not None:
                    n_rev_commits += 1
                    log(f"[tracking_stream] reversal committed: {evt['dir']} (amp={evt['amp']:.0f}px) -> {sm_stream.dir} cur_t:{format_time(cur_t)}")
        pos_list.append(50.0)  # placeholder, real value computed in post-pass

        # ── duplicate baseline: this frame is now the reference for the
        # NEXT frame's 3-1/3-2 check (timestamp + raw pixels).
        prev_ms = cur_ms
        prev_frame = frame.copy()

        # ── preview ──
        canvas, scale, x0, y0 = fit(frame)

        if t is not None:
            px, py = int(t[0] * scale) + x0, int(t[1] * scale) + y0
            if tracker.kind == "sot":
                # ── SOT: solid green point p - EXACTLY like the ROI window ──
                # (identify._identify_sot draws the tracked point as a filled
                # green circle with a small black center dot).  No red base
                # ring / ghost / rebind / flow rings: a single solid green dot
                # always marks the current tracked point (accepted or ghost).
                cv2.circle(canvas, (px, py), 8, (0, 255, 0), -1)
                cv2.circle(canvas, (px, py), 2, (0, 0, 0), -1)
                # status rings only when the track is in a special state
                if ghost_now:
                    cv2.circle(canvas, (px, py), 12, (255, 255, 255), 2)
                    cv2.putText(canvas, "ghost", (px + 16, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif rebind_now:
                    cv2.circle(canvas, (px, py), 14, (0, 140, 255), 2)
            else:
                # Base accepted-point marker - ALWAYS drawn, even with flow=off
                # and no ghost/rebind, so the tracked point is visible every
                # frame (previously only the ghost / rebind / flow states drew
                # anything, so a plain accepted frame showed no marker at all).
                cv2.circle(canvas, (px, py), 10, (0, 0, 255), 2)
                if ghost_now:
                    # white hollow = LK-predicted ghost (active not in frame or
                    # matcher grabbed the passive - carried on flow prediction)
                    cv2.circle(canvas, (px, py), 10, (255, 255, 255), 2)
                    cv2.putText(canvas, "ghost", (px + 14, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    if raw_anchor is not None:
                        # red ring = the (rejected / far) anchor
                        ax, ay = int(raw_anchor[0] * scale) + x0, \
                                 int(raw_anchor[1] * scale) + y0
                        cv2.circle(canvas, (ax, ay), 10, (0, 0, 255), 2)
                elif rebind_now:
                    # orange ring = jumped farther than the running budget
                    cv2.circle(canvas, (px, py), 14, (0, 140, 255), 2)
                if flow is not None and not ghost_now:
                    # green filled = accepted (flow-refined) point
                    cv2.circle(canvas, (px, py), 8, (0, 255, 0), -1)
                    cv2.circle(canvas, (px, py), 2, (0, 0, 0), -1)
                if (flow is not None and not ghost_now and raw_anchor is not None
                        and (abs(raw_anchor[0] - t[0]) > 3 or abs(raw_anchor[1] - t[1]) > 3)):
                    # red ring = raw pose anchor (so you can see where they differ)
                    ax, ay = int(raw_anchor[0] * scale) + x0, int(raw_anchor[1] * scale) + y0
                    cv2.circle(canvas, (ax, ay), 10, (0, 0, 255), 2)
        elif raw_anchor is not None:
            ax, ay = int(raw_anchor[0] * scale) + x0, int(raw_anchor[1] * scale) + y0
            cv2.circle(canvas, (ax, ay), 10, (0, 0, 255), 2)

        # ── OPTIONAL ps (static reference) marker in the preview ────────
        # BGR (255, 0, 0) = pure BLUE — the same color as the ps ROI in the
        # identify window, so the user immediately sees which element is the
        # static reference.
        if ps_tracker is not None and ps_cur is not None:
            sx = int(ps_cur[0] * scale) + x0
            sy = int(ps_cur[1] * scale) + y0
            cv2.circle(canvas, (sx, sy), 6, (255, 0, 0), -1)
            cv2.circle(canvas, (sx, sy), 2, (0, 0, 0), -1)
            cv2.putText(canvas, "ps", (sx + 10, sy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # SOT ROI trackers: draw the CURRENT tracked box every frame so the
        # user can see exactly which zone is being followed (not just the
        # point).  Uses tracker.last_raw_box - the RAW per-frame tracker
        # response (BEFORE size EMA): the EMA-smoothed Person.box is seeded
        # from the initial ROI and damped so heavily (alpha 0.1) that it
        # visually barely drifts, i.e. "the initial size".  The raw box is
        # the real live output; on a lost frame it is None -> skip the box
        # (the green point ghost still shows the held-over position).
        if tracker.kind == "sot":
            raw_box = getattr(tracker, "last_raw_box", None)
            if raw_box is None and cand_person is not None:
                raw_box = cand_person.box  # fallback pre first update
            if raw_box is not None:
                rx, ry, rw, rh = raw_box
                cv2.rectangle(canvas,
                              (int(rx * scale) + x0, int(ry * scale) + y0),
                              (int((rx + rw) * scale) + x0, int((ry + rh) * scale) + y0),
                              (0, 0, 255), 3)
                # point p of the raw box (what the tracker reports) + the
                # EMA-smoothed p actually consumed downstream: draw the raw
                # center when it disagrees with the accepted point by more
                # than 3 px - visible box-size/center drift per frame.
                rcx = (rx + rw / 2.0) * scale + x0
                rcy = (ry + rh / 2.0) * scale + y0
                if t is not None and (abs(rcx - px) > 3 or abs(rcy - py) > 3):
                    cv2.circle(canvas, (int(rcx), int(rcy)), 4, (0, 255, 255), -1)

        if complex_mode:
            # ── DYNAMIC f/b (complex mode) ──
            # Only ONE end-point (the origin of the CURRENT stroke), so the
            # user always sees the f/b that the post-pass will use right now:
            #   current stroke RETRACT (p heading toward b)  -> draw dyn f
            #   current stroke ADVANCE (p heading toward f)  -> draw dyn b
            # Before the first commit there is only one stroke running - its
            # anchor is the USER's b (advance) or f (retract) per rule 1.
            if (sm_stream is not None and sm_stream.started
                    and sm_stream.anchor is not None):
                dpt = sm_stream.anchor
                if (ps_tracker is not None and ps_cur is not None and ps_base is not None):
                    dpt = (dpt[0] + (ps_cur[0] - ps_base[0]),
                            dpt[1] + (ps_cur[1] - ps_base[1]))
                cax = int(dpt[0] * scale) + x0
                cay = int(dpt[1] * scale) + y0
                if sm_stream.dir == "advance":
                    _lbl = "b* "   # origin = pos 100 (retract start)
                    _col = (0, 255, 255)
                else:
                    _lbl = "f* "   # origin = pos 0 (advance start)
                    _col = (0, 160, 255)
                cv2.circle(canvas, (cax, cay), 12, _col, 3)
                cv2.circle(canvas, (cax, cay), 2, _col, -1)
                cv2.putText(canvas, _lbl, (cax + 15, cay + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _col, 2)
                # optional: thin line anchor -> current p (visible pump amplitude)
                if t is not None:
                    px2, py2 = int(t[0] * scale) + x0, int(t[1] * scale) + y0
                    cv2.line(canvas, (cax, cay), (px2, py2), _col, 1)
            if n_rev_commits > 0:
                dyn_hint = f"[{n_rev_commits} rev]  "

        hint = "q / space = stop & save   [hold S] = active+passive skeletons"
        if tracker.kind == "sot":
            hint = "[R] = re-select ROI   " + hint
        if tracker.kind == "dino":
            hint = "[R] = re-aim point p (ROI window)   " + hint
        if dyn_hint:
            hint = dyn_hint + hint
        if flow is not None:
            hint = f"flow={flow.mode} win={flow.win_size}  |  {hint}"
        if ghost_now and hold.lost_run:
            hint = f"[hold={hold.lost_run}]  {hint}"
        if n_rebinds:
            hint = f"[rebinds={n_rebinds}]  {hint}"
        if ps_tracker is not None:
            hint = "[ps=on]  " + hint
        (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, hint, ((PREVIEW_W - tw) // 2, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cur_time = cur_t     # REAL time (POS_MSEC), not frame_idx/fps
        time_txt = f"{format_time(cur_time)} / {format_time(total_dur)}"
        (tw2, _), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, time_txt, (PREVIEW_W - tw2 - 10, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if key_held(ord('s')) or key_held(ord('S')):
            if persons:
                ai, _c = match_active(persons, last_point, last_box)
                if ai < 0:
                    ai = int(np.argmax([p.score for p in persons]))
                draw_skeleton(canvas, scale, x0, y0, persons[ai],
                              color=(0, 220, 255), label="active p")
                if passive is not None and len(persons) > 1:
                    pi = -1
                    best_p = -1.0
                    for j, p in enumerate(persons):
                        if j == ai:
                            continue
                        v = box_iou(p.box, passive.box)
                        if v > best_p:
                            best_p, pi = v, j
                    if pi == -1:
                        pi = [j for j in range(len(persons)) if j != ai][0]
                    draw_skeleton(canvas, scale, x0, y0, persons[pi],
                                  color=(0, 140, 255), label="passive p")
        cv2.imshow(win, canvas)
        if frame_idx % 30 == 0:
            center_window(win, PREVIEW_W, PREVIEW_H)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 32):
            print(f"[stream] Stopped at {format_time(cur_time)}")
            break
        # SOT: R re-opens the ROI selection and re-initializes the tracker
        if tracker.kind == "sot" and key in (ord('r'), ord('R')):
            from core.interface.identify import _identify_sot
            active2, _ = _identify_sot(video_path, cur_t, fps, tracker,
                                       current_video_frame=frame_idx)
            if active2 is not None:
                last_point = tuple(active2.point)
                last_box = active2.box
            # reset the hold-over state so the new ROI re-seeds cleanly
            hold.__init__(accept_tol=hold_tol, max_lost=hold_max)
            hold.enabled = not (hold_tol == float("inf")) and hold_max > 0
            continue
        # DINO: R re-opens the ROI window to re-aim point p inside the box
        if tracker.kind == "dino" and key in (ord('r'), ord('R')):
            from core.interface.identify import _identify_dino
            # hard_error=False: a lost frame during the stream must not kill
            # the run (Q / abort -> (None, None) = keep the previous point)
            active2, _ = _identify_dino(video_path, cur_t, fps,
                                        tracker, hard_error=False)
            if active2 is not None:
                last_point = tuple(active2.point)
                last_box = active2.box
            continue

        frame_idx += 1
        # ── stop conditions (REAL time, per project requirement #2) ─────
        # cur_t comes from cap.get(CAP_PROP_POS_MSEC) after the current
        # retrieve.  Some backends report a STUCK timestamp (e.g. always 0)
        # while still decoding forward, so we fall back to the NOMINAL
        # frame_idx/fps value ONLY in that degenerate case.
        _real_t = (cur_t if cur_t > 1e-9 else (frame_idx / fps))
        if stop_after_ms is not None and (_real_t - first_t) * 1000.0 >= stop_after_ms:
            print(f"[stream] stop_after_ms reached ({stop_after_ms:.0f} ms)")
            break
        if stop_at is not None and _real_t >= stop_at:
            # Requirement #2: the NEW segment ends at the FIRST pre-existing
            # point on the timeline (main.lua computed `stop_at` from the
            # existing-points dump): stop after we've saved THIS frame so the
            # new points meet the old ones without over-writing them.
            print(f"[stream] reached the first existing point on the timeline "
                  f"at {format_time(_real_t)} - stopping")
            break
        if frame_idx % 150 == 0:
            print(f"[stream] {format_time(_real_t)} / {format_time(total_dur)}  "
                  f"({100 * frame_idx / max(total_frames, 1):.0f}%))")

    cap.release()
    cv2.destroyAllWindows()
    if ps_tracker is not None:
        ps_tracker.close()

    if n_rebinds:
        print(f"[stream] session rebinds: {n_rebinds}")
    if n_hold:
        print(f"[stream] session hold-over frames: {n_hold} "
              f"(longest run {hold.max_run})")
    pos_arr = np.array(pos_list, dtype=np.float64)
    # expose session telemetry to harnesses (harmless for normal callers)
    tracks.hold_frames = n_hold
    tracks.hold_max_run = hold.max_run
    tracks.rebinds = n_rebinds
    tracks.times = time_list          # REAL presentation time per track slot
    tracks.dup_skipped = n_dup
    tracks.ghost_count = n_ghost_out
    if n_dup:
        print(f"[stream] duplicate frames skipped: {n_dup}")
    if n_ghost_out:
        print(f"[stream] ghost (hold-over) frames EXCLUDED from the result "
              f"(stored as None slots): {n_ghost_out}")
    return pos_arr, tracks