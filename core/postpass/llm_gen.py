"""LLM (Ollama) generation of a FunScript track.

The pipeline for this mode (tracker == "ollama" in the registry):
  1. the user draws a SINGLE fixed ROI on one chosen frame (see
     ``core/interface/llm_roi.py``);
  2. this module samples every ``clip_s`` of the selected video span,
     crops that ROI out of each sampled frame, shrinks it with
     ``PIL.thumbnail`` and base64-encodes it (JPEG, quality 85);
  3. Ollama's ``generate()`` endpoint is called ONCE with every image
     in a single multimodal request;
  4. the answer is a STRICT JSON array of ``{"frame": N, "depth_pct": X}``
     objects - parsed by a tolerant regex + ``json.loads`` (see
     ``_extract_json`` below);
  5. the (t_sec, pos) track is built CLIENT-SIDE:
       t_sec = start_s + N * clip_s
       pos   = depth_pct          (direct: 0% inside -> pos 0, 100% outside -> pos 100)

Design rules (see plans/llm-ollama-generation.md):
  - the crop is a SINGLE fixed rectangle (x, y, w, h) in ORIGINAL
    frame pixels, applied identically to every sampled frame - no
    per-frame ROI tracking;
  - the model never knows the absolute time - we tell it only the
    image ORDER, it returns the per-image depth;
  - every error raises ``RuntimeError`` (the error-propagation rule
    for the ``core`` package: ``sys.exit(str)`` is forbidden because
    on OFS's closed-parent-end stderr pipes it silently drops the
    message - see funscript_gen.py's main() exception handler).
"""
from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

from core.config import (LLM_DEFAULT_CLIP_S, LLM_DEFAULT_NUM_BATCH,
                         LLM_DEFAULT_NUM_CTX, LLM_DEFAULT_NUM_PRED,
                         LLM_DEFAULT_PROMPT, LLM_DEFAULT_THUMB,
                         LLM_JPEG_QUALITY, LLM_REPEAT_PENALTY,
                         LLM_TEMPERATURE)
from core.logging import log
from core.tracking.seek import seek_to_time_exact

#: hard cap on the number of images sent in ONE request - beyond that the
#: multimodal context (image tokens + text) overflows most VLMs even at
#: num_ctx=128k.  64 images at 1920x1920 JPEG is a sensible default ceiling.
MAX_IMAGES_PER_CALL = 64


@dataclass
class LLMGenParams:
    """All knobs for one LLM generation pass (CLI / GUI surface)."""
    clip_s: float             = LLM_DEFAULT_CLIP_S
    num_ctx: int              = LLM_DEFAULT_NUM_CTX
    num_predict: int          = LLM_DEFAULT_NUM_PRED
    num_batch: int            = LLM_DEFAULT_NUM_BATCH
    thumb: int                = LLM_DEFAULT_THUMB
    temperature: float        = LLM_TEMPERATURE
    repeat_penalty: float     = LLM_REPEAT_PENALTY
    prompt: str               = LLM_DEFAULT_PROMPT
    jpeg_quality: int         = LLM_JPEG_QUALITY
    max_images: int           = MAX_IMAGES_PER_CALL
    # ``model`` is set separately by the CLI / GUI (the selected Ollama model)
    model: str = ""
    # the SINGLE fixed crop in ORIGINAL frame pixels: (x, y, w, h);
    # if None the FULL frame is used (no crop).
    roi: Optional[Tuple[int, int, int, int]] = None
    # the stop time for the sampled span (default: end of video)
    stop_s: Optional[float] = None


#: the JSON shape the model must return (a hard requirement in the prompt):
#:   [ { "frame": N, "depth_pct": X }, ... ]
_JSON_OBJ_RE = re.compile(
    r'\{\s*"frame"\s*:\s*(\d+),\s*"depth_pct"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}'
)


def _ts_hms(t_s: float) -> str:
    """Format an absolute video time (seconds) as HH:MM:SS.mmm.

    e.g. 1760.932 -> '00:29:20.932'  (the format used for the per-chunk
    log lines so the user can line them up with the video timeline)."""
    ms = int(round((t_s - int(t_s)) * 1000))
    total = int(t_s)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _extract_json(text: str) -> List[Tuple[int, float]]:
    """Extract ALL ``{"frame": N, "depth_pct": X}`` objects out of the
    model's raw answer.

    Tolerant parser: we do NOT require the whole answer to be valid JSON —
    VLMs still like to add a few "Here are the depths:" prologues around
    the array.  We find every object with the two mandatory keys, in
    order of appearance, and return ``[(frame_int, depth_float), ...]``.

    Raises RuntimeError if the answer contains NONE of the required
    objects - the model did not follow the format instruction.
    """
    if not text:
        raise RuntimeError("[llm_gen] empty response from the VLM "
                           "(did the ollama server start? model loaded?)")
    found = _JSON_OBJ_RE.findall(text)
    if not found:
        snippet = (text[:400] + "...") if len(text) > 400 else text
        raise RuntimeError(
            "[llm_gen] the VLM answer does NOT contain the required "
            "JSON objects {\"frame\": N, \"depth_pct\": X}. "
            f"Answer begins: {snippet!r}")
    out: List[Tuple[int, float]] = []
    for frame_s, pct_s in found:
        try:
            n = int(frame_s)
            x = float(pct_s)
        except ValueError:
            continue
        out.append((n, x))
    # preserve ORDER by frame index (the model may emit them in any order;
    # we want the track sorted anyway - and dedup)
    seen = set()
    dedup = []
    for n, x in sorted(out, key=lambda kv: kv[0]):
        if n in seen:
            continue
        seen.add(n)
        dedup.append((n, x))
    return dedup


def _thumb_b64(frame: np.ndarray, max_side: int,
               quality: int) -> str:
    """BGR ndarray -> PIL thumbnail (keeps aspect) -> JPEG q -> base64 ascii.

    Only SHRINK is applied (never upscale): a 1080p frame cropped to a
    200x100 ROI stays at 200x100 (upscaling a 100-px source to 1920 would
    add no information and bloat the base64 payload).
    """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - Pillow ships with ollama
        raise RuntimeError(
            "[llm_gen] Pillow is required for image encoding - "
            "`pip install Pillow` (it is a dependency of `ollama`)") from e
    # BGR -> RGB (PIL expects RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max_side and max(img.size) > max_side:
        # only shrink (never upscale); PIL.thumbnail preserves aspect
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = base64.b64encode(
        _jpeg_bytes(img, quality)).decode("ascii")
    return buf


def _jpeg_bytes(img, quality: int) -> bytes:
    import io
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=int(quality))
    return bio.getvalue()


def _encode_frame(frame: np.ndarray,
                  roi: Optional[Tuple[int, int, int, int]],
                  thumb: int, quality: int) -> str:
    """Crop (if any) -> PIL thumbnail -> JPEG base64 string."""
    if roi is not None:
        x, y, w, h = (int(round(v)) for v in roi)
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        crop = frame[y:y + h, x:x + w]
    else:
        crop = frame
    return _thumb_b64(crop, thumb, quality)


def _b64_to_preview_bgr(b64: str) -> Tuple[np.ndarray, int, int]:
    """Decode a base64-JPEG (EXACTLY the string that gets passed to
    ``ollama.generate(images=[...])``) back into a BGR ndarray +
    (width, height).

    This is the SINGLE SOURCE OF TRUTH for the live "LLM thinking"
    preview: the SAME byte string that flies off to Ollama on the
    wire is the one the window renders, so there is no divergence
    between "what preview shows" and "what the model receives".

    Raises RuntimeError on malformed base64 / unsupported codec —
    the caller wraps in try/except so a preview hiccup never breaks
    the generation pipeline (the window shows a fallback message).
    """
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise RuntimeError(f"[llm_gen] preview b64 decode failed: {e!r}") from e
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(
            "[llm_gen] preview b64 decode: cv2.imdecode returned None "
            "(unsupported codec?)")
    return img, int(img.shape[1]), int(img.shape[0])


def _list_ollama_models(host: str = "http://127.0.0.1:11434") -> List[str]:
    """``ollama list`` (the Python client; the CLI `ollama list` prints
    the same).  Returns the sorted model names (one per element).
    Raises RuntimeError when the server is unreachable - the user must
    start it (or fix the model name) before running this mode.
    """
    try:
        from ollama import Client
        client = Client(host=host)
        resp = client.list()
        # ollama v0.6.2+ returns a pydantic ListResponse with a `.models`
        # attribute (a list of Model objects with the name in `model`);
        # older v0.1.x returned a plain dict with key "models" and
        # sub-dicts with key "name".  We handle BOTH so a future or
        # older install of the client does not break the list query.
        raw = None
        if hasattr(resp, "models") and resp.models is not None:
            raw = list(resp.models)
        elif isinstance(resp, dict):
            raw = resp.get("models", [])
        elif hasattr(resp, "get"):
            raw = resp.get("models", [])
        names = []
        for m in raw:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model")
            else:
                n = getattr(m, "model", None) or getattr(m, "name", None)
            if n:
                names.append(n)
    except ImportError:
        raise RuntimeError(
            "[llm_gen] the `ollama` Python client is not installed - "
            "run `pip install ollama` (see requirements.txt)")
    except Exception as e:
        raise RuntimeError(
            f"[llm_gen] cannot reach the Ollama server at {host}: {e!r}. "
            "Start `ollama serve` on this machine, or check the model name.")
    names = sorted(set(names))
    if not names:
        raise RuntimeError(
            f"[llm_gen] the Ollama server at {host} has NO models - "
            "pull one with `ollama pull <name>` first.")
    return names


def _call_ollama(model: str, images_b64: List[str],
                 prompt: str,
                 num_ctx: int, num_predict: int, num_batch: int,
                 temperature: float, repeat_penalty: float,
                 host: str = "http://127.0.0.1:11434") -> str:
    """One multimodal generate() call.  Returns the raw response text.

    Uses ``client.generate(..., images=[...])`` (the `ollama` client's
    generate API - the same call as in the reference implementation;
    chat() also works for multimodal, but generate keeps the code path
    closest to the proven sample).
    """
    try:
        from ollama import Client
    except ImportError as e:
        raise RuntimeError(
            "the `ollama` Python client is not installed - `pip install ollama`"
        ) from e
    log(f"[llm_gen] calling ollama model={model!r}, images={len(images_b64)}, "
        f"num_ctx={num_ctx}, num_predict={num_predict}, num_batch={num_batch}, "
        f"temperature={temperature}")
    client = Client(host=host)
    try:
        resp = client.generate(
            model=model,
            prompt=prompt,
            images=images_b64,
            options={
                "num_ctx": int(num_ctx),
                "num_predict": int(num_predict),
                "num_batch": int(num_batch),
                "temperature": float(temperature),
                "repeat_penalty": float(repeat_penalty),
            },
        )
    except Exception as e:
        raise RuntimeError(
            f"[llm_gen] Ollama generate() failed: {e!r}") from e
    # The `ollama` Python client returns:
    #   * a plain dict with key "response" in older builds,
    #   * a pydantic GenerateResponse with a `.response` attribute in v0.4+
    # (same field name, different transport).  Handle BOTH so the parser
    # never depends on the installed client version.
    text = None
    if isinstance(resp, dict):
        text = resp.get("response")
    else:
        text = getattr(resp, "response", None)
        if not isinstance(text, str):
            msg = getattr(resp, "message", None)
            if msg is not None:
                text = getattr(msg, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            f"[llm_gen] empty or unexpected Ollama response: {resp!r}")
    return text


def generate_track(video_path: str,
                   start_s: float,
                   fps: float,
                   p: LLMGenParams,
                   on_phase: Optional[Callable[[str], None]] = None,
                   should_abort: Optional[Callable[[], bool]] = None
                   ) -> List[Tuple[float, float]]:
    """End-to-end LLM generation of the (t_sec, pos) track.

    Returns the RAW track (not RDP-smoothed, no normalization): one
    (t_sec, pos) pair per SAMPLED image the model answered for.
    ``t_sec`` is ABSOLUTE (already carries ``start_s`` + index * clip_s).
    The caller (funscript_gen._run) routes the result through
    interactive_rdp and then build_funscript - same as the tracking
    pipeline does, no post-pass (stroke segmentation is N/A).

    ``on_phase`` is an OPTIONAL progress callback (phase, **kw); it is
    called on "sampling" (with ``sampled=<count>`` at every iteration)
    and on "thinking" (right before the Ollama call).  A callback
    never raises out of this function (it is wrapped in try/except).

    Raises RuntimeError on every hard failure (server down, model not
    found, empty answer, no parsable JSON objects).
    """
    def _phase(phase: str, **kw: Any) -> None:
        if on_phase is None:
            return
        try:
            on_phase(phase, **kw)
        except Exception as e:  # never break the pipeline over UI updates
            log(f"[llm_gen] on_phase({phase!r}) failed: {e!r}")
    _phase("starting")
    if p.model is None or not p.model:
        raise RuntimeError("[llm_gen] an Ollama model name is required "
                           "(--ollama-model <name>)")
    if not (0.05 <= p.clip_s <= 60.0):
        raise RuntimeError(
            f"[llm_gen] clip_s={p.clip_s} is outside the plausible range "
            f"[0.05, 60] seconds")

    # 1. figure out chunk count and total frame budget
    log(f"[llm_gen] step 1/5: cv2.VideoCapture({video_path!r}) ...")
    cap = cv2.VideoCapture(video_path)
    log("[llm_gen] step 1/5: VideoCapture constructor returned "
        f"opened={cap.isOpened()}")
    if not cap.isOpened():
        raise RuntimeError(f"[llm_gen] cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"[llm_gen] video has no frames: {video_path}")
    end_s = total / max(1e-6, fps)
    stop_s = p.stop_s if p.stop_s is not None else end_s
    t_hi = min(end_s, stop_s)
    span = max(0.0, t_hi - max(0.0, start_s))
    if span <= 0.0:
        cap.release()
        raise RuntimeError(
            f"[llm_gen] empty span: start_s={start_s}, stop_s={stop_s}")
    # Each chunk covers clip_s of video; inside a chunk ALL frames are sent
    # (e.g. 0.5s @ 30fps = 15 frames).  n_chunks = how many calls we make.
    n_chunks = max(1, int(span / max(1e-3, p.clip_s)))
    frames_per_chunk = int(p.clip_s * fps)  # e.g. 0.5*30=15
    frames_per_chunk = max(1, min(frames_per_chunk, p.max_images))
    total_frames = n_chunks * frames_per_chunk
    log(f"[llm_gen] span=[{start_s:.2f} .. {t_hi:.2f}]s ({span:.1f}s), "
        f"clip_s={p.clip_s}s -> {n_chunks} chunk(s) x "
        f"{frames_per_chunk} frame(s) = {total_frames} total frame(s)")

    # 2. fast seek to start_s.  We deliberately do NOT use
    #    seek_to_time_exact here: that helper scans forward frame-by-
    #    frame from `max(0, target - 2.0s)` and is a great fit for
    #    the streaming path (real POS_MSEC timestamps), but on a
    #    long file at e.g. 29:20 it means 50k+ grab() calls = a
    #    ~100s stall BEFORE processing even ONE LLM chunk.  For
    #    the LLM pipeline we only need to START reading near start_s;
    #    per-frame absolute time comes from cap.get(POS_MSEC) on
    #    every read().  So a single O(1) keyframe seek + one read()
    #    is sufficient AND fast, and we still read real frame
    #    timestamps for every subsequent frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(0.0, start_s) * fps)))
    ret0, fr0 = cap.read()
    if (not ret0) or (fr0 is None) or (fr0.size == 0):
        # fallback for codecs whose POS_FRAMES is a bit unreliable
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
        ret0, fr0 = cap.read()
    if (not ret0) or (fr0 is None) or (fr0.size == 0):
        cap.release()
        raise RuntimeError(f"[llm_gen] seek to start_s={start_s} failed")
    ts_ms0 = cap.get(cv2.CAP_PROP_POS_MSEC)
    base_ts = ts_ms0 / 1000.0 if ts_ms0 > 0 else float(start_s)
    log(f"[llm_gen] fast-seek: target={start_s:.2f}s real ts={base_ts:.3f}s "
        f"delta={base_ts - start_s:+.3f}s")
    # fr0 becomes frame #0 of chunk 1 (avoids a redundant double-read).
    pending_fr: Optional[np.ndarray] = fr0
    pending_ts = float(base_ts)

    # 3. chunked pipeline: read a chunk -> LLM call -> map to track.
    #    ONE seek only (above), then sequential cap.read() - this is what
    #    makes the phase=starting stall go away (the old code did one
    #    expensive seek_per_frame against a 30-minute file).
    def _read_next() -> Tuple[Optional[np.ndarray], float]:
        nonlocal pending_fr, pending_ts
        if pending_fr is not None:
            fr, ts = pending_fr, pending_ts
            pending_fr = None
            pending_ts = base_ts
            return fr, ts
        ret, fr = cap.read()
        if (not ret) or (fr is None) or fr.size == 0:
            return None, 0.0
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = ts_ms / 1000.0 if ts_ms > 0 else 0.0
        return fr, float(ts)

    processed = 0
    track: List[Tuple[float, float]] = []
    done = False
    aborted_early = False
    chunk_idx = 0
    while not done and chunk_idx < n_chunks:
        # Check abort signal before starting a new chunk (the user pressed
        # Q or Esc); we are BETWEEN chunks here so this is a safe break point.
        if should_abort is not None:
            try:
                if should_abort():
                    log(f"[llm_gen] ABORT after chunk {chunk_idx}/{n_chunks} "
                        f"- returning partial track ({len(track)} pts)")
                    aborted_early = True
                    done = True
                    break
            except Exception:
                pass
        chunk_idx += 1
        chunk_frames: List[np.ndarray] = []
        chunk_ts: List[float] = []
        # read up to frames_per_chunk frames, stopping at t_hi or EOF
        while len(chunk_frames) < frames_per_chunk:
            fr, ts = _read_next()
            if fr is None:
                break
            if ts <= 0.0:
                ts = base_ts + (processed + len(chunk_frames)) / max(1e-6, fps)
            if ts > t_hi and chunk_frames:
                break  # past the stop point
            chunk_frames.append(fr)
            chunk_ts.append(float(ts))
            processed += 1
            _phase("reading", processed=processed, total_frames=total_frames)
        if not chunk_frames:
            done = True
            break
        log(f"[llm_gen] chunk {chunk_idx}/{n_chunks}: "
            f"{len(chunk_frames)} frame(s), "
            f"t=[{_ts_hms(chunk_ts[0])} .. {_ts_hms(chunk_ts[-1])}]")
        images_b64 = [_encode_frame(f, p.roi, p.thumb, p.jpeg_quality)
                      for f in chunk_frames]
        # Show the user the FIRST frame of this chunk, decoded FROM the
        # SAME base64 string that about to fly off to Ollama on the wire
        # (single source of truth: preview == send).  A decode hiccup
        # never breaks the pipeline (fallback "no preview" on screen).
        _prev_bgr: Optional[np.ndarray] = None
        _prev_wh: Optional[Tuple[int, int]] = None
        try:
            _prev_bgr, _pw, _ph = _b64_to_preview_bgr(images_b64[0])
            _prev_wh = (_pw, _ph)
        except Exception as _e:
            log(f"[llm_gen] preview decode failed (non-fatal): {_e!r}")
        _phase("thinking", chunk=chunk_idx, n_chunks=n_chunks,
               processed=processed, total_frames=total_frames,
               preview=_prev_bgr, preview_size=_prev_wh)
        log("[llm_gen] chunk %d/%d: READY, sending %d image(s) to Ollama"
            % (chunk_idx, n_chunks, len(images_b64)))
        t_call0 = time.time()
        text = _call_ollama(
            model=p.model,
            images_b64=images_b64,
            prompt=p.prompt,
            num_ctx=p.num_ctx,
            num_predict=p.num_predict,
            num_batch=p.num_batch,
            temperature=p.temperature,
            repeat_penalty=p.repeat_penalty,
        )
        call_s = time.time() - t_call0
        log(f"[llm_gen] chunk {chunk_idx}/{n_chunks} answered in {call_s:.1f}s "
            f"({len(text)} chars)  <-- Ollama call RETURNED, parsing now")
        log("[llm_gen] chunk %d/%d: calling _extract_json ... "
            % (chunk_idx, n_chunks))
        log(f"[llm_gen] raw answer begins: {text[:200]!r} ...")
        parsed = _extract_json(text)
        log(f"[llm_gen] chunk {chunk_idx}: parsed {len(parsed)} pair(s)")
        # The model's "frame" N is 0-based WITHIN THIS CHUNK; map through
        # chunk_ts to the absolute (real) time.
        for n, x in parsed:
            if n < 0 or n >= len(chunk_ts):
                continue  # the model miscounted - skip this pair
            t = float(chunk_ts[n])
            pos = float(max(0.0, min(100.0, x)))
            track.append((t, pos))

    log("[llm_gen] chunk loop DONE; track size at this point = %d"
        % (len(track),))
    cap.release()
    log("[llm_gen] VideoCapture released; checking track ...")

    if not track and not aborted_early:
        log("[llm_gen] EMPTY track -> raising RuntimeError")
        raise RuntimeError("[llm_gen] the model did not answer for ANY "
                           "of the sampled images - the track is empty")
    if aborted_early and track:
        log(f"[llm_gen] early abort: returning {len(track)} partial track point(s)")
    track.sort(key=lambda kv: kv[0])
    log("[llm_gen] track sorted, about to return")
    log(f"[llm_gen] final track: {len(track)} points, "
        f"t=[{_ts_hms(track[0][0])} .. {_ts_hms(track[-1][0])}], "
        f"pos range=[{min(x for _, x in track):.1f} .. "
        f"{max(x for _, x in track):.1f}]")
    return track
