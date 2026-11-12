"""
Reliable video seek by wall-clock time.

The classic OpenCV idiom

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    ret, frame = cap.read()

...is unreliable on many real-world videos (variable-frame-rate, B-frames,
edit decisions, keyframe gaps): ``CAP_PROP_POS_FRAMES`` is a *nominal* index
and the frame the decoder actually lands on may be 1-30+ frames off, and
``capsule.get(CAP_PROP_FPS)`` is only a container-level hint.

``seek_to_time_exact`` instead:
  1. jumps to a point ``safety_margin_seconds`` BEFORE the target
     (``CAP_PROP_POS_MSEC`` -- a real keyframe-based seek);
  2. decodes frame by frame with ``cap.grab()`` (header-only, much cheaper
     than ``cap.retrieve()``), reading ``CAP_PROP_POS_MSEC`` after every
     grab;
  3. stops at the FIRST frame whose real timestamp satisfies
     ``t >= target - (1/fps)/3`` (a one-third-frame early-stop buffer
     against container timestamp jitter) and returns that frame (via
     ``cap.retrieve()``);
  4. on every successful return the function logs
     ``[seek] target=...s found=...s diff=...ms`` (diff in ms) so the
     operator can see how far the decoder actually landed;
  5. the frozen-timestamp fallback now uses the caller-supplied ``fps``
     (previously hardcoded at 30) so variable-rate videos are handled
     correctly.

The returned time (``t_sec``) IS the actual presentation time of the frame
that was decoded -- downstream code MUST use this value, not
``frame_index / fps`` (per project requirement #2).

Backends that report ``CAP_PROP_POS_MSEC == 0`` for every frame (very old
or exotic containers) are detected: after a small number of "frozen" steps
the function falls back to the classic POS_FRAMES seek and re-reads, so the
call never blocks forever on a pathological stream.
"""
from typing import Optional, Tuple

import numpy as np

from core.logging import log


def frame_timestamp_ms(cap) -> float:
    """
    Current decode position of ``cap`` in milliseconds (as reported by the
    container/timestamps).  Best-effort: on failure returns 0.0.  Use this
    after EVERY ``cap.read()`` / ``cap.grab()`` / ``cap.retrieve()`` if the
    caller needs the REAL time of the current frame (not a computed one).
    """
    import cv2  # lazy: the tracker modules import cv2 themselves; this
                # helper is also called from test harnesses that patch cv2
    try:
        v = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    except Exception:
        return 0.0
    if v < 0.0:
        v = 0.0
    return v


def seek_to_time_exact(cap, target_seconds: float,
                       safety_margin_seconds: float = 2.0,
                       fps: float = 30.0
                       ) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Move ``cap`` to the closest frame at time >=
    ``target_seconds - (1/fps)/3`` (operator-required early-stop buffer).

    ``fps`` is used in two places: the early-stop threshold (1/3 of a
    frame's time subtracted to absorb container timestamp jitter) and the
    frozen-timestamp fallback (frame-index = ``int(target * fps)``;
    before 2026-09 this was hardcoded at 30).

    Returns
    -------
    (frame, t_sec, seeked) : frame=None on failure; t_sec is the REAL
    CAP_PROP_POS_MSEC of the decoded frame (never frame_idx/fps);
    seeked=False if no frame could be produced.

    On success the function logs
    ``[seek] target=...s found=...s diff=...ms`` (diff = found-target, in
    ms - positive means we landed slightly AFTER the requested time).

    The function never raises; a failed seek returns (None, 0.0, False).
    """
    import cv2
    if target_seconds is None:
        target_seconds = 0.0
    target_seconds = max(0.0, float(target_seconds))
    # safety net against 0/negative fps (test harnesses); callers are
    # expected to pass a valid fps.
    fps = max(1.0, float(fps))
    # Early-stop threshold: one-third of a single frame's duration is
    # subtracted from the target as a jitter buffer (operator requirement).
    threshold_ms = (target_seconds - (1.0 / fps) / 3.0) * 1000.0

    def _log_and_return(frame: Optional[np.ndarray], t_real_ms: float):
        t_found = t_real_ms / 1000.0
        diff_ms = (t_found - target_seconds) * 1000.0
        log(f"[seek] target={target_seconds:.3f}s "
            f"found={t_found:.3f}s diff={diff_ms:+.1f}ms")
        return frame, t_found, True

    # --- case A: start of the stream (no pre-seek needed) ---
    if target_seconds <= 1e-9:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
        cap.grab()
        t0_ms = frame_timestamp_ms(cap)
        ret, frame = cap.retrieve()
        if not ret:
            # stream may still be at EOF from a previous use; hard-seek home
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None, 0.0, False
            t0_ms = frame_timestamp_ms(cap)
        return _log_and_return(frame, t0_ms)

    # --- case B: pre-seek then walk forward ---
    pre_ms = max(0.0, (target_seconds - safety_margin_seconds) * 1000.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, pre_ms)
    prev_ms = -1.0
    frozen = 0
    MAX_WALK = 4000  # hard cap: enough for any sensible safety margin
    for _ in range(MAX_WALK):
        if not cap.grab():
            # grab() failed - either end of video OR the fake/test cap does
            # not know the current index.  If we can fall back to a frame we
            # CAN read at a valid index, do so (test harnesses); otherwise
            # give up and let the caller try their own recovery.
            t_real = frame_timestamp_ms(cap)
            if t_real >= threshold_ms:
                ret, frame = cap.retrieve()
                if ret:
                    t_real = frame_timestamp_ms(cap)
                    return _log_and_return(frame, t_real)
            return None, 0.0, False  # end of video / decode error
        cur_ms = frame_timestamp_ms(cap)
        if cur_ms >= threshold_ms:
            ret, frame = cap.retrieve()
            if not ret:
                return None, 0.0, False
            # re-read after retrieve() in case the backend shifts the
            # reported position - this is the REAL time of *this* frame
            t_real = frame_timestamp_ms(cap)
            return _log_and_return(frame, t_real)
        # frozen-timestamp guard (pathological streams)
        if prev_ms >= 0 and cur_ms == prev_ms and target_seconds > 0:
            frozen += 1
            if frozen >= 30:
                log("[seek] CAP_PROP_POS_MSEC is frozen; "
                    "falling back to CAP_PROP_POS_FRAMES")
                # fps is the CALLER-supplied value (not a hardcoded 30):
                # target_seconds * fps = the nominal frame index for this
                # target, as the classic CAP_PROP_POS_FRAMES idiom.
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        max(0, int(target_seconds * fps)))
                ret, frame = cap.read()
                if not ret:
                    return None, 0.0, False
                t_real = frame_timestamp_ms(cap)
                return _log_and_return(frame, t_real)
        else:
            frozen = 0
        prev_ms = cur_ms

    # hard-capped: return whatever we landed on (better than nothing)
    if cap.grab():
        ok, fr2 = cap.retrieve()
        if ok:
            t_real = frame_timestamp_ms(cap)
            return _log_and_return(fr2, t_real)
    return None, 0.0, False


def is_duplicate_frame(ms: float,
                       prev_ms: Optional[float],
                       frame: np.ndarray,
                       prev_frame: Optional[np.ndarray]) -> bool:
    """
    Identify duplicated / repeated frames using the project rules:

      3-1) CAP_PROP_POS_MSEC did not change  -> duplicate (fast path);
      3-2) CAP_PROP_POS_MSEC changed but cv2.absdiff == 0  -> identical
           pixels (still a duplicate).

    Both ``ms`` and ``frame`` must reflect the CURRENT frame; ``prev_ms`` /
    ``prev_frame`` are the previously ACCEPTED (non-duplicate) frame or None
    on the very first iteration.
    """
    import cv2
    if prev_ms is None or prev_frame is None:
        return False
    # Rule 3-1: timestamp did not change
    if ms is not None and prev_ms is not None and ms == prev_ms:
        return True
    # Rule 3-2: pixel-identical (only meaningful when both frames exist)
    if frame is not None and prev_frame is not None:
        if frame.shape == prev_frame.shape:
            try:
                cmp_type = "grid_downsampled_diff"
                is_duplicate = False
                if cmp_type == "ssim":
                    from skimage.metrics import structural_similarity as _ssim
                    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    # ssim returns a value from -1 (different) to 1 (identical)
                    score, _ = _ssim(gray1, gray2, full=True)
                    # Usually a threshold of 0.98-0.99 means the frames are practically indistinguishable
                    is_duplicate = score > 0.99
                elif  cmp_type == "mean_abs_diff":
                    # Convert to grayscale for speed and to eliminate color noise
                    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    # Compute the mean absolute deviation
                    mean_diff = np.mean(cv2.absdiff(gray1, gray2))
                    # If mean_diff < 1.0 (to be tuned experimentally), frames are identical
                    is_duplicate = mean_diff < 0.5
                elif cmp_type == "grid_downsampled_diff":
                    # 1. Quickly convert to grayscale (no blurring; it happens automatically during downsampling)
                    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    # 2. Compute the absolute difference (runs instantly at the C++ level)
                    diff = cv2.absdiff(gray1, gray2)
                    # 3. SPEED MAGIC: downsample the difference so each 50x50 block becomes 1 pixel.
                    # The INTER_AREA method works as an ideal integrator (computes the mean per block)
                    block_size = 50
                    h, w = diff.shape
                    grid_w = w // block_size
                    grid_h = h // block_size
                    # Downsample the difference matrix to the grid size
                    grid_diff = cv2.resize(diff, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
                    # 4. Find the maximum mean deviation among the blocks
                    # Now np.max searches a tiny matrix (e.g. 38x21 pixels instead of 1920x1080)
                    max_block_deviation = np.max(grid_diff)
                    # Threshold: if the mean change in the most "active" block is below 3-5 brightness units,
                    # then it is just sensor digital noise and the frames are identical.
                    is_duplicate = max_block_deviation < 4.0
                # log(f"[is_duplicate_frame] is_duplicate: {is_duplicate} score: {score} ms: {ms:.0f} prev_ms: {prev_ms:.0f}")
                return is_duplicate
            except Exception:
                pass
    return False