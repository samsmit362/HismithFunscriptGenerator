"""
Reliable video seek by wall-clock time, on top of BaseVideoReader.

The classic OpenCV idiom

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    ret, frame = cap.read()

...is unreliable on many real-world videos (variable-frame-rate, B-frames,
edit decisions, keyframe gaps): ``CAP_PROP_POS_FRAMES`` is a *nominal* index
and the frame the decoder actually lands on may be 1-30+ frames off, and
``capsule.get(CAP_PROP_FPS)`` is only a container-level hint.

``seek_to_time_exact`` instead asks the reader for the FIRST frame whose REAL
presentation time (``frame.pts`` normalised against the stream start - see
``core.video_reader``) satisfies ``t >= target - (1/fps)/3`` (a one-third-
frame early-stop buffer against container timestamp jitter) and returns that
frame.  The returned time (``t_sec``) IS the actual presentation time of the
decoded frame - downstream code MUST use this value, not ``frame_index / fps``
(per project requirement #2).

The reader backend (``core.video_reader.PyAVVideoReader``) does the precise
keyframe seek + codec-buffer flush + forward walk, so this helper stays a thin
operator-facing wrapper that only adds the early-stop buffer and the
``[seek] target=... found=... diff=...ms`` log line.
"""
from typing import Optional, Tuple

import numpy as np

from core.logging import log
from core.video_reader import BaseVideoReader


def frame_timestamp_ms(reader: BaseVideoReader) -> float:
    """
    Current decode position of ``reader`` in milliseconds (the REAL
    ``frame.pts``-based presentation time of the most recently read frame).
    Best-effort: returns 0.0 before the first frame is read.  Use this after
    EVERY ``reader.read()`` / ``reader.seek_ms()`` if the caller needs the
    REAL time of the current frame (not a computed one).
    """
    try:
        v = float(reader.current_time_ms)
    except Exception:
        return 0.0
    if v < 0.0:
        v = 0.0
    return v


def seek_to_time_exact(reader: BaseVideoReader,
                       target_seconds: float,
                       safety_margin_seconds: float = 2.0,
                       fps: float = 30.0
                       ) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Move ``reader`` to the closest frame at time >=
    ``target_seconds - (1/fps)/3`` (operator-required early-stop buffer).

    ``fps`` is used ONLY for the early-stop threshold (1/3 of a frame's time
    subtracted to absorb container timestamp jitter).  The actual frame time
    comes from the reader's real ``frame.pts`` - NEVER from ``fps``.

    Returns
    -------
    (frame, t_sec, seeked) : frame=None on failure; t_sec is the REAL
    presentation time of the decoded frame (never frame_idx/fps);
    seeked=False if no frame could be produced.

    On success the function logs
    ``[seek] target=...s found=...s diff=...ms`` (diff = found-target, in
    ms - positive means we landed slightly AFTER the requested time).

    The function never raises; a failed seek returns (None, 0.0, False).
    """
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

    # The reader's seek_ms already walks forward to the first frame whose
    # real time is >= the requested ms (keyframe seek + codec flush + decode
    # walk).  Pass the buffered threshold so we land just before the target.
    frame, t_ms = (None, 0.0)
    try:
        res = reader.seek_ms(threshold_ms)
        if res is not None:
            frame, t_ms = res
    except Exception:
        frame, t_ms = None, 0.0

    if frame is None:
        # The target was past the end of the stream (or the reader failed):
        # return whatever the last successfully decoded frame was, if any.
        try:
            cur = reader.read()
            if cur is not None:
                frame, t_ms = cur
        except Exception:
            pass
    if frame is None:
        return None, 0.0, False
    return _log_and_return(frame, t_ms)


def is_duplicate_frame(ms: float,
                       prev_ms: Optional[float],
                       frame: np.ndarray,
                       prev_frame: Optional[np.ndarray]) -> bool:
    """
    Identify duplicated / repeated frames using the project rules:

      3-1) presentation time (ms) did not change  -> duplicate (fast path);
      3-2) time changed but cv2.absdiff == 0  -> identical
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
                    grid_w = max(1, w // block_size)
                    grid_h = max(1, h // block_size)
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
