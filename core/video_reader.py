"""
BaseVideoReader - a backend-agnostic, precise video-frame reader.

The whole pipeline used to drive ``cv2.VideoCapture`` directly and read frame
times via ``cap.get(cv2.CAP_PROP_POS_MSEC)``.  That value is only a
*container-level hint*: it is frequently quantised, off by one or two frames
relative to libmpv (the engine OFS plays video with), and on variable-frame-
rate (VFR) content it is simply wrong.  ``frame_index / fps`` is even worse
(``fps`` is itself only a nominal average).

This module introduces a small reader interface (``BaseVideoReader``) with a
single, precise time source: the REAL presentation timestamp (``frame.pts``)
of each decoded frame, normalised against the stream start:

    time_ms = (frame.pts - start_pts) * float(stream.time_base) * 1000.0

where ``start_pts`` is captured ONCE at open time (before any seek) from
``stream.start_time`` when it is a usable non-negative value, otherwise from
the pts of the first decoded frame.  That removes the 1-2 frame offset
relative to libmpv and never relies on ``fps``.

``PyAVVideoReader`` is the first (and currently the only) concrete
implementation, built on PyAV (``av``).  The rest of the codebase is written
against ``BaseVideoReader`` so a different backend (decord, a native decoder,
a test fake, ...) can be swapped in without touching the pipeline:

    from core.video_reader import open_video_reader
    reader = open_video_reader(path)        # -> a BaseVideoReader

Downstream consumers (``core/tracking/seek.py``, ``streaming.py``,
``identify.py``, ``calibrate.py``, ...) only ever call the interface methods
below and NEVER touch ``fps`` for a frame's time (project requirement #2).
"""
from typing import Optional, Tuple

import numpy as np


class BaseVideoReader:
    """Backend-agnostic video reader.

    Every implementation MUST expose:

      * ``read() -> Optional[Tuple[np.ndarray, float]]``
          Decode and return the NEXT frame as a BGR ``uint8`` ndarray together
          with its REAL presentation time in milliseconds (``frame.pts``
          normalised against the stream start - NEVER ``frame_index / fps``).
          Returns ``None`` at end of stream / on a decode failure.
      * ``seek_ms(ms: float) -> Optional[Tuple[np.ndarray, float]]``
          Move to the first frame whose real time is ``>= ms`` and return it
          (``None`` when the target is past the end of the stream).
      * ``step(n: int) -> Optional[Tuple[np.ndarray, float]]``
          Advance (``n > 0``) or go back (``n < 0``) by ``n`` frames from the
          current position and return the landed-on frame + its time
          (``None`` when the target is outside the stream).
      * ``current_time_ms -> float``
          The real time (ms) of the most recently read frame (0.0 before the
          first ``read``).
      * ``width`` / ``height`` - the decoded frame size in pixels.
      * ``duration_s -> Optional[float]`` - the total video duration in
          seconds, or ``None`` when the container does not report one.
      * ``frame_count -> int`` - the total frame count, or 0 when unknown.
      * ``reset()`` - return to the very start of the stream (flushing any
          buffered decoder state) and read the first frame.
      * ``close()`` - release the underlying decoder / container.

    The interface is deliberately small so it is trivial to fake in tests and
    to re-implement on top of a different decoder in the future.
    """

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        raise NotImplementedError

    def seek_ms(self, ms: float) -> Optional[Tuple[np.ndarray, float]]:
        raise NotImplementedError

    def step(self, n: int) -> Optional[Tuple[np.ndarray, float]]:
        raise NotImplementedError

    def reset(self) -> Optional[Tuple[np.ndarray, float]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def current_time_ms(self) -> float:
        raise NotImplementedError

    @property
    def width(self) -> int:
        raise NotImplementedError

    @property
    def height(self) -> int:
        raise NotImplementedError

    @property
    def duration_s(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        raise NotImplementedError


class PyAVVideoReader(BaseVideoReader):
    """PyAV (``av``) implementation of :class:`BaseVideoReader`.

    Frames are decoded as BGR ``uint8`` (``frame.to_ndarray(format="bgr24")``)
    so the rest of the pipeline (which is BGR-native via OpenCV) consumes them
    unchanged.  The real presentation time of every frame is derived from
    ``frame.pts`` and the stream ``time_base``, normalised against the stream
    start (see the module docstring) - ``fps`` is never used for a frame's
    time.
    """

    def __init__(self, path: str):
        import av  # lazy: keep the module importable without `av` installed
        self._av = av
        self._path = str(path)
        self._container = av.open(self._path)
        if not self._container.streams.video:
            self._container.close()
            raise RuntimeError(f"No video stream found in {path!r}")
        self._stream = self._container.streams.video[0]

        # --- normalisation base, captured ONCE at open (before any seek) ---
        # `start_pts` is the pts value that corresponds to t = 0 ms.  Prefer
        # the container's own start_time when it is a usable non-negative
        # value; otherwise fall back to the pts of the first decoded frame.
        # It is computed here (not lazily on the first read) so a later
        # seek_ms() can never overwrite it (per the design notes).
        start_pts = self._stream.start_time
        if start_pts is None or start_pts < 0:
            first = self._decode_next()
            if first is None:
                raise RuntimeError(
                    f"Could not decode the first frame of {path!r}")
            start_pts = first[0]
            self._last = first
        self._start_pts = int(start_pts)

        # --- frame size / duration / count (container hints, best effort) ---
        self._width = int(self._stream.width or 0)
        self._height = int(self._stream.height or 0)
        self._duration_s: Optional[float] = None
        dur_tb = self._stream.time_base
        if self._stream.duration is not None and dur_tb is not None:
            self._duration_s = float(self._stream.duration) * float(dur_tb)
        elif self._container.duration is not None:
            self._duration_s = float(self._container.duration) / 1_000_000.0
        self._frame_count = 0
        if self._stream.frames:
            self._frame_count = int(self._stream.frames)
        elif self._duration_s is not None:
            # nominal fallback only (the per-frame time never uses fps)
            fps = float(self._stream.average_rate or 0.0)
            if fps > 0:
                self._frame_count = int(round(self._duration_s * fps))

        self._last: Optional[Tuple[int, np.ndarray]] = None
        self._closed = False

    # ── low-level helpers ────────────────────────────────────────────────
    def _decode_next(self) -> Optional[Tuple[int, np.ndarray]]:
        """Decode the next frame; return (pts, BGR ndarray) or None at EOF.

        PyAV raises ``av.error.EOFError`` when the stream is exhausted (e.g.
        after a seek past the end), so we catch it and return None instead of
        propagating.  A genuine decode error (corrupt packet) is re-raised.
        """
        _AvEOF = self._av.error.EOFError
        try:
            for frame in self._container.decode(self._stream):
                if frame.pts is None:
                    continue
                try:
                    arr = frame.to_ndarray(format="bgr24")
                except Exception:
                    continue
                return (int(frame.pts), np.ascontiguousarray(arr))
        except _AvEOF:
            return None
        return None

    def _time_ms(self, pts: int) -> float:
        """REAL presentation time of `pts`, in ms (start_pts normalised)."""
        tb = self._stream.time_base
        if tb is None:
            return 0.0
        return (pts - self._start_pts) * float(tb) * 1000.0

    def _seek_to_ms(self, ms: float) -> None:
        """Position the decoder so the NEXT ``_decode_next`` is the first
        frame at time >= ms (the container seek is keyframe-accurate, so we
        flush the codec buffers and walk forward frame by frame)."""
        tb = self._stream.time_base
        if tb is None:
            return
        # PyAV: with stream=<stream> the offset is in stream.time_base units.
        offset = int(round(ms / 1000.0 / float(tb)))
        try:
            self._container.seek(offset, stream=self._stream,
                                backward=True, any_frame=False)
        except Exception:
            # some containers reject the seek; decode from the current point
            pass
        # drop any stale buffered frames left by a previous decode position
        try:
            self._stream.codec_context.flush_buffers()
        except Exception:
            pass

    # ── BaseVideoReader interface ────────────────────────────────────────
    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        if self._closed:
            return None
        nxt = self._decode_next()
        if nxt is None:
            return None
        self._last = nxt
        return (nxt[1], self._time_ms(nxt[0]))

    def seek_ms(self, ms: float) -> Optional[Tuple[np.ndarray, float]]:
        if self._closed:
            return None
        ms = max(0.0, float(ms))
        self._seek_to_ms(ms)
        tb = self._stream.time_base
        target_pts = self._start_pts + int(round(ms / 1000.0 / float(tb))) \
            if tb is not None else 0
        for _ in range(100_000):  # hard cap: enough for any safety margin
            nxt = self._decode_next()
            if nxt is None:
                return None
            if nxt[0] >= target_pts:
                self._last = nxt
                return (nxt[1], self._time_ms(nxt[0]))
        return None

    def step(self, n: int) -> Optional[Tuple[np.ndarray, float]]:
        if self._closed:
            return None
        n = int(n)
        if n == 0:
            if self._last is None:
                return self.read()
            return (self._last[1], self._time_ms(self._last[0]))
        if n > 0:
            for _ in range(n):
                cur = self.read()
                if cur is None:
                    return None
            return cur
        # n < 0: time-based backward seek (same mechanism as seek_to_time_exact).
        # Compute the target time as current - |n| frames worth of time using
        # the stream's average rate (VFR-safe: the actual landing point is
        # determined by keyframe seek + forward decode, never by frame index).
        avg_fps = float(self._stream.average_rate) if self._stream.average_rate else 30.0
        target_ms = max(0.0, self.current_time_ms - (abs(n) / avg_fps) * 1000.0)
        return self.seek_ms(target_ms)

    def reset(self) -> Optional[Tuple[np.ndarray, float]]:
        self._seek_to_ms(0.0)
        return self.read()

    def close(self) -> None:
        if not self._closed:
            try:
                self._container.close()
            except Exception:
                pass
            self._closed = True

    # ── properties ───────────────────────────────────────────────────────
    @property
    def current_time_ms(self) -> float:
        if self._last is None:
            return 0.0
        return self._time_ms(self._last[0])

    @property
    def width(self) -> int:
        if self._last is not None:
            h, w = self._last[1].shape[:2]
            return int(w)
        return self._width

    @property
    def height(self) -> int:
        if self._last is not None:
            h, w = self._last[1].shape[:2]
            return int(h)
        return self._height

    @property
    def duration_s(self) -> Optional[float]:
        return self._duration_s

    @property
    def frame_count(self) -> int:
        return self._frame_count


def open_video_reader(path: str) -> BaseVideoReader:
    """Open ``path`` and return a :class:`BaseVideoReader`.

    Currently the only backend is :class:`PyAVVideoReader`; this factory is
    the single place to add / swap a different backend in the future.
    """
    return PyAVVideoReader(path)
