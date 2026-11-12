"""Live "LLM thinking" progress window for the Ollama generation mode.

Design goals
============
1. **Open BEFORE the LLM call** (so the user sees the crop + settings
   + a ticking seconds counter) and **stay open until the call returns**
   (or the user presses Q to abandon).
2. **Window on the MAIN thread, LLM call on a WORKER thread**:
   ``cv2.waitKey`` is reliable on Windows only on the thread that
   created the window; the Ollama ``generate()`` is a blocking httpx
   call — running it on a daemon thread lets the window keep ticking
   even while the model is thinking.
3. **ETA from history**: on every call we persist (images, elapsed_s)
   into ``llm_session_stats.json`` (see ``core/postpass/llm_stats.py``);
   the window shows a live ``ETA = avg_image_s * N - already_elapsed``.
4. **Abort semantics**: on Q/Esc the worker is signalled (via a
   ``threading.Event``) to STOP after the CURRENT chunk completes.
   The worker then returns the partial track accumulated so far
   (all previously answered chunks), and ``funscript_gen`` proceeds
   to RDP + ``build_funscript`` with that partial track.
   If ZERO chunks have completed, the process exits with code 2
   (nothing to import).

Usage (from ``funscript_gen._run``)::

    from core.interface.llm_thinking import ThinkingWindow
    from core.postpass.llm_gen import LLMGenParams, generate_track

    roi = select_llm_roi(video, start_s, fps)          # existing window
    if roi is None: sys.exit(2)

    llm_params = LLMGenParams(model=..., clip_s=..., ...)
    llm_params.roi = roi

    tw = ThinkingWindow(
        frame_bgr=letterboxed_crop,         # what the model sees
        model=llm_params.model,
        n_images_target=n_send,
        video_t_start=start_s,
        video_t_end=stop_s,
        task_kwargs=dict(
            video_path=video, start_s=start_s, fps=fps,
            p=llm_params, on_phase=None,  # window handles it
        ),
    )
    aborted, error, track = tw.run()
    if aborted: sys.exit(2)
    if error: raise RuntimeError(error)
    # ... continue to RDP + build_funscript
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.interface.preview import PREVIEW_H, PREVIEW_W, setup_window
from core.postpass.llm_stats import estimate_eta, format_eta, record_call

def _install_thread_exc_hook() -> None:
    """Make exceptions in daemon threads VISIBLE in the log file.
    Python's default just prints to stderr, which under OFS is a pipe
    nobody drains -> the exception is silently lost and the window sits
    in phase=starting forever.  Install once per process."""
    import sys as _sys
    import threading as _th
    if getattr(_sys, "_llm_think_hook_installed", False):
        return

    def _hook(args):
        import traceback as _tb
        exc = args.exc_value
        text = ("[THREAD-EXC] %r in thread %r\n"
                % (exc, args.thread.name if args.thread else "?"))
        if args.exc_type is not None:
            text += "".join(_tb.format_exception(args.exc_type,
                                                 exc, args.exc_traceback))
        log_file = None
        try:
            from core import log_file_path
            log_file = str(log_file_path())
        except Exception:
            pass
        if not log_file:
            import os as _os
            log_file = _os.path.normpath(_os.path.join(
                _os.path.dirname(__file__), "..", "..",
                "funscript_gen.stderr.log"))
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass
        try:
            print(text, file=_sys.stderr, flush=True)
        except Exception:
            pass

    _sys.excepthook = _hook
    _th.excepthook = _hook
    _sys._llm_think_hook_installed = True
    try:
        from core.logging import log as _log
        _log("[boot] installed global thread-except hook")
    except Exception:
        pass

_install_thread_exc_hook()


_WIN = "LLM thinking (Q=abandon)"

# ────────────────────────────────────────────────────────────────
# Bulletproof logger: tries core.logging first, falls back to a
# direct file write if ANYTHING about the core logging package
# fails (import error, missing dependency, bad file handle).  In
# OFS the stderr pipe is on a closed parent end, so the FILE is
# the only reliable channel.
# ────────────────────────────────────────────────────────────────
def _hlog(msg: str) -> None:
    """Write msg to funscript_gen.stderr.log (best effort, never
    raises).  Used for every step in the worker-thread lifecycle
    so that when it dies we see exactly where."""
    text = "[" + time.strftime("%H:%M:%S") + "] " + str(msg) + "\n"
    try:
        from core.logging import log as _cl
        _cl(msg)
        return
    except Exception:
        pass
    log_file = None
    try:
        from core import log_file_path
        log_file = log_file_path()
    except Exception:
        pass
    if not log_file:
        import os as _os
        log_file = _os.path.join(_os.path.dirname(__file__),
                                 "..", "..", "funscript_gen.stderr.log")
    try:
        fh = open(log_file, "a", encoding="utf-8", buffering=1)
        try:
            fh.write(text)
            fh.flush()
        finally:
            fh.close()
    except Exception:
        pass
    try:
        import sys as _sys
        print(text, file=_sys.stderr, flush=True)
    except Exception:
        pass


def _fmt_mmss(t_s: float) -> str:
    if t_s is None or t_s < 0.0:
        t_s = 0.0
    h, rem = divmod(int(t_s), 3600)
    m, sc = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sc:02d}"
    return f"{m}:{sc:02d}"


@dataclass
class _SharedState:
    """Shared between the window thread and the worker thread."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    phase: str = "starting"        # starting / sampling / thinking / done / error
    target_images: int = 0
    sampled_so_far: int = 0
    model: str = ""
    num_ctx: int = 0
    thumb: int = 0
    error: str = ""
    track: Optional[List[Tuple[float, float]]] = None
    aborted: bool = False
    # chunked pipeline info (llm_gen reads in clip_s chunks, one LLM call each)
    n_chunks: int = 0
    cur_chunk: int = 0
    frames_done: int = 0
    total_frames: int = 0
    # live preview of the first frame of the current chunk, at its FINAL
    # (sent-to-model) resolution: ndarray + (w, h).  Updated by the worker
    # on each "thinking" phase; drawn full-frame by _render() so the user
    # sees EXACTLY what the VLM receives.
    preview: Optional[np.ndarray] = None
    preview_size: Optional[Tuple[int, int]] = None
    _finished: bool = False
    _started_wall: float = field(default_factory=time.time)
    _elapsed_final: float = 0.0
    _v_start: float = 0.0
    _v_end: float = 0.0

    @property
    def finished(self) -> bool:
        with self.lock:
            return self._finished

    def elapsed(self) -> float:
        with self.lock:
            if self._finished:
                return self._elapsed_final
            return max(0.0, time.time() - self._started_wall)


class ThinkingWindow:
    """A progress / ETA window with an ``abandoned`` / ``aborted`` path.

    Runs the LLM task (a 1-arg callable) on a daemon worker thread and
    shows a live ``cv2`` window on the main thread.  Returns a tuple
    ``(aborted: bool, error: Optional[str], track: Optional[list])``;
    the caller should ``sys.exit(2)`` when aborted.
    """
    def __init__(self,
                 frame_bgr: np.ndarray,
                 model: str,
                 n_images_target: int,
                 video_t_start: float,
                 video_t_end: float,
                 task_callable: Callable[..., List[Tuple[float, float]]],
                 task_args: Tuple = (),
                 task_kwargs: Optional[Dict[str, Any]] = None):
        self._base = frame_bgr.copy()
        self._state = _SharedState()
        self._state.model = str(model or "")
        self._state.target_images = int(n_images_target or 0)
        self._state._v_start = float(video_t_start or 0.0)
        self._state._v_end = float(video_t_end or 0.0)
        # extract model ctx / thumb from task_kwargs if present
        p = (task_kwargs or {}).get("p")
        if p is not None:
            self._state.num_ctx = int(getattr(p, "num_ctx", 0) or 0)
            self._state.thumb = int(getattr(p, "thumb", 0) or 0)
        self._task_callable = task_callable
        self._task_args = tuple(task_args or ())
        self._task_kwargs = dict(task_kwargs or {})
        self._wthr: Optional[threading.Thread] = None   # the actual Thread object (NOT the method _worker)
        # Aborting is cooperative: the worker checks self._abort_ev.is_set()
        # BETWEEN chunks and returns the partial track accumulated so far.
        self._abort_ev = threading.Event()

    # ── worker (background thread) ──────────────────────────
    def _worker(self) -> None:
        """Worker thread: runs the LLM task and reports via self._state."""
        # ABSOLUTELY FIRST ACTION: a raw file write with NO core.* import
        # and NO _hlog call.  If this line does not land in the file, the
        # thread simply never got to run (GIL starvation / never scheduled
        # / died in the interpreter before the body).  If it DOES land but
        # the next line does not, the fault is in the statement between
        # them.
        def _raw(msg):
            try:
                import os as _os
                lp = _os.path.normpath(_os.path.join(
                    _os.path.dirname(__file__), "..", "..",
                    "funscript_gen.stderr.log"))
                with open(lp, "a", encoding="utf-8") as fh:
                    fh.write("[" + time.strftime("%H:%M:%S") + "] "
                             + str(msg) + "\n")
                    fh.flush()
            except Exception:
                pass
        import os as _os
        _raw("[worker] === RAW FIRST LINE (thread id=%d, pid=%d) ==="
             % (threading.get_ident(), _os.getpid()))
        import core.logging as _clog_mod
        _raw("[worker] imported core.logging (module=%s)"
             % (getattr(_clog_mod, "__file__", "?"),))
        _cl = _clog_mod.log
        _cl("[worker] thread entered id=%d name=%s" % (
            threading.get_ident(), threading.current_thread().name))
        t0 = time.time()
        success = False
        try:
            _hlog("[worker] calling %s (args=%d kwargs=%d)"
                 % (getattr(self._task_callable, "__qualname__", "?"),
                    len(self._task_args), len(self._task_kwargs)))
            result = self._task_callable(*self._task_args,
                                         on_phase=self._on_phase,
                                         should_abort=self._abort_ev.is_set,
                                         **self._task_kwargs)
            _hlog("[worker] task returned, result len=%s"
                 % (len(result) if result is not None else "None"))
            # generate_track returns list-of-(t,pos)
            track = list(result)
            success = True
            with self._state.lock:
                self._state.track = track
                self._state.phase = "done"
        except BaseException as e:
            import traceback as _tb
            _hlog("[worker] EXCEPTION  %s: %r" % (type(e).__name__, e))
            _hlog("[worker] ---- traceback ----")
            try:
                for line in _tb.format_tb(e.__traceback__):
                    _hlog("[worker]   " + line.rstrip())
            except Exception:
                pass
            _hlog("[worker] ---- end traceback ----")
            with self._state.lock:
                self._state.phase = "error"
                self._state.error = str(e) or repr(e)
        finally:
            _hlog("[worker] finally: persisting stats, elapsed=%.1fs"
                 % max(0.0, time.time() - t0))
            elapsed = max(0.0, time.time() - t0)
            with self._state.lock:
                self._state._elapsed_final = elapsed
                self._state._finished = True
            # Persist to stats (for future ETA); best-effort, never raises.
            n_img = int(self._state.target_images or 0)
            if n_img > 0 and self._state.model:
                record_call(
                    model=self._state.model,
                    images=n_img,
                    ctx=int(self._state.num_ctx or 0),
                    thumb=int(self._state.thumb or 0),
                    elapsed_s=elapsed,
                    success=success,
                )

    # ── callback the task can use to update the window ──────
    def _on_phase(self, phase: str, **kw: Any) -> None:
        _hlog("[phase] worker -> window: phase=%r kw=%r" % (phase, kw))
        with self._state.lock:
            self._state.phase = str(phase)
            if "sampled" in kw:
                try:
                    self._state.sampled_so_far = int(kw["sampled"])
                except Exception:
                    pass
            if "processed" in kw:
                try:
                    self._state.frames_done = int(kw["processed"])
                except Exception:
                    pass
            if "total_frames" in kw:
                try:
                    self._state.total_frames = int(kw["total_frames"])
                except Exception:
                    pass
            if "chunk" in kw:
                try:
                    self._state.cur_chunk = int(kw["chunk"])
                except Exception:
                    pass
            if "n_chunks" in kw:
               try:
                   self._state.n_chunks = int(kw["n_chunks"])
               except Exception:
                   pass
            if "preview" in kw and kw["preview"] is not None:
               self._state.preview = kw["preview"]
               self._state.preview_size = kw.get("preview_size")

    # ── render the current frame ────────────────────────────
    def _render(self) -> np.ndarray:
        # Snapshot shared state ONCE (incl. optional live preview), then
        # render.  When a preview is present (the first frame of the current
        # chunk, at its FINAL sent-to-model resolution), we swap it in for
        # the base frame and letterbox it into 960x540 - this is what the
        # user sees as "what the model sees".
        with self._state.lock:
            phase = self._state.phase
            model = self._state.model
            target = self._state.target_images
            sampled = self._state.sampled_so_far
            v_start = self._state._v_start
            v_end = self._state._v_end
            num_ctx = self._state.num_ctx
            thumb = self._state.thumb
            error = self._state.error
            n_chunks = self._state.n_chunks
            cur_chunk = self._state.cur_chunk
            frames_done = self._state.frames_done
            total_frames = self._state.total_frames
            prev = self._state.preview
            prev_wh = self._state.preview_size

        # Live preview base (letterboxed if a preview is available).
        live_preview = False
        if (phase in ("reading", "thinking") and prev is not None
                and prev_wh is not None and prev.size > 0):
            try:
                from core.interface.preview import fit as _fit
                frame, _s, _x0, _y0 = _fit(prev)
                live_preview = True
            except Exception:
                frame = self._base.copy()
        else:
            frame = self._base.copy()
        elapsed = self._state.elapsed()
        eta = estimate_eta(target, elapsed) if target else None
        eta_txt = format_eta(eta) if eta is not None else "(no prior runs yet)"

        # header bar
        color = (60, 200, 255) if phase in ("starting", "reading", "thinking") \
                else (60, 255, 120) if phase == "done" else (80, 80, 255)
        cv2.rectangle(frame, (0, 0), (PREVIEW_W, 26), (20, 20, 30), cv2.FILLED)
        head = f" OLLAMA LLM: phase={phase}   "
        if phase == "reading" and total_frames:
            head += f"reading {frames_done}/{total_frames}f "
        if phase == "thinking" and (n_chunks or cur_chunk):
            head += f"chunk {cur_chunk}/{n_chunks or '?'} "
        head += f"elapsed {elapsed:6.1f}s"
        cv2.putText(frame, head, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # info block on the left side
        lines = [
            f"model       : {model}"   if len(model) <= 55 else f"model       : {model[:52]}…",
            f"frames      : {total_frames or target} total  (context {num_ctx:,}, thumb {thumb}px)",
            f"chunks      : {n_chunks} call(s), clip_s windows  "
            f"(video {_fmt_mmss(v_start)} .. {_fmt_mmss(v_end)}, "
            f"{max(0.0, v_end - v_start):.1f}s)",
            f"progress    : chunk {cur_chunk}/{n_chunks}  "
            f"frames {frames_done}/{total_frames}",
            f"elapsed     : {elapsed:.1f}s",
            f"ETA         : {eta_txt}",
        ]
        if live_preview and prev_wh is not None:
            lines.append(f"preview     : first frame of current chunk  "
                         f"{prev_wh[0]}x{prev_wh[1]} px (as sent to LLM)")
        else:
            lines.append("preview     : (waiting for first chunk ...)")
        if error:
            lines.append("ERROR     : " + (error if len(error) <= 60 else f"{error[:55]}…"))
        y = 44
        for line in lines:
            # word-wrap if too long
            cv2.putText(frame, line, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1,
                        cv2.LINE_AA)
            y += 24

        # hint bar
        hint = "Q / Esc = stop after current chunk (partial track is kept)"
        if phase == "done":
            hint = "Ollama responded — closing..."
        elif phase == "error":
            hint = "ERROR — see stderr log for details"
        cv2.rectangle(frame, (0, PREVIEW_H - 28), (PREVIEW_W, PREVIEW_H),
                      (25, 25, 35), cv2.FILLED)
        hcolor = (255, 200, 90) if phase in ("starting", "reading", "thinking") \
            else (255, 90, 90) if phase == "error" else (90, 255, 140)
        cv2.putText(frame, hint, (12, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hcolor, 1, cv2.LINE_AA)

        # thinking dots on the right side (ticking)
        if phase in ("starting", "reading", "thinking"):
            dots = "." * (int(elapsed) % 4)
            cv2.putText(frame, f" {dots} ", (PREVIEW_W - 70, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return frame

    # ── main entry point ────────────────────────────────────
    def run(self) -> Tuple[bool, Optional[str],
                           Optional[List[Tuple[float, float]]]]:
        """Block until the task finishes or the user presses Q.

        Returns ``(aborted, error, track)``.  ``aborted=True`` means
        the user quit mid-call (the worker thread keeps running as a
        daemon thread and may append to the stats file later);
        ``error`` is set if the worker raised; ``track`` is the list
        of ``(t_sec, pos)`` pairs on success, else ``None``.
        """
        _hlog("[run] enter run(); target_images=%s model=%r"
             % (self._state.target_images, self._state.model))
        # 1. open the window (main thread; reliable key input)
        _hlog("[run] calling setup_window ...")
        setup_window(_WIN)
        _hlog("[run] setup_window OK")
        # 2. launch the worker thread
        _hlog("[run] creating worker thread ...")
        # IMPORTANT: target is the METHOD self._worker (bound on the class);
        # the Thread object itself lives on self._wthr so it never shadows
        # the method name (a self._worker attribute here would make
        # `target=self._worker` evaluate to None and the thread would die
        # with "NoneType not callable" before its first line runs).
        self._wthr = threading.Thread(target=self._worker, daemon=True,
                                      name="llm-think-worker")
        _hlog("[run] calling thread.start() ...")
        try:
            self._wthr.start()
        except BaseException as e:
            _hlog("[run] thread.start() FAILED: %r" % e)
            raise
        _hlog("[run] thread.start() returned tid=%s -> render loop"
             % (self._wthr.ident,))
        # 3. render loop
        aborted = False
        hb = 0
        _t_loop0 = time.time()
        while True:
            # main-thread watchdog: every ~2s, if the worker has not
            # reported "thread entered", say so.  This proves the main
            # thread is alive while the worker is (or is not) running.
            hb += 1
            if hb % 50 == 0:
                with self._state.lock:
                    _ph = self._state.phase
                _hlog("[watchdog] t=%.1fs main alive; phase=%s (worker "
                      "may be dead or in-flight)" %
                      (time.time() - _t_loop0, _ph))
            frame = self._render()
            cv2.imshow(_WIN, frame)
            k = cv2.waitKeyEx(40)
            if k in (ord('q'), ord('Q'), 27):  # 27 = Esc
                _hlog("[run] Q/Esc pressed - signalling worker to stop "
                      "after current chunk")
                self._abort_ev.set()
                aborted = True
                break
            if self._state.finished:
                # give the frame one extra render so the final state is visible
                cv2.imshow(_WIN, self._render())
                cv2.waitKeyEx(120)
                break
        # When the user pressed Q, wait for the worker thread to finish the
        # in-flight LLM call (it will stop after the CURRENT chunk completes
        # and see the abort_event → break → return partial track).
        # Timeout: 120s should be more than enough for a single chunk.
        if aborted and self._wthr is not None and self._wthr.is_alive():
            _hlog("[run] waiting for worker to finish current chunk (120s max) ...")
            self._wthr.join(timeout=120.0)
            if self._wthr.is_alive():
                _hlog("[run] WARNING: worker still alive after 120s join timeout; "
                      "returning whatever track is available so far")
            else:
                _hlog("[run] worker finished after abort; track="
                      f"{'None' if self._state.track is None else str(len(self._state.track))+' pts'}")
        cv2.destroyWindow(_WIN)
        with self._state.lock:
            error = self._state.error
            track = self._state.track
        return aborted, error, track


__all__ = ["ThinkingWindow"]
