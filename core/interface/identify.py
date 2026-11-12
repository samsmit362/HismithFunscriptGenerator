"""
First-frame partner selection.

Three modes:
  * pose trackers (yolo / mediapipe) - pick the ACTIVE and the PASSIVE
    person among the detections (keys A / P, Enter to confirm);
  * groundingdino - the window shows BOTH the detected person/person
    boxes AND the real detection ROI boxes of the current target prompt
    (same green color), plus the tracked point p; the user presses +
    holds LMB to place the tracked point p INSIDE the detection box
    (as percents, default = box center = (50, 50)); R re-runs detection
    (re-draw), Enter/Space confirms (stores the p placement on the
    tracker), Q aborts.
  * SOT trackers - NO partner selection (only one manually drawn object):
    press + hold LMB to draw the box around the exact body zone to track
    (release = box is fixed), Enter/Space = confirm (initializes the tracker
    with that ROI), Q = abort.
"""
import sys
from typing import Dict, List, Optional, Tuple

import cv2

from core.detectors.box import box_center
from core.detectors.person_tracker import PersonTracker
from core.interface.preview import PREVIEW_H, PREVIEW_W, draw_skeleton, fit, setup_window
from core.interface.win32 import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, key_held
from core.tracking.scene_warp import (ANGLE_STEP_DEG, PAN_CLAMP_PX, PAN_STEP_PX,
                                      SceneTransform, ZOOM_LEVELS)
from core.tracking.seek import seek_to_time_exact
from core.types import Person


# ── stateful mouse drag (OpenCV has no waitMouse; tiny stateful helper) ──
# DOWN / MOVE / UP are edge EVENTS, so a poller that reads "the last event"
# once per loop tick can miss the LBUTTONDOWN when a following MOUSEMOVE
# overwrites the slot before the ~30 ms waitKeyEx wakes up - the drag then
# never starts.  Stateful transitions instead: the callback records
# down / anchor / cursor; the loop reads the STATE every tick and
# recomputes the live box while the button is held.  On LBUTTONUP the box
# simply stops being updated = the selection is FIXED at its last position.
class _DragSlot:
    """One independently draggable ROI slot (either p or ps).  The two slots
    live INSIDE a single _DualDragState so the same mouse callback can route
    each press to the right slot based on the Ctrl modifier at press time -
    LMB = the tracked point p, Ctrl+LMB = the optional static part ps."""
    __slots__ = ("down", "anchor", "px", "py")

    def __init__(self):
        self.down = False
        self.anchor = None  # (x, y) in VIEW px - where the press happened
        self.px = self.py = 0  # last cursor position in VIEW px

    def reset(self) -> None:
        self.down = False
        self.anchor = None


class _DualDragState:
    """Dual-ROI drag state for one window.

    *  ``p``   - the primary (required) SOT ROI.  LMB (no modifier) draws it;
    *  ``ps``  - the second, OPTIONAL "static part" ROI.  Ctrl+LMB draws it.

    The press EDGE event routes to exactly one slot (capturing the cursor and
    anchor), MOUSEMOVE refreshes the cursor of the currently-held slot(s), and
    the release only clears that slot's ``down`` flag - so a press+release
    simply stops the refresh and the box stays fixed at its last position.
    Reading .down == False in the loop = the selection has been FIXed in the
    last press session.
    """
    __slots__ = ("p", "ps")

    def __init__(self):
        self.p = _DragSlot()
        self.ps = _DragSlot()

    def reset(self) -> None:
        self.p.reset()
        self.ps.reset()


_drag_registry: Dict[str, _DualDragState] = {}


def _drag_for(win: str) -> _DualDragState:
    """Mouse-drag state for the window (callback installed once).
    Coordinates are in VIEW pixels of the last drawn image; the caller
    applies its own letterbox scale (fit()).  The state exposes TWO slots:
    ``st.p``  = the primary (required) ROI, drawn with plain LMB;
    ``st.ps`` = the optional "static part" ROI, drawn with Ctrl+LMB.
    Each slot is fully independent (own down/anchor/cursor) so the two
    boxes can be selected in any order and re-dragged independently."""
    st = _drag_registry.get(win)
    if st is None:
        st = _DualDragState()

        def cb(ev, x, y, flags, _p, st=st):
            from core.interface.win32 import KEY_CTRL, key_held
            # OpenCV normalizes `flags` around press/release events: bit 0
            # (left button) is SET on LBUTTONDOWN and CLEARED on LBUTTONUP,
            # so only the DOWN event needs the bit check (to ignore the
            # middle/right button) - the UP event arrives only for its own
            # button.  Reading bit 0 on UP would wrongly say "released".
            if ev == cv2.EVENT_LBUTTONDOWN and (flags & 1):
                # Route to the correct slot: Ctrl held -> ps, else p.
                # The slot for this press is now fixed (won't switch until
                # a new press fires).
                slot = st.ps if key_held(KEY_CTRL) else st.p
                # If the OTHER slot was still "down" (e.g. a stale state from
                # a previous drag that never released cleanly), force-release
                # it first so only one box is refreshed at a time.
                if slot is st.ps:
                    st.p.reset()
                else:
                    st.ps.reset()
                slot.down = True
                slot.anchor = (x, y)
                # the cursor is at the press point - sync it, otherwise a
                # plain click (press+release, no movement) would compute a
                # box against the STALE cursor left from the previous drag
                slot.px, slot.py = x, y
            elif ev == cv2.EVENT_MOUSEMOVE:
                # refresh the cursor for whichever slot is being held; if
                # both were somehow down (should not happen after the press
                # above reset the other), refresh BOTH to stay in sync.
                if st.p.down:
                    st.p.px, st.p.py = x, y
                if st.ps.down:
                    st.ps.px, st.ps.py = x, y
            elif ev == cv2.EVENT_LBUTTONUP:
                # release: record the release position EXPLICITLY (not via a
                # preceding MOUSEMOVE) so the caller can tell "a click
                # without movement" apart from a real drag - if no MOUSEMOVE
                # fired between DOWN and UP, slot.px/.py would still hold the
                # STALE cursor from the previous drag.
                # Only the slot(s) that were DOWN on DOWN event are
                # affected: if both flags happened to be True, release both.
                if st.p.down:
                    st.p.px, st.p.py = x, y
                    st.p.down = False
                if st.ps.down:
                    st.ps.px, st.ps.py = x, y
                    st.ps.down = False

        _drag_registry[win] = st
        cv2.setMouseCallback(win, cb)
    return st


def _drag_reset(win: str) -> None:
    """Clear BOTH slots (used by 'R' = clear ROI, so a half-drag of either
    ROI does not keep refreshing an old box)."""
    st = _drag_registry.get(win)
    if st is not None:
        st.reset()


def _setup_scene(frame: np.ndarray,
                 tracker: PersonTracker,
                 skip: bool = False) -> SceneTransform:
    """SOT step 1: scene setup window (zoom / rotate / pan / optional ROI crop).

    When ``skip=True`` (e.g. ``--no-scene-roi``) an identity SceneTransform
    is returned immediately without opening a window.

    Keys in the window:
        +/-        zoom in / out  (1x -> 2x -> 3x)
        [ / ]      rotate +15 / -15 deg  (clamped -180..180)
        Left/Right pan left/right 10 px (in output space)
        Up/Down    pan up/down  10 px  (in output space)
        R          reset all settings to identity
        Q / Esc    abort  (exit code 2)
        Enter/Space  confirm
        LMB drag   draw the scene-ROI rectangle (in OUTPUT/warped space);
                   release fixes it, re-drag to re-select.

    The result is stored on ``tracker.scene``; downstream code (calibrate,
    tracking_stream, SOT tracker) reads ``tracker.scene`` and calls
    ``tracker.scene.apply(raw_frame)`` on every frame it processes.
    """
    if skip:
        sc = SceneTransform()
        tracker.scene = sc
        print("[scene] --no-scene-roi: identity (full frame, no transform)")
        return sc

    # seed from the tracker's existing scene (R-reselect path keeps it)
    sc = SceneTransform()
    prev = getattr(tracker, "scene", None)
    if prev is not None:
        sc = SceneTransform(zoom=prev.zoom, angle_deg=prev.angle_deg,
                            pan_dx=prev.pan_dx, pan_dy=prev.pan_dy,
                            scene_roi=prev.scene_roi)

    H, W = frame.shape[:2]
    win = ("SOT scene: +/- zoom  [ ] rotate  arrows pan10px  "
           "LMB drag=ROI  R reset  Enter=OK  Q=abort")
    setup_window(win)
    drag = _drag_for(win)
    _drag_reset(win)
    roi_out = None  # (x, y, w, h) in OUTPUT (pre-crop warp) pixels

    def _warp_now() -> np.ndarray:
        """Return the pre-crop warped output (always H x W)."""
        if (sc.zoom == 1.0 and sc.angle_deg == 0.0
                and sc.pan_dx == 0.0 and sc.pan_dy == 0.0):
            return frame
        from core.tracking.scene_warp import _warp_view
        return _warp_view(frame, sc.zoom, sc.angle_deg,
                          sc.pan_dx, sc.pan_dy)

    def _to_out(mx, my, scale_, x0, y0):
        return (max(0, min(W, int((mx - x0) / scale_))),
                max(0, min(H, int((my - y0) / scale_))))

    while True:
        warped = _warp_now()
        view, scale, x0, y0 = fit(warped)

        # live scene-ROI drag (OUTPUT = warp coordinates)
        if drag.p.down and drag.p.anchor is not None:
            ax, ay = _to_out(drag.p.anchor[0], drag.p.anchor[1], scale, x0, y0)
            cx, cy = _to_out(drag.p.px, drag.p.py, scale, x0, y0)
            roi_out = (min(ax, cx), min(ay, cy),
                       max(1, abs(cx - ax)), max(1, abs(cy - ay)))
        if roi_out is not None:
            rx, ry, rw, rh = roi_out
            cv2.rectangle(view,
                          (int(rx * scale) + x0, int(ry * scale) + y0),
                          (int((rx + rw) * scale) + x0, int((ry + rh) * scale) + y0),
                          (255, 200, 0), 3)

        # top status bar (green)
        stat = (f"zoom={sc.zoom:g}x  angle={sc.angle_deg:+.0f}deg  "
                f"pan=({int(sc.pan_dx)},{int(sc.pan_dy)})"
                + (f"  ROI={roi_out[2]}x{roi_out[3]}@({roi_out[0]},{roi_out[1]})"
                   if roi_out else "  ROI=full"))
        (sw_, _sh_2), _ = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.putText(view, stat, ((PREVIEW_W - sw_) // 2, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # bottom hint bar
        hint = ("+/- zoom   [ ] rotate 15deg   arrows pan 10px   "
                "LMB drag = scene ROI   R reset   Enter OK   Q abort")
        (tw, th_), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
        cv2.putText(view, hint, ((PREVIEW_W - tw) // 2, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

        cv2.imshow(win, view)
        k = cv2.waitKeyEx(30)

        if k in (ord('q'), ord('Q'), 27):
            cv2.destroyAllWindows()
            print("[scene] aborted by user")
            sys.exit(2)
        elif k in (ord('+'), ord('=')):
            i = 0
            try:
                i = ZOOM_LEVELS.index(sc.zoom)
            except ValueError:
                pass
            if i < len(ZOOM_LEVELS) - 1:
                sc.zoom = ZOOM_LEVELS[i + 1]
        elif k in (ord('-'), ord('_')):
            i = 0
            try:
                i = ZOOM_LEVELS.index(sc.zoom)
            except ValueError:
                pass
            if i > 0:
                sc.zoom = ZOOM_LEVELS[i - 1]
        elif k in (ord(']'),):
            sc.angle_deg = max(-180.0, sc.angle_deg - ANGLE_STEP_DEG)
        elif k in (ord('['),):
            sc.angle_deg = min(180.0, sc.angle_deg + ANGLE_STEP_DEG)
        elif k in (KEY_LEFT,):
            sc.pan_dx = max(-PAN_CLAMP_PX, sc.pan_dx - PAN_STEP_PX)
        elif k in (KEY_RIGHT,):
            sc.pan_dx = min(PAN_CLAMP_PX, sc.pan_dx + PAN_STEP_PX)
        elif k in (KEY_UP,):
            sc.pan_dy = max(-PAN_CLAMP_PX, sc.pan_dy - PAN_STEP_PX)
        elif k in (KEY_DOWN,):
            sc.pan_dy = min(PAN_CLAMP_PX, sc.pan_dy + PAN_STEP_PX)
        elif k in (ord('r'), ord('R')):
            sc.reset()
            roi_out = None
            _drag_reset(win)
        elif k in (13, 32):  # Enter / Space = confirm
            sc.scene_roi = roi_out
            tracker.scene = sc
            print(f"[scene] confirmed: {sc.describe()}")
            cv2.destroyAllWindows()
            return sc
def _identify_sot(video_path: str, start_s: float, fps: float,
                  tracker: PersonTracker,
                  current_video_frame: Optional[int] = None,
                  no_scene: bool = False
                  ) -> Tuple[Person, Optional[Person]]:
    """SOT mode, two-step manual flow.

    Step 1 (``_setup_scene``; skipped when ``no_scene=True``): set up an
    optional scene transform (zoom 1x/2x/3x, rotate 15-deg steps, pan 10 px
    steps, and an optional scene-ROI crop) BEFORE the SOT tracking ROI is
    drawn.  The result is stored on ``tracker.scene`` — every downstream
    consumer (calibrate, tracking_stream, SOT tracker init) then runs on
    ``tracker.scene.apply(raw_frame)`` and no coordinate remap is needed.

    Step 2: press + hold LMB to draw the ROI box around the point to track
    (release = the box is fixed); Enter/Space then calls tracker.init() on
    the SCENE-SEEDED warped frame, so SOT internal state is consistent with
    the per-frame update() calls in tracking_stream.
    Q = abort (exit code 2, same convention as the person-selection window).

    `current_video_frame` (default None → int(start_s*fps)): the SAM 2
    source frame for the "seed" conditioning mask.  The other SOT trackers
    (cv2, OTTRACK) do NOT care about it - they work on raw pixels - so
    the kwarg exists only for `gsam2` (see core/detectors/
    sot_grounded_sam_2.py GroundedSam2Tracker.init()).  `tracking_stream`
    will use the same value via `int(start_s*fps)` when opening the
    identify window from the preview.
    """
    cap = cv2.VideoCapture(video_path)
    frame, _t, seeked = seek_to_time_exact(cap, start_s, fps=fps)
    if not seeked or frame is None:
        cap.release()
        # RuntimeError (not sys.exit(str) = os._exit): funscript_gen.main()
        # catches it, writes the message + traceback to the stderr log and
        # exits with code 1 - the GUI / log show the reason.
        raise RuntimeError("Failed to open the first frame.")
    # ── step 1: scene transform (zoom / rotate / pan / optional ROI crop) ──
    # ``no_scene=True`` (e.g. --no-scene-roi, or the R-reselect path where
    # the scene was already fixed) skips the window entirely; the fast path
    # returns the original frame (identity apply) with no extra work.
    _setup_scene(frame, tracker, skip=no_scene)
    # step 2: seed the ROI drag on the SCENE-SEEDED warped frame (identity
    # fast path when no scene transform is set -> SAME ndarray, zero copy)
    # The SOT tracker then sees the SAME frame in init() and in the
    # per-frame detect_all(update) calls of tracking_stream, so no
    # coordinate remapping is required downstream.
    frame = tracker.scene.apply(frame) if tracker.scene is not None else frame
    win = ("SOT: LMB=ROI p (green), Ctrl+LMB=ROI ps (blue)   "
           "R=clear   Enter=confirm   Q=abort")
    setup_window(win)
    drag = _drag_for(win)
    _drag_reset(win)

    roi_frame = None   # (x, y, w, h) in FRAME coordinates - point p (green)
    ps_frame = None    # (x, y, w, h) in FRAME coordinates - static ps (blue)
    init_err = None    # last tracker.init() failure -> shown as a red hint
    H, W = frame.shape[:2]

    def _to_frame(mx, my, scale, x0, y0):
        return max(0, min(W, int((mx - x0) / scale))), \
               max(0, min(H, int((my - y0) / scale)))

    def _draw_roi(view, scale, x0, y0, roi, color, label):
        if roi is None:
            return
        rx, ry, rw, rh = roi
        cv2.rectangle(view,
                      (int(rx * scale) + x0, int(ry * scale) + y0),
                      (int((rx + rw) * scale) + x0, int((ry + rh) * scale) + y0),
                      color, 3)
        cx, cy = box_center(roi)
        cv2.circle(view, (int(cx * scale) + x0, int(cy * scale) + y0), 6, color, -1)
        cv2.putText(view, label, (int(rx * scale) + x0,
                    max(15, int(ry * scale) + y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    while True:
        view, scale, x0, y0 = fit(frame)
        # ── live ROI p (green, required, plain LMB) ──
        if drag.p.down and drag.p.anchor is not None:
            ax, ay = _to_frame(drag.p.anchor[0], drag.p.anchor[1], scale, x0, y0)
            cx, cy = _to_frame(drag.p.px, drag.p.py, scale, x0, y0)
            roi_frame = (min(ax, cx), min(ay, cy),
                         max(1, abs(cx - ax)), max(1, abs(cy - ay)))
        # ── live ROI ps (blue, optional, Ctrl+LMB) ──
        if drag.ps.down and drag.ps.anchor is not None:
            ax, ay = _to_frame(drag.ps.anchor[0], drag.ps.anchor[1], scale, x0, y0)
            cx, cy = _to_frame(drag.ps.px, drag.ps.py, scale, x0, y0)
            ps_frame = (min(ax, cx), min(ay, cy),
                        max(1, abs(cx - ax)), max(1, abs(cy - ay)))
        _draw_roi(view, scale, x0, y0, roi_frame, (0, 0, 255), "p (required)")
        # BGR (255, 0, 0) = pure BLUE - the ps ROI color per the requirement.
        _draw_roi(view, scale, x0, y0, ps_frame,   (255, 0, 0),  "ps static (optional)")
        if init_err is not None:
            err = (f"tracker init failed: {init_err}  - LMB=new p ROI, "
                   f"Ctrl+LMB=new ps ROI, R=clear")
            (ew, _eh), _ = cv2.getTextSize(err, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            if ew > PREVIEW_W - 20:
                err = err[:max(1, (PREVIEW_W - 20) * len(err) // ew)]
                (ew, _eh), _ = cv2.getTextSize(err, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(view, err, ((PREVIEW_W - ew) // 2, PREVIEW_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 64, 255), 2)
        info = ("LMB=draw p ROI (release=fixed)   Ctrl+LMB=draw ps ROI "
                "(release=fixed)   R=clear all    Enter=confirm    Q=abort")
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.putText(view, info, ((PREVIEW_W - tw) // 2, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.imshow(win, view)

        k = cv2.waitKeyEx(30) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            print("[identify] SOT ROI aborted by user")
            sys.exit(2)
        if k in (ord('r'), ord('R')):
            roi_frame = None
            ps_frame = None
            init_err = None
            _drag_reset(win)
            continue
        if k in (13, 32):  # Enter / Space - confirm (p required, ps optional)
            if roi_frame is None:
                print("[identify] press + hold LMB to draw the p ROI first")
                continue
            x, y, w, h = roi_frame
            print(f"[identify] SOT ROI confirmed:       p=({x},{y},{w}x{h})")
            if ps_frame is not None:
                px0, py0, pw, ph = ps_frame
                print(f"[identify] SOT ROI confirmed:  ps=({px0},{py0},{pw}x{ph})")
            # OpenCV 5.x: Tracker.init() is void - success = no exception
            # (invalid params raise a cv2 exception), so do NOT check the
            # return value. On failure keep the window open and let the user
            # re-draw the ROI (red hint + logged traceback reason).
            try:
                # duck-typed: trackers that care about the video frame index
                # (gsam2, i.e. SAM 2) set `pending_seed_frame` (consumed by
                # their init()); other trackers don't have the attribute so
                # this is a no-op for them.
                if hasattr(tracker, "pending_seed_frame") and \
                        current_video_frame is not None:
                    tracker.pending_seed_frame = current_video_frame
                tracker.init(frame, (x, y, w, h))
            except Exception as e:
                tracker.reset()  # discard the half-initialized tracker
                # A bare exception like `OSError()` (errno=None, empty args)
                # prints as just "OSError" and hides the real cause
                # (errno / filename / module).  Format with class name + full
                # args and dump a traceback to the stderr log (so the OFS GUI
                # "Open stderr log" panel shows the true origin of the raise).
                import traceback as _tb
                _args = tuple(e.args) if getattr(e, "args", None) else (str(e),)
                _tb_text = _tb.format_exc(limit=20)
                try:
                    from core.logging import log as _sot_log
                    for _line in _tb_text.split("\n"):
                        _sot_log(f"[identify] SOT init traceback: {_line}")
                except Exception:
                    pass
                print(f"[identify] SOT tracker init failed: {type(e).__name__}(args={_args!r})")
                init_err = f"{type(e).__name__}" + (f"({', '.join(repr(a) for a in _args)})" if _args else "") + " - see stderr log for the traceback"
                continue
            cap.release()
            cv2.destroyAllWindows()
            person = Person((x, y, w, h),
                            box_center(roi_frame), score=1.0)
            # Stash the optional ps ROI (static reference element) on the
            # tracker object so funscript_gen (the caller) can clone the
            # tracker for it and seed the clone with this box.  None when
            # the user didn't draw a ps - downstream code must treat the
            # absence as "no camera drift compensation, baseline = identity".
            if ps_frame is not None:
                tracker.ps_roi = ps_frame
                print(f"[identify] ps_roi attached to tracker: {ps_frame}")
            return person, None
    # unreachable (loop exits only via sys.exit / return)
    cap.release()
    sys.exit(2)


def identify_partners(video_path: str, start_s: float, fps: float,
                      tracker: PersonTracker,
                      auto: bool = False,
                      no_scene_roi: bool = False
                      ) -> Tuple[Person, Optional[Person]]:
    """Returns (active, passive).  SOT trackers: (roi_person, None).
    GroundingDINO: (prompt-person with user-placed point, passive-or-None).

    ``no_scene_roi`` (SOT only): skip the scene setup window entirely and
    use identity (full frame, no transform) - useful for batch / automated
    runs where the video already has the right framing.
    """
    if tracker.kind == "sot":
        return _identify_sot(video_path, start_s, fps, tracker,
                             no_scene=no_scene_roi)
    if tracker.kind == "dino":
        return _identify_dino(video_path, start_s, fps, tracker)
    return _identify_persons(video_path, start_s, fps, tracker, auto)


def _identify_dino(video_path: str, start_s: float, fps: float,
                   tracker: PersonTracker,
                   hard_error: bool = True) -> Tuple[Person, Optional[Person]]:
    """GroundingDINO mode: the window shows the detected boxes AND the
    real detection ROI of the current target prompt (same green), plus
    the tracked point p.

    Point p = a PERCENT position inside the ACTIVE detection box:
      * default = (50, 50) = the box CENTER (the chosen default);
      * press + hold LMB inside the active box - the point follows the
        cursor (clamped to the box); RELEASE fixes it (the same
        press/hold/release interaction as the SOT ROI window);
      * R = re-run detection on the next frame (the percent selection is
        KEPT - it is relative to the box, so it follows the box);
      * Enter / Space = confirm (stores the selection on the tracker:
        every subsequent frame places p at the same percent inside the
        box detected by the prompt);
      * Q = abort (exit code 2).
    """
    cap = cv2.VideoCapture(video_path)
    frame, _t, seeked = seek_to_time_exact(cap, start_s, fps=fps)
    if not seeked or frame is None:
        cap.release()
        raise RuntimeError("Failed to open the first frame.")

    persons: List[Person] = []
    hint = None

    def _redetect():
        nonlocal persons, hint, frame
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1)
        ret, fr = cap.read()
        if not ret:
            frame, _t2, seeked2 = seek_to_time_exact(cap, start_s, fps=fps)
            if not seeked2:
                cap.release()
                raise RuntimeError("End of video reached while re-detecting.")
            fr = frame
        if fr is None:
            cap.release()
            raise RuntimeError("End of video reached while re-detecting.")
        persons = sorted(tracker.detect_all(fr), key=lambda p: -p.score)
        hint = None
        if not persons:
            if hard_error:
                cap.release()
                raise RuntimeError(
                    f"GroundingDINO: no detection for prompt "
                    f"'{getattr(tracker, 'prompt', '?')}' on this frame. "
                    "Try a different prompt/target or --start.")
            hint = ("no detection on this frame (prompt found nothing) - "
                    "press R to try the next frame, Q to go back")

    _redetect()
    if not persons and not hard_error:
        # re-aim mode (R from the preview): a lost frame must not kill the
        # whole run - the caller (tracking_stream) treats (None, None) as
        # "keep the previous point, no state change".
        cap.release()
        print("[identify] DINO ROI: no detection on this frame - "
              "keeping the previous point placement")
        return None, None
    if not persons:
        cap.release()
        raise RuntimeError(
            f"GroundingDINO: no detection for prompt "
            f"'{getattr(tracker, 'prompt', '?')}'. Try another --dino-target, "
            "--dino-prompt, a lower threshold, or a different --start.")

    _DINO_GREEN = (0, 255, 0)
    win = ("GroundingDINO: LMB inside the active box = move point p "
           "(release = fix)  R = next frame  Enter = confirm  Q = abort")
    setup_window(win)
    drag = _drag_for(win)
    _drag_reset(win)

    # current percent selection (relative to the ACTIVE box); default =
    # the tracker's selection (50, 50) = box center
    sel_u, sel_v = (tracker.point_percent
                    if hasattr(tracker, "point_percent") else (50.0, 50.0))

    def _rect(cnv, scale, x0, y0, x, y, w, h, color, t=2):
        cv2.rectangle(cnv,
                      (int(x * scale) + x0, int(y * scale) + y0),
                      (int((x + w) * scale) + x0, int((y + h) * scale) + y0),
                      color, t)

    while True:
        best = persons[0] if persons else None  # most confident = ACTIVE
        view, scale, x0, y0 = fit(frame)

        # every detected box in the SAME green - both the person contexts
        # AND the real target-prompt ROI - with per-box labels
        for i, per in enumerate(persons):
            x, y, w, h = per.box
            t = 3 if i == 0 else 2
            _rect(view, scale, x0, y0, x, y, w, h, _DINO_GREEN, t=t)
            label = (f"ACTIVE  {per.score:.2f}" if i == 0
                     else f"#{i + 1}  {per.score:.2f}")
            cv2.putText(view, label,
                        (int(x * scale) + x0, max(15, int(y * scale) + y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DINO_GREEN, 2)

        # live point p: while LMB is held the marker follows the cursor,
        # clamped INSIDE the active box (percents); on release the update
        # stops = the selection is FIXED at the last position
        if best is not None and drag.p.down:
            bx, by, bw, bh = best.box
            fx = (drag.p.px - x0) / scale
            fy = (drag.p.py - y0) / scale
            sel_u = min(100.0, max(0.0, (fx - bx) / max(1.0, bw) * 100.0))
            sel_v = min(100.0, max(0.0, (fy - by) / max(1.0, bh) * 100.0))

        if best is not None:
            bx, by, bw, bh = best.box
            px = bx + sel_u / 100.0 * bw
            py = by + sel_v / 100.0 * bh
            cx, cy = int(px * scale) + x0, int(py * scale) + y0
            cv2.circle(view, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(view, (cx - 13, cy), (cx + 13, cy), (0, 0, 0), 2)
            cv2.line(view, (cx, cy - 13), (cx, cy + 13), (0, 0, 0), 2)
            cv2.putText(view, ptxt, (cx + 16, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(view, f"u={sel_u:.0f}%  v={sel_v:.0f}%",
                        (cx + 16, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if hint is not None:
            (hw, _hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(view, hint, ((PREVIEW_W - hw) // 2, PREVIEW_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 64, 255), 2)

        info = ("LMB inside active box: press + hold = move p, release = fix    "
                "R = next frame    Enter = confirm    Q = abort")
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.putText(view, info, ((PREVIEW_W - tw) // 2, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.imshow(win, view)

        k = cv2.waitKeyEx(30) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            if hard_error:
                print("[identify] GroundingDINO ROI aborted by user")
                sys.exit(2)
            # re-aim mode (R from the preview): Q = go back, keep the
            # previous point placement (caller continues the stream)
            print("[identify] DINO ROI re-aim closed - "
                  "keeping the previous point placement")
            return None, None
        if k in (ord('r'), ord('R')):
            _drag_reset(win)
            _redetect()
            continue
        if k in (13, 32):  # Enter / Space
            if not persons:
                continue  # no detection yet - the hint says to press R
            best = persons[0]
            bx, by, bw, bh = best.box
            tracker.set_point_percent(sel_u, sel_v)  # store on the tracker
            px = bx + sel_u / 100.0 * bw
            py = by + sel_v / 100.0 * bh
            print(f"[identify] DINO ROI confirmed: box={best.box} "
                  f"score={best.score:.2f} "
                  f"point p=({px:.1f}, {py:.1f}) u={sel_u:.1f}% v={sel_v:.1f}% "
                  f"prompt='{getattr(tracker, 'prompt', '?')}'")
            cap.release()
            cv2.destroyAllWindows()
            active = Person(best.box, (float(px), float(py)), best.score,
                            kp=best.kp, kp_conf=best.kp_conf)
            passive = persons[1] if len(persons) > 1 else None
            return active, passive
    # unreachable (loop exits only via sys.exit / return)
    cap.release()
    sys.exit(2)


def _identify_persons(video_path: str, start_s: float, fps: float,
                      tracker: PersonTracker,
                      auto: bool) -> Tuple[Person, Optional[Person]]:
    """
    First-frame identify window: choose who is ACTIVE and who is PASSIVE.

    Keys:
        ↑/↓ or 1..N  - select candidate
        A / a        - mark selected as ACTIVE
        P / p        - mark selected as PASSIVE
        R / r        - re-detect (next frame)
        Enter        - confirm (requires ACTIVE)
        Q / Esc      - abort
    """
    cap = cv2.VideoCapture(video_path)
    frame, _t, seeked = seek_to_time_exact(cap, start_s, fps=fps)
    if not seeked or frame is None:
        cap.release()
        raise RuntimeError("Failed to open the first frame.")
    persons = tracker.detect_all(frame)
    if not persons:
        cap.release()
        # both modes: a real error (exit code 1) -- no result to import.
        # RuntimeError (not sys.exit(str) = os._exit) so funscript_gen.main()
        # can log the reason to the stderr log before exiting.
        raise RuntimeError("No person found. Try --tracker yolo "
                           "or a different --start.")
    persons = sorted(persons, key=lambda p: -p.score)

    if auto:
        passive = persons[1] if len(persons) > 1 else None
        cap.release()
        print(f"[identify] auto: active=person#0 score={persons[0].score:.2f}")
        if passive:
            print(f"[identify] passive=person#1 score={passive.score:.2f}")
        else:
            print("[identify] NO PASSIVE DETECTED (auto)")
        return persons[0], passive

    win = ("Hismith Funscript - identify partners "
           "(UP/DOWN=select, A=active, P=passive, Enter=confirm, Q=abort)")
    setup_window(win)
    active = passive = None
    sel_idx = 0

    def _to_canvas(scale, x0, y0, x, y):
        return (int(x * scale) + x0, int(y * scale) + y0)

    def _rect(cnv, scale, x0, y0, x, y, w, h, color, t=2):
        cv2.rectangle(cnv,
                      (int(x * scale) + x0, int(y * scale) + y0),
                      (int((x + w) * scale) + x0, int((y + h) * scale) + y0),
                      color, t)

    def _dot(cnv, scale, x0, y0, x, y, r, color, t=-1):
        cv2.circle(cnv, _to_canvas(scale, x0, y0, x, y), int(r * scale), color, t)

    def _cross(cnv, scale, x0, y0, x, y, r, color, t=2):
        c = _to_canvas(scale, x0, y0, x, y)
        rr = int(r * scale)
        cv2.line(cnv, (c[0] - rr, c[1]), (c[0] + rr, c[1]), color, t)
        cv2.line(cnv, (c[0], c[1] - rr), (c[0], c[1] + rr), color, t)

    _SEL_RED = (0, 0, 255)

    while True:
        view, scale, x0, y0 = fit(frame)
        # pass 1: draw every person box + label (selected box drawn LAST)
        for i, per in enumerate(persons):
            x, y, w, h = per.box
            if per is active:
                color, tag = (0, 200, 0), "ACTIVE"
            elif per is passive:
                color, tag = (0, 200, 255), "PASSIVE"
            elif i == sel_idx:
                color, tag = _SEL_RED, "selected"
            else:
                color, tag = (255, 128, 0), f"#{i + 1}"
            _rect(view, scale, x0, y0, x, y, w, h, color, t=3 if i == sel_idx else 2)
            if i == sel_idx:
                continue  # selected label is drawn in pass 2, on top
            _dot(view, scale, x0, y0, per.point[0], per.point[1], 7, color)
            _cross(view, scale, x0, y0, per.point[0], per.point[1], 12, color)
            label = f"person {i + 1}  {tag}  {per.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(th + 2, int(y * scale) + y0 - 6)
            cv2.putText(view, label,
                        (int(x * scale) + x0, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # pass 2: selected person - solid red box + bigger red label, drawn last
        per = persons[sel_idx]
        x, y, w, h = per.box
        _rect(view, scale, x0, y0, x, y, w, h, _SEL_RED, t=4)
        label = f"person {sel_idx + 1}  selected  {per.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        text_y = max(th + 4, int(y * scale) + y0 - 8)
        cv2.putText(view, label,
                    (int(x * scale) + x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, _SEL_RED, 2)
        _dot(view, scale, x0, y0, per.point[0], per.point[1], 7, _SEL_RED)
        _cross(view, scale, x0, y0, per.point[0], per.point[1], 12, _SEL_RED)
        info = (f"[UP/DOWN or 1..{len(persons)}] select   "
                f"[A]active  [P]passive  [R]re-detect  "
                f"[Enter]confirm  [Q]abort   "
                f"[hold S]=skeleton of selected  "
                f"  active={'yes' if active else '-'} "
                f"passive={'yes' if passive else '-'}")
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(view, info, ((PREVIEW_W - tw) // 2, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if key_held(ord('s')) or key_held(ord('S')):
            draw_skeleton(view, scale, x0, y0, persons[sel_idx])
        cv2.imshow(win, view)
        k = cv2.waitKeyEx(30)
        if k in (ord('q'), ord('Q')):
            cap.release()
            cv2.destroyAllWindows()
            # exit code 2 => "aborted by user" - main.lua treats this
            # separately from a real error.
            print("[identify] aborted by user")
            sys.exit(2)
        if k in (ord('r'), ord('R')):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1)
            ret, frame = cap.read()
            if not ret:
                frame, _t, seeked = seek_to_time_exact(cap, start_s, fps=fps)
                if not seeked:
                    raise RuntimeError("End of video reached while re-detecting.")
            persons = sorted(tracker.detect_all(frame),
                             key=lambda p: -p.score)
            sel_idx = 0
            active = passive = None
        elif k == KEY_UP:
            sel_idx = (sel_idx - 1) % max(len(persons), 1)
        elif k == KEY_DOWN:
            sel_idx = (sel_idx + 1) % max(len(persons), 1)
        elif ord('1') <= k <= ord('9'):                                  # 1..9
            n = k - 49
            if 0 <= n < len(persons):
                sel_idx = n
        elif k in (ord('a'), ord('A')):
            active = persons[sel_idx]
        elif k in (ord('p'), ord('P')):
            passive = persons[sel_idx]
        elif (k == ord('\r')) and active is not None:                # Enter: require active
            cap.release()
            cv2.destroyAllWindows()
            print(f"[identify] ACTIVE:   box={active.box} score={active.score:.2f}")
            if passive:
                print(f"[identify] PASSIVE:  box={passive.box} score={passive.score:.2f}")
            return active, passive
    # unreachable (loop only exits via q/esc/enter)
    cap.release()
    sys.exit(2)

