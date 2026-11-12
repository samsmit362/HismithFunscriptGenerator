"""Interactive selection of the SINGLE fixed crop used by the LLM generator.

The LLM mode needs a ONE rectangle (x, y, w, h) in ORIGINAL frame pixels
that is cropped out of EVERY sampled frame and shrunk to a thumbnail for
the multimodal prompt.  This window:

  * opens on the frame at ``start_s`` (the reliable seek from
    ``core.tracking.seek``);
  * lets the user press + hold LMB to draw the rectangle live;
  * RELEASE fixes the rectangle at its last position (same stateful-drag
    pattern as ``core/interface/identify.py::_setup_scene`` and
    ``_identify_sot``);
  * ``R`` clears and re-draws;
  * ``Enter`` / ``Space`` CONFIRMS and returns ``(x, y, w, h)`` in
    ORIGINAL frame pixels (floats rounded to ints);
  * ``Q`` / ``Esc`` ABORTS with ``None`` (the caller must sys.exit(2)).

No partner selection, no per-frame tracking, no calibration - the user
just decides "what part of the frame should the model look at".

Keyboard / mouse cheatsheet (drawn in the hint line):
    LMB drag       draw / re-draw the crop
    R              clear and start over
    Enter / Space  confirm (returns the fixed crop)
    Q / Esc        abort (returns None)
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from core.interface.identify import _drag_for, _drag_reset
from core.interface.preview import PREVIEW_H, PREVIEW_W, fit, setup_window
from core.tracking.seek import seek_to_time_exact


def select_llm_roi(video_path: str,
                   start_s: float,
                   fps: float,
                   win: str = "LLM ROI (LMB drag, Enter=OK, Q=abort)"
                   ) -> Optional[Tuple[int, int, int, int]]:
    """Open the ROI window.  Returns the fixed crop in FRAME pixels
    (x, y, w, h) or None when the user aborts (Q / Esc).

    The caller is expected to:
      * pass the exact ``start_s`` the LLM span will start from
        (usually the same value the rest of the pipeline uses);
      * abort the whole run (``sys.exit(2)``) when None is returned.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"[llm-roi] cannot open video: {video_path!r}")
    # RELIABLE seek (POS_MSEC-based, see core/tracking/seek.py).  The
    # user-facing window frame is the exact starting frame of the
    # sampled span, so the crop is drawn where the LLM will see it.
    try:
        frame, _t0, ok = seek_to_time_exact(cap, start_s, fps=fps)
    except Exception:
        frame, ok = None, False
    if (frame is None) or (not ok):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
        ret, frame = cap.read()
        if (not ret) or (frame is None):
            cap.release()
            raise RuntimeError(
                "[llm-roi] could not read the frame at "
                f"start_s={start_s!r}")
    cap.release()
    fh, fw = frame.shape[:2]

    # Letterboxed preview + hint.
    view, scale, x0, y0 = fit(frame)
    setup_window(win)
    drag = _drag_for(win)
    _drag_reset(win)
    roi_draw = None  # (x, y, w, h) in FRAME pixels, None until the user
                    # finishes a real (non-zero-size) drag
    last_status = ""

    while True:
        # Live box (VIEW -> FRAME)
        if drag.p.down and drag.p.anchor is not None:
            ax, ay = drag.p.anchor
            cx, cy = drag.p.px, drag.p.py
            ax = (ax - x0) / scale
            ay = (ay - y0) / scale
            cx = (cx - x0) / scale
            cy = (cy - y0) / scale
            x1 = max(0.0, min(ax, cx))
            y1 = max(0.0, min(ay, cy))
            x2 = min(float(fw), max(ax, cx))
            y2 = min(float(fh), max(ay, cy))
            w, h = x2 - x1, y2 - y1
            if w >= 4 and h >= 4:          # ignore sub-pixel accidental drags
                roi_draw = (int(round(x1)), int(round(y1)),
                            int(round(w)), int(round(h)))
        elif (not drag.p.down) and roi_draw is not None:
            # the selection is now FIXED at its last dragged position
            # (the callback cleared .down after the LBUTTONUP)
            pass
        else:
            roi_draw = None

        display = view.copy()
        if roi_draw is not None:
            x, y, w, h = roi_draw
            p1 = (int(x * scale) + x0, int(y * scale) + y0)
            p2 = (int((x + w) * scale) + x0, int((y + h) * scale) + y0)
            cv2.rectangle(display, p1, p2, (0, 220, 0), 2)
            # darken everything OUTSIDE the ROI to show what will be
            # cropped out
            mask = np.zeros((PREVIEW_H, PREVIEW_W), np.uint8)
            cv2.rectangle(mask, p1, p2, 255, -1)
            overlay = display.copy()
            overlay[:] = (overlay * 0.25).astype(np.uint8)
            cv2.copyTo(overlay, mask, display)
            cv2.rectangle(display, p1, p2, (0, 220, 0), 2)
            # label: crop dims
            lab = f"crop {w}x{h}"
            (tw, th_), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            ox, oy = p2[0] + 6, p1[1] + max(12, th_ + 4)
            cv2.putText(display, lab, (ox, oy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 2)

        # Hint
        live = drag.p.down and drag.p.anchor is not None
        if roi_draw is not None:
            hint = ("LMB drag = re-size  R = reset  Enter = OK  Q = abort")
        else:
            hint = ("LMB drag = draw the crop  R = reset  Enter = OK  Q = abort")
        if live:
            hint = "dragging: " + hint
        hint_color = (0, 200, 255) if not live else (0, 255, 255)
        (tw, th_), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.putText(display, hint, (12, PREVIEW_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hint_color, 2)

        cv2.imshow(win, display)
        wait = 30 if not live else 1
        k = cv2.waitKey(wait) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            # abort - cleanup
            cv2.destroyWindow(win)
            return None
        if k in (13, 32):  # Enter / Space = confirm
            if roi_draw is not None and roi_draw[2] >= 4 and roi_draw[3] >= 4:
                cv2.destroyWindow(win)
                return (int(roi_draw[0]), int(roi_draw[1]),
                        int(roi_draw[2]), int(roi_draw[3]))
            # no ROI yet - flash a hint instead of confirming empty
            cv2.putText(display, "draw a crop first (LMB drag) or press Q to abort",
                        (12, PREVIEW_H - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow(win, display)
        elif k in (ord('r'), ord('R')):
            _drag_reset(win)
            roi_draw = None
