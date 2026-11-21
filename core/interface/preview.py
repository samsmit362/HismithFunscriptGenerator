"""
OpenCV preview helpers: letterbox fitting, window centering, and the
skeleton overlay (hold S in any window).
"""
import cv2
import numpy as np

from core.types import Person
from core.logging import log

PREVIEW_W, PREVIEW_H = 960, 540  # 480x270 * 2, 16:9


def put_text_centered(img, text, y, font_scale=0.7, color=(255, 255, 255),
                      thickness=2, min_scale=0.3, max_width=None):
    """Draw text centered horizontally on img.
    If the text is wider than max_width (default PREVIEW_W - 20), the font
    scale is reduced step-by-step until it fits or min_scale is reached.
    The x position is clamped to [5, img_width - tw - 5] so it never goes
    off the left or right edge."""
    if max_width is None:
        max_width = img.shape[1] - 10
    fs = font_scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
    tw_start = tw
    while tw > max_width and fs > min_scale:
        fs -= 0.05
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
    # log(f"[put_text_centered] PREVIEW_W: {PREVIEW_W} img_w: {img.shape[1]} "
    #     f"tw_start: {tw_start} tw_end: {tw} fs_start: {font_scale} fs_end: {fs}")
    x = max(5, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, thickness)


def fit(frame: np.ndarray):
    """Letterboxes the frame into PREVIEW_W x PREVIEW_H keeping aspect ratio.
    Returns (canvas, scale, x0, y0)."""
    h, w = frame.shape[:2]
    scale = min(PREVIEW_W / w, PREVIEW_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PREVIEW_H, PREVIEW_W, 3), np.uint8)
    x0 = (PREVIEW_W - nw) // 2
    y0 = (PREVIEW_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, scale, x0, y0


def center_window(win_name: str, w: int, h: int):
    """Centers an OpenCV window on the screen."""
    try:
        sw, sh = cv2.getWindowImageSize(win_name)
        if sw > 0 and sh > 0:
            import ctypes
            user32 = ctypes.windll.user32
            swin = user32.GetSystemMetrics(0)
            shgt = user32.GetSystemMetrics(1)
            x = max(0, (swin - sw) // 2)
            y = max(0, (shgt - sh) // 2)
            cv2.moveWindow(win_name, x, y)
    except Exception:
        pass


def setup_window(win: str, w: int = PREVIEW_W, h: int = PREVIEW_H) -> None:
    """Create a resizable window with aspect ratio, fixed size, centered."""
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if hasattr(cv2, "WND_KEEP_ASPECT_RATIO"):
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WND_KEEP_ASPECT_RATIO)
    cv2.resizeWindow(win, w, h)
    center_window(win, w, h)


def tracker_label(tracker) -> str:
    """Short display name of the tracking method for window titles / hints
    (the registry name, e.g. 'sot-csrt'; falls back to the tracker kind,
    then '?', for manually constructed tracker objects)."""
    n = getattr(tracker, "name", None)
    if n:
        return str(n)
    k = getattr(tracker, "kind", None)
    return str(k) if k else "?"


# bone index pairs per keypoint format (17 = COCO/YOLO, >=33 = mediapipe)
COCO_BONES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 8), (6, 8), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16))
MP_BONES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28))


def draw_skeleton(img, scale, x0, y0, person: Person,
                  color: tuple = (0, 220, 255),
                  label: str = "tracked p") -> None:
    """Debug overlay: draw person.kp as a stick figure and highlight the
    tracked point (ring + label).  No-op for box-based trackers (kp None),
    which draw the point ring + ROI instead.
    `color` (BGR) and `label` are per-person (active=yellow, passive=orange).
    """
    kps = None
    if person.kp is not None:
        kps = np.asarray(person.kp, dtype=np.float32)
        if kps.ndim != 2 or len(kps) < 2:
            kps = None
    if kps is not None:
        bones = COCO_BONES if len(kps) == 17 else (MP_BONES if len(kps) >= 33 else [])
        for a, b in bones:
            if b >= len(kps):
                continue
            p1 = (int(kps[a, 0] * scale) + x0, int(kps[a, 1] * scale) + y0)
            p2 = (int(kps[b, 0] * scale) + x0, int(kps[b, 1] * scale) + y0)
            cv2.line(img, p1, p2, color, 2)
            cv2.circle(img, p1, 3, color, -1)
    px, py = person.point
    cx, cy = int(px * scale) + x0, int(py * scale) + y0
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), 2)
    if kps is None:
        # box-based tracker: show the detection box too
        x, y, w, h = person.box
        cv2.rectangle(img,
                      (int(x * scale) + x0, int(y * scale) + y0),
                      (int((x + w) * scale) + x0, int((y + h) * scale) + y0),
                      color, 1)
    cv2.putText(img, label, (cx + 14, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
