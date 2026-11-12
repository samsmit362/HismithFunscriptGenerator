"""
OpenCV preview helpers: letterbox fitting, window centering, and the
skeleton overlay (hold S in any window).
"""
import cv2
import numpy as np

from core.types import Person

PREVIEW_W, PREVIEW_H = 960, 540  # 480x270 * 2, 16:9


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
