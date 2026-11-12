"""
MediaPipe Pose tracker (legacy solutions API or Tasks API, automatic
fallback) - 33 landmarks (BlazePose).

person_target -> BlazePose landmark indices.  Left/right are the person's
own side: 23 = left hip, 24 = right hip (mirrored in image space, same
convention as the COCO mapping in yolo.py).
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.config import TARGETS
from core.detectors.models import mediapipe_model_path
from core.detectors.person_tracker import PersonTracker
from core.types import Person

TARGET_MAP: Dict[str, Tuple] = {
    "mid-hips": (23, 24),
    "left-hip": (23, 23),
    "right-hip": (24, 24),
    "thigh": (25, 26),
    "torso": None,          # special-case, see _resolve
    "shoulders": (11, 12),
    "head": (0, 0),
    "ankle": (27, 28),
}


class MediaPipeTracker(PersonTracker):
    """MediaPipe Pose -> the person's point p per the person_target choice."""

    kind = "pose"

    def __init__(self, fps: float = 30.0, target: str = "mid-hips"):
        if target not in TARGETS:
            raise ValueError(f"unknown person_target: {target!r} (choose from {TARGETS})")
        import mediapipe as mp
        self.target = target
        self._map = TARGET_MAP[target]
        self._torso = (target == "torso")
        self.mp = mp
        self.ts_ms = 0
        self.fps = fps

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.mode = "legacy"
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                num_poses=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.mode = "tasks"
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            model_path = mediapipe_model_path()
            options = vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=3,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _resolve(self, get_kp, w: int, h: int) -> Tuple[float, float]:
        """`get_kp(i)` -> normalized landmark (x, y); returns px (x, y)."""
        if self._torso:
            l_h, r_h = get_kp(23), get_kp(24)
            l_s, r_s = get_kp(11), get_kp(12)
            hmid = ((l_h[0] + r_h[0]) / 2.0, (l_h[1] + r_h[1]) / 2.0)
            smid = ((l_s[0] + r_s[0]) / 2.0, (l_s[1] + r_s[1]) / 2.0)
            return (((hmid[0] + smid[0]) / 2.0) * w,
                    ((hmid[1] + smid[1]) / 2.0) * h)
        pa, pb = self._map
        ka, kb = get_kp(pa), get_kp(pb)
        return ((ka[0] + kb[0]) / 2.0 * w,
                (ka[1] + kb[1]) / 2.0 * h)

    def _gate(self, visibility, idxs) -> bool:
        if self._torso:
            idxs = idxs + (11, 12)
        return any(visibility[i] >= 0.3 for i in dict.fromkeys(idxs))

    def detect_all(self, frame: np.ndarray) -> List[Person]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out: List[Person] = []
        if self.mode == "legacy":
            res = self.pose.process(rgb)
            for p in (res.pose_landmarks or []):
                lm = p.landmark
                vis = [float(getattr(m, "visibility", 1.0)) for m in lm]
                if not self._gate(vis, self._map + self._map if self._map else ()):
                    continue
                px, py = self._resolve(lambda i: (lm[i].x, lm[i].y), w, h)
                xs = [m.x * w for m in lm]
                ys = [m.y * h for m in lm]
                box = (int(min(xs)), int(min(ys)),
                       int(max(xs) - min(xs)) + 1, int(max(ys) - min(ys)) + 1)
                kps = np.asarray([(m.x * w, m.y * h) for m in lm], dtype=np.float32)
                kpconf = np.asarray(vis, dtype=np.float32)
                out.append(Person(box, (px, py), float(getattr(p, "presence", 0.0) or 1.0),
                                  kp=kps, kp_conf=kpconf))
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(mp_image, self.ts_ms)
            self.ts_ms += int(round(1000.0 / self.fps))
            for p in res.pose_landmarks:
                vis = [float(getattr(lm, "visibility", 1.0)) for lm in p]
                if self._map is None:
                    if not any(vis[i] >= 0.3 for i in (23, 24, 11, 12)):
                        continue
                else:
                    if not self._gate(vis, self._map):
                        continue
                px, py = self._resolve(lambda i: (p[i].x, p[i].y), w, h)
                xs = [lm.x * w for lm in p]
                ys = [lm.y * h for lm in p]
                box = (int(min(xs)), int(min(ys)),
                       int(max(xs) - min(xs)) + 1, int(max(ys) - min(ys)) + 1)
                kps = np.asarray(list(zip(xs, ys)), dtype=np.float32)
                kpconf = np.asarray(vis, dtype=np.float32)
                out.append(Person(box, (px, py), float(vis[0] if vis else 1.0),
                                  kp=kps, kp_conf=kpconf))
        return out

    def close(self):
        if self.mode == "legacy":
            self.pose.close()
        else:
            self.landmarker.close()
