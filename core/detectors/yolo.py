"""
YOLO-pose tracker (v8 / v11 / v26 families, shared COCO-17 keypoints).

person_target -> COCO-17 keypoint indices.  'torso' is a derived point
(midpoint of hip-midpoint and shoulder-midpoint) - computed here, the only
place that knows the YOLO keypoint layout for that zone.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import TARGETS
from core.detectors.models import yolo_model_path
from core.detectors.person_tracker import PersonTracker
from core.types import Person

TARGET_MAP: Dict[str, Tuple] = {
    # (pa, pb) - the two keypoints whose midpoint is point p
    "mid-hips": (11, 12),
    "left-hip": (11, 11),
    "right-hip": (12, 12),
    "thigh": (13, 14),
    "torso": None,          # special-case, see _resolve
    "shoulders": (5, 6),
    "head": (0, 0),
    "ankle": (15, 16),
}


class YoloTracker(PersonTracker):
    """YOLO-pose -> the person's point p per the person_target choice."""

    kind = "pose"

    def __init__(self, model: str = "yolo26x-pose", target: str = "mid-hips",
                 imgsz: int = 1280):
        if target not in TARGETS:
            raise ValueError(f"unknown person_target: {target!r} (choose from {TARGETS})")
        from ultralytics import YOLO
        self.model_path = yolo_model_path(model)
        self.model = YOLO(self.model_path)
        self.imgsz = imgsz
        self.target = target
        self._map = TARGET_MAP[target]
        self._torso = (target == "torso")

    def _resolve(self, kps: np.ndarray) -> Tuple[float, float]:
        if self._torso:
            # torso center: midpoint of the hip midpoint and shoulder midpoint
            hmid = ((kps[11, 0] + kps[12, 0]) / 2.0, (kps[11, 1] + kps[12, 1]) / 2.0)
            smid = ((kps[5, 0] + kps[6, 0]) / 2.0, (kps[5, 1] + kps[6, 1]) / 2.0)
            return ((hmid[0] + smid[0]) / 2.0, (hmid[1] + smid[1]) / 2.0)
        pa, pb = self._map
        return (float((kps[pa, 0] + kps[pb, 0]) / 2.0),
                float((kps[pa, 1] + kps[pb, 1]) / 2.0))

    def detect_all(self, frame: np.ndarray) -> List[Person]:
        # conf=0.3: yolo26 (end-to-end) calibrates box confidence lower than
        # v11 (the same person scored ~0.40 on yolo26x where v11x scored ~0.5+).
        pred = self.model(frame, conf=0.3, imgsz=self.imgsz, verbose=False)[0]
        out: List[Person] = []
        if pred.boxes is None or pred.keypoints is None:
            return out
        for i in range(pred.boxes.conf.shape[0]):
            kps = pred.keypoints[i].xy[0].cpu().numpy()
            confs = pred.keypoints[i].conf[0].cpu().numpy()
            if self._torso:
                if confs[11] < 0.3 and confs[12] < 0.3 and \
                   confs[5] < 0.3 and confs[6] < 0.3:
                    continue
            else:
                pa, pb = self._map
                if confs[pa] < 0.3 and confs[pb] < 0.3:
                    continue
            point_p = self._resolve(kps)
            x1, y1, x2, y2 = pred.boxes.xyxy[i].cpu().numpy().astype(int)
            score = float(pred.boxes.conf[i])
            out.append(Person((int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                              point_p, score, kp=np.asarray(kps, dtype=np.float32),
                              kp_conf=np.asarray(confs, dtype=np.float32)))
        return out
