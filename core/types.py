from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Person:
    box: Tuple[int, int, int, int]  # x, y, w, h
    point: Tuple[float, float]      # (x, y) of the tracked point p (see config.TARGETS)
    score: float = 1.0
    kp: Optional[np.ndarray] = None  # (K, 2) all keypoints in px; skeleton overlay (hold S);
                                     # None for box-based trackers (GroundingDINO / SOT)
    kp_conf: Optional[np.ndarray] = None  # (K,) per-keypoint confidence / visibility (if the tracker provides it)


class ThrustCalibration:
    """
    Holds the manual f/b marks for the ACTIVE partner and the initial thrust axis.
    Axis is re-estimated per stroke (see postpass.strokes._stroke_axis) - the
    camera may drift, so the b->f angle can change gradually across the video.
    """

    def __init__(self):
        self.f: Optional[Tuple[float, float]] = None   # full-insertion point (pos=0)
        self.b: Optional[Tuple[float, float]] = None   # full-retracted point (pos=100)
        self.m: Optional[Tuple[float, float]] = None   # optional 50% mark (arc/parabola fit)
        self.frames: Dict[str, int] = {}

    @property
    def initial_axis(self) -> Optional[np.ndarray]:
        """Unit vector from b to f (insertion direction, the main motion).
        None if not calibrated."""
        if self.f is None or self.b is None:
            return None
        v = np.asarray(self.f, dtype=np.float64) - np.asarray(self.b, dtype=np.float64)
        nrm = np.linalg.norm(v)
        if nrm < 1e-6:
            return None
        return v / nrm
