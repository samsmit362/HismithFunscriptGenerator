"""
Window-size helpers shared by the flow refiner and the hold-over ghost:
an LK window clamped to odd sizes in 15..71 (91 for the hold-over), and
a person-scale estimate from the detected keypoints.
"""
from typing import List, Optional

import numpy as np

from core.types import Person


def odd_clamp(v: float, lo: int = 15, hi: int = 71) -> int:
    """Clamp to [lo, hi] and round to the nearest ODD int (LK winSize)."""
    v = min(max(float(v), lo), hi)
    v = int(round(v))
    if v % 2 == 0:
        v += 1
    return v


def person_scale_estimate(person: Person) -> Optional[float]:
    """
    Person-scale proxy (px), robust to partial visibility:
    candidates = hip span / 2, thigh bone length / 10 (both confidence-
    filtered: only keypoints with conf/visibility >= 0.5 count), plus the
    box height / 8 as an always-available fallback.
    Returns the median of the available candidates, or None.
    Key index layout by array length: COCO-17 (17) vs mediapipe-33 (33):
      COCO: 11/12 hips, 13/14 knees; mediapipe: 23/24 hips, 25/26 knees.
    Box-based trackers (dino / sot) have kp=None -> only the box fallback.
    """
    cands: List[float] = []
    kp = person.kp
    if kp is not None and len(kp) in (17, 33):
        LHIP, RHIP, LKN, RKN = (11, 12, 13, 14) if len(kp) == 17 else (23, 24, 25, 26)
        conf = (lambda i: float(person.kp_conf[i])) if person.kp_conf is not None \
            else (lambda i: 1.0)
        ok = lambda i: conf(i) >= 0.5
        if ok(LHIP) and ok(RHIP):
            span = max(abs(kp[LHIP][0] - kp[RHIP][0]),
                       abs(kp[LHIP][1] - kp[RHIP][1])) / 2.0
            if span >= 4.0:
                cands.append(span)
        for hi_i, kn_i in ((LHIP, LKN), (RHIP, RKN)):
            if ok(hi_i) and ok(kn_i):
                bone = float(np.hypot(kp[hi_i][0] - kp[kn_i][0],
                                      kp[hi_i][1] - kp[kn_i][1]))
                if bone >= 8.0:
                    cands.append(bone / 10.0)
    if person.box[3] >= 20:
        cands.append(float(person.box[3]) / 8.0)
    if not cands:
        return None
    return float(np.median(cands))
