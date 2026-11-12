"""
Per-frame partner matching: which of the current detections is the ACTIVE
partner.  Distance-dominant (the candidate whose point p moved LEAST from
the previously tracked point wins) with a small box-IoU continuity term.

Tracker-agnostic: works for pose, text-prompt and SOT trackers alike —
it only consumes Person.box / Person.point / Person.score.
"""
from typing import List, Optional, Tuple

import numpy as np

from core.types import Person


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU for boxes as (x, y, w, h)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-6)


def match_person(current: List[Person],
                 tracked_person: Optional[Person],
                 iou_thr: float = 0.25) -> Optional[int]:
    """
    Match the tracked person against a fresh detection by box IoU
    (pose trackers do not guarantee a stable person ordering between frames).
    Returns index into `current` or None.
    """
    if tracked_person is None or not current:
        return 0 if current else None
    best_idx, best_iou = None, iou_thr
    for i, p in enumerate(current):
        v = box_iou(p.box, tracked_person.box)
        if v > best_iou:
            best_iou, best_idx = i, v
    return best_idx


def match_active(current: List[Person],
                 last_point: Optional[Tuple[float, float]],
                 prev_box: Optional[Tuple[int, int, int, int]] = None
                 ) -> Tuple[int, Optional[float]]:
    """
    Pick which of the current detections is the ACTIVE partner.

    Trackers (YOLO / MediaPipe / GroundingDINO) do not guarantee a stable
    person ordering between frames, so the old "iou 0.15 -> max(score)"
    fallback was silently re-binding the track to the OTHER partner whenever
    the active person's box confidence dropped.  In that case the point-jump
    is large (hundreds of px) and the user sees "the tracked point teleports
    to the passive partner".

    Here every candidate is scored and the best wins:
        cost = W_DIST * rel_distance + W_IOU * (1 - iou_with_prev_box)
        rel_distance = ||cand.point - last_point|| / candidate_box_scale

    `rel_distance` dominates: it is exactly the "minimum change from the
    previous point" criterion - the candidate that moved least wins,
    regardless of which partner the tracker assigned a higher score to.

    Returns (index, cost) or (-1, None) if nothing was scored.
    """
    if not current:
        return (-1, None)
    if last_point is None:
        # No reference point yet: fall back to best box IoU with the stored
        # box (calibration / identify first frame).
        if prev_box is not None:
            best_i, best = None, -1.0
            for i, p in enumerate(current):
                v = box_iou(p.box, prev_box)
                if v > best:
                    best, best_i = v, i
            if best_i is not None:
                return (best_i, float(best))
        return (int(np.argmax([p.score for p in current])), None)

    W_DIST = 10.0   # distance dominates
    W_IOU = 1.0     # secondary: box continuity
    best_idx = -1
    best_cost = float('inf')
    for i, p in enumerate(current):
        box_w, box_h = max(1, p.box[2]), max(1, p.box[3])
        s = 0.5 * (box_w + box_h)
        d = float(np.hypot(p.point[0] - last_point[0],
                           p.point[1] - last_point[1]))
        rel_d = d / s
        iou_part = 1.0
        if prev_box is not None:
            iou_part = 1.0 - box_iou(p.box, prev_box)
        cost = W_DIST * rel_d + W_IOU * iou_part
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return (best_idx, best_cost)
