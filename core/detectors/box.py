"""
Box-only trackers have no landmarks - point p is derived from the box:

  * SOT ROI trackers (sot-csrt/kcf/vit/mil, ostrack) - the user draws the ROI
    exactly around the zone to track, so p = the box CENTER (box_center);
  * GroundingDINO (detection by text prompt) - p = a user-selected PERCENT
    position inside the box (GroundingDinoTracker.point_percent, default
    (50, 50) = box center; set via set_point_percent in the _identify_dino
    window).  point_from_box(box, target) below is the legacy pose-target
    fraction helper (core.config.TARGET_BOX_FRAC) - pose trackers only;
    DINO uses its own percent-based method (same subpixel math).
"""
from typing import Tuple

from core.config import TARGET_BOX_FRAC

Box = Tuple[int, int, int, int]


def box_center(box: Box) -> Tuple[float, float]:
    """Center of the box (SUBPIXEL floats). Point p for the SOT ROI trackers.
    The user drew the ROI around the exact zone to track, so its center IS
    the tracked point - no per-target vertical offset needed."""
    x, y, w, h = box
    return (float(x + w / 2.0), float(y + h / 2.0))


def point_from_box(box: Box, target: str) -> Tuple[float, float]:
    """Point p for a box-based tracker (GroundingDINO: fraction per target)."""
    x, y, w, h = box
    frac = TARGET_BOX_FRAC.get(target, 0.65)
    return (float(x + w / 2.0), float(y + frac * h))
