"""
Base contract for every point source ("tracker"): pose models (YOLO,
MediaPipe), text-prompt box detectors (GroundingDINO) and single-object
trackers (SOT: OpenCV ViT/MIL/CSRT/KCF on a manually drawn ROI).

Three ways a tracker produces the point p from its own output:
  * pose trackers  - person_target -> keypoint indices (TARGET_MAP), point =
                     midpoint of the mapped pair;
  * SOT ROI trackers - the ROI was drawn around the zone to track, so
                     point = the box CENTER (core.detectors.box.box_center);
  * GroundingDINO  - person_target -> vertical fraction inside the box:
                     point = (box_cx, box_y + TARGET_BOX_FRAC[target] * box_h)
                     (see core.detectors.box.point_from_box).
Downstream code (matching, flow, hold-over, preview, post-pass) only ever
consumes Person.box / Person.point / Person.score / Person.kp - it never
knows where the point came from.

Lifecycle:
  detect_all(frame) -> List[Person]   (every tracker, every frame)
  detect(frame)     -> Optional point (convenience: best-score person)
  init(frame, roi) / reset()          (SOT trackers; default no-op)
  kind: "pose" | "dino" | "sot"       (UI hints, logging, identify window)
"""
from typing import List, Optional, Tuple

import numpy as np

from core.types import Person


class PersonTracker:
    """Base contract. Subclasses implement detect_all(); SOT trackers add
    init()/reset()."""

    #: tracker family: "pose" | "dino" | "sot"
    kind: str = "pose"

    #: Last per-frame RAW detection box as (x, y, w, h) in FRAME pixels
    #: (floats), or None when nothing was detected this frame.  SOT trackers
    #: populate it with the tracker's per-frame response BEFORE the size EMA
    #: smoothing - the preview uses it to draw the CURRENT ROI size (the
    #: EMA-smoothed Person.box is deliberately NOT used for the drawn box:
    #: with a slow EMA it visually barely drifts from the initial ROI).  Pose /
    #: dino trackers leave it as None (they have no per-frame ROI box to show).
    last_raw_box: Optional[Tuple[float, float, float, float]] = None

    #: Optional scene transform applied to EVERY frame before passing it to
    #: detect_all()/init() (set by the _setup_scene window in
    #: core/interface/identify.py for SOT trackers; None / identity for
    #: pose and dino trackers - the fast path in apply() returns the frame
    #: unchanged so pose/dino code paths stay as fast as before).  All
    #: downstream consumers run in the apply(frame) coordinate space.
    scene: Optional["SceneTransform"] = None  # type: ignore[name-defined]

    #: OPTIONAL second (static reference) ROI drawn with Ctrl+LMB in the SOT
    #: identify window (core/interface/identify.py).  Set to (x, y, w, h) in
    #: frame coordinates when the user drew a ps box; None when not selected.
    #: The caller (funscript_gen) clones a second SOT tracker from it for
    #: camera-drift compensation (see core/tracking/streaming.py).
    ps_roi: Optional[Tuple[int, int, int, int]] = None

    def detect_all(self, frame: np.ndarray) -> List[Person]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Best-score person's point p, or None when nothing was detected."""
        persons = self.detect_all(frame)
        if not persons:
            return None
        return max(persons, key=lambda p: p.score).point

    # ── SOT lifecycle: not used by per-frame detectors (no-op defaults) ──
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        return True

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
