"""
Scene ROI + view transform for SOT trackers.

The user sets this up in the ``_setup_scene`` window (see
core/interface/identify.py) BEFORE drawing the SOT tracking ROI.  All
downstream consumers (SOT tracker init / detect_all, calibrate, preview
overlay, matching, hold-over, post-pass) run in the coordinate system of
``SceneTransform.apply(frame)`` - the SOT tracker's init() / update() pair
receive the SAME transformed frame, so no coordinate remapping is required
downstream: the tracked point is still "the center of the SOT ROI" from the
tracker's perspective, and the stroke path is still "the fraction of the
path already traversed" in the warped frame's pixels.

Fast path (identity):
    When the user confirms the scene with the identity settings (zoom=1,
    angle=0, pan=0, no ROI), ``apply(frame)`` returns the SAME ndarray
    object (no copy, no cv2.warpAffine).  This is the "full scene, no
    transforms" case the user explicitly asked to remain as fast as the
    pre-feature code path.

Pipeline (order matches the UX: "transformation of the video" first, then
crop to a sub-region):
    1. zoom      - center-anchored scale (1x / 2x / 3x, discrete)
    2. rotate    - center-anchored rotation (15 deg steps, -180..180 clamp)
    3. pan       - output-space translation (10 px steps, user-controlled)
    4. crop      - the user-drawn scene ROI (a RECTANGLE in the
                    "zoom/rotate/pan" output space, pre-crop size equals
                    the input frame size)

After the crop, downstream sees a (possibly smaller) ndarray of shape
``(outH, outW, 3)`` - the SOT tracker's init() / update() operate in THAT
space and the stroke post-pass works in those pixels.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


#: allowed discrete zoom levels (1 = identity, 2 = 2x, 3 = 3x)
ZOOM_LEVELS = (1.0, 2.0, 3.0)
#: step for the view rotation (degrees, signed)
ANGLE_STEP_DEG = 15
#: step for the output-space pan (px, signed)
PAN_STEP_PX = 10
#: absolute cap on the pan offset (px) - prevents the pan running off to
#: infinity when the user keeps tapping an arrow.
PAN_CLAMP_PX = 2000.0


@dataclass
class SceneTransform:
    """Optional frame preprocessing (zoom + rotate + pan + crop) for SOT.

    See the module docstring for the pipeline and the identity fast-path.

    ``scene_roi`` is a user-drawn RECTANGLE in the OUTPUT (transformed,
    pre-crop) space: (x, y, w, h) ints in pixels.  ``None`` = full output
    (no crop).  Note the pre-crop output has the SAME size as the input
    frame (warpAffine with ``dsize=(w, h)``), so ``scene_roi`` indexes
    into a ``warpAffine`` result of shape (H, W, 3).

    ``ref_w`` / ``ref_h`` are the frame dimensions the transform was
    designed around; ``is_identity(shape=None)`` uses them to decide
    "the scene ROI covers the full frame" when no live frame is available.
    """
    zoom: float = 1.0
    angle_deg: float = 0.0
    pan_dx: float = 0.0
    pan_dy: float = 0.0
    scene_roi: Optional[Tuple[int, int, int, int]] = None
    ref_w: Optional[int] = None
    ref_h: Optional[int] = None

    def reset(self) -> None:
        """Restore the identity transform (full frame, no ops)."""
        self.zoom = 1.0
        self.angle_deg = 0.0
        self.pan_dx = 0.0
        self.pan_dy = 0.0
        self.scene_roi = None
        self.ref_w = None
        self.ref_h = None

    def is_identity(self, shape: Optional[Tuple[int, int]] = None) -> bool:
        """True when ``apply(frame)`` would return the input frame unchanged.

        ``shape`` (H, W), when given, is the ACTUAL frame shape — used to
        check "the scene ROI covers the full frame" against the right size.
        When omitted, falls back to the stored ``ref_w/ref_h`` (set on the
        first ``apply()``); if unset, a set ``scene_roi`` is treated as
        non-identity (conservative default).
        """
        if (self.zoom != 1.0 or self.angle_deg != 0.0
                or self.pan_dx != 0.0 or self.pan_dy != 0.0):
            return False
        if self.scene_roi is None:
            return True
        if shape is not None:
            h, w = shape
        else:
            if self.ref_w is None or self.ref_h is None:
                return False
            h, w = self.ref_h, self.ref_w
        return self.scene_roi == (0, 0, w, h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the preprocessed frame.

        Identity case: returns ``frame`` unchanged (SAME object — no copy,
        no warpAffine; this is the "maximum speed, no extra transformation"
        branch the user asked for when the ROI is the whole scene).

        Non-identity: zoom 1x/2x/3x + rotate about the frame center + pan,
        then a final crop to the user-drawn scene ROI if one was drawn.
        The returned array has the (possibly smaller) shape
        (roi_h, roi_w, 3) — downstream consumers (SOT tracker init /
        detect_all, calibrate, preview overlay, matching, hold-over,
        post-pass) all operate in THAT coordinate system; no remapping is
        needed because the tracker's init()/update() pair receive the SAME
        transformed frame.
        """
        h, w = frame.shape[:2]
        if self.is_identity(shape=(h, w)):
            return frame
        out = _warp_view(frame, self.zoom, self.angle_deg,
                         float(self.pan_dx), float(self.pan_dy))
        if self.scene_roi is not None:
            x, y, rw, rh = (int(v) for v in self.scene_roi)
            oh, ow = int(out.shape[0]), int(out.shape[1])
            x = max(0, min(ow - 1, x))
            y = max(0, min(oh - 1, y))
            x2 = max(x + 1, min(ow, x + rw))
            y2 = max(y + 1, min(oh, y + rh))
            out = out[y:y2, x:x2]
        # remember the reference frame size for shape-less is_identity()
        if self.ref_w is None or self.ref_h is None:
            self.ref_w, self.ref_h = int(w), int(h)
        return out

    def describe(self) -> str:
        """One-line human-readable description (logs / GUI status bar)."""
        if self.scene_roi is None:
            if (self.zoom == 1.0 and self.angle_deg == 0.0
                    and self.pan_dx == 0.0 and self.pan_dy == 0.0):
                roi = "full"
            else:
                roi = "full frame after transform"
        else:
            x, y, w, h = self.scene_roi
            roi = f"{w}x{h} @ ({x},{y})"
        return (f"zoom={self.zoom:g}  angle={self.angle_deg:+g}deg  "
                f"pan=({int(self.pan_dx):+d},{int(self.pan_dy):+d})  "
                f"roi={roi}")


def _warp_view(frame: np.ndarray, zoom: float, angle_deg: float,
               pan_dx: float, pan_dy: float) -> np.ndarray:
    """Center-anchored zoom + rotate + pan (output size = input size).

    Order: zoom, rotate (both about the input frame center), then pan.
    Uses INTER_LINEAR + BORDER_REPLICATE so zoom > 1 and rotated corners
    don't reveal a black band and the pan can slide the video inside its
    own extent.
    """
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # getRotationMatrix2D returns the 2x3 affine matrix for the combined
    # (rotate + scale) about `center` - the translation part keeps the
    # center fixed, which is the desired "origin = video center" behavior.
    r2 = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), float(zoom))
    # compose two 2x3 affine matrices (rotation+zoom then pan translation):
    #   [A|b] = [A_r | b_r]   (rotation + scale about the frame center)
    #   [C|d] = [I     | b_p] (pure translation = the pan)
    # product  [C|d] @ [A|b] = [C*A | C*b + d]
    A_r = r2[:, 0:2]
    b_r = r2[:, 2:3]
    A_p = np.identity(2, dtype=np.float64)
    b_p = np.array([[float(pan_dx)], [float(pan_dy)]], dtype=np.float64)
    M = np.hstack([A_p @ A_r, A_p @ b_r + b_p]).astype(np.float32)
    return cv2.warpAffine(frame, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

