"""
SOT (Single Object Tracking) tracker: OpenCV's built-in correlation
trackers - ViT / MIL / CSRT / KCF (all present in opencv-contrib >= 4.10,
verified installed: cv2 5.0.0).

The ROI is chosen MANUALLY - the identify window switches to "drag a box
around the point you want to track" (see core.interface.identify).

Flow after init:
  * detect_all() feeds the tracker with a frame (update()) and sets point p
    to the box CENTER (core.detectors.box.box_center) - the ROI was chosen
    manually around the zone to track, so its center is the tracked point;
  * on loss the tracker returns an EMPTY list - the existing HoldoverPoint
    (LK ghost) keeps the point alive for a few frames;
  * press R in the preview window to re-open the ROI selection and
    reset()/init() the tracker on a new box.

Size handling (scale-drift protection):
  OpenCV trackers (especially CSRT) re-estimate the box SIZE every frame and
  that per-frame size "breathing" shifts the point
  y = top + frac*height, producing comb-noise in the post-pass graph.  The
  camera can ALSO drift slowly over the video (scene scale changes), so the
  size must not be frozen outright: it is EMA-smoothed per frame
  (SCALE_EMA_ALPHA - slow drift is followed, fast breathing is damped) and
  FLOORED at MIN_ROI_W/H so the box can shrink but never below 30x30 px.
  The tracker's CENTER is taken as-is (it is the real motion signal) and IS the tracked point p,
  the point p is computed IN SUBPIXELS from unrounded geometry (same rule
  as GroundingDINO; Person.box ints are for drawing/IoU only).

Score (tracking-quality proxy, used in logs / matcher fallback):
  Euclidean displacement of the box CENTER (px): 1.0 while the step is
  within 0.5*min(w,h), then a linear decay floored at 0.05.  The old
  "IoU with the previous frame's box" proxy was misleading for a fast-
  translating box (plain translation already lowers IoU without the
  tracker having failed).
"""
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.config import TARGETS
from core.detectors.box import box_center
from core.detectors.person_tracker import PersonTracker
from core.types import Person
from core.logging import log

#: cv2 factory per algorithm name (all four available in cv2 5.0.0)
_ALGOS = {
    "vit": "TrackerVit_create",
    "mil": "TrackerMIL_create",
    "csrt": "TrackerCSRT_create",
    "kcf": "TrackerKCF_create",
}

# ── scale-drift protection (see module docstring) ─────────────────
#: lower bound for the smoothed ROI size: it may shrink, never below this
MIN_ROI_W = 30
MIN_ROI_H = 30
#: EMA coefficient on the tracker-reported size (smaller = stronger damping
#: of per-frame breathing); slow camera drift is still followed
SCALE_EMA_ALPHA = 0.1


class SotOpenCVTracker(PersonTracker):
    """OpenCV SOT tracker on a manually drawn ROI.
    point p = the box CENTER (the ROI defines the zone; --point / TARGETS
    only validates the choice, it has no vertical offset here).
    The emitted box follows the tracker CENTER and carries the EMA-smoothed
    size (floored at MIN_ROI_W/H) - scale-drift protection that keeps p
    stable frame to frame.

    ONNX weights:
      * sot-vit - `onnx_file` may be supplied as a bare filename (looked up
        in `models/vittrack/` then CWD, core.config.MODEL_DIRS["vittrack"])
        or a full path. When present we construct the
        tracker via  `cv2.TrackerVit_Params(); p.net = path;  TrackerVit_create(p)`
        - this is the documented OpenCV way.  When `onnx_file` is None the
        auto-download path (`ensure_sot_onnx`) is used, which stores
        `vitTracker.onnx` in `models/vittrack/` then copies it into the CWD.
      * sot-mil - `onnx_file` is IGNORED on purpose (cv2 5.x's
        `TrackerMIL_Params` exposes only sampler-parameters, not a `net`
        field, and MIL trains its own classifier online - multiple
        instance learning - so no external weight file is required at all).
      * sot-csrt / sot-kcf - self-contained; `onnx_file` is ignored."""

    kind = "sot"

    def __init__(self, algo: str = "vit", target: str = "mid-hips",
                 onnx_file: Optional[str] = None):
        if algo not in _ALGOS:
            raise ValueError(f"unknown SOT algo: {algo!r} (choose from {list(_ALGOS)})")
        if target not in TARGETS:
            raise ValueError(f"unknown person_target: {target!r} (choose from {TARGETS})")
        self.algo = algo
        self.target = target
        # ── ONNX model resolution (vit-only; mil/csr/kcf are self-contained)
        self.onnx_file = onnx_file
        if algo == "vit" and onnx_file:
            from core.detectors.models import vittrack_model_path
            self._onnx_path = vittrack_model_path(onnx_file)
            print(f"[sot-vit] using explicit ONNX weights: {self._onnx_path}")
        else:
            self._onnx_path = None
            if algo == "vit":
                # legacy path: ensure_sot_onnx auto-downloads vitTracker.onnx
                # into the CWD when missing.
                from core.detectors.models import ensure_sot_onnx
                ensure_sot_onnx(algo)
            # mil / csrt / kcf: nothing to do - the cv2 factory is called
            # with no parameters (mil trains online; csrt/kcf are built-in).
            if algo == "mil" and onnx_file:
                print(f"[sot-mil] ignoring --sot-model {onnx_file!r}: "
                      f"TrackerMIL uses online multiple-instance training "
                      f"(no external ONNX weights in cv2 5.x)")

        factory = getattr(cv2, _ALGOS[algo], None)
        if factory is None:
            raise RuntimeError(
                f"cv2.{_ALGOS[algo]} is not available in this OpenCV build "
                f"(have {cv2.__version__}); opencv-contrib-python is required")
        # ── build via the Params object when an explicit ONNX file is set ─
        # (this is the only documented way to point OpenCV at specific
        # weights - the no-arg factory relies on the file in the CWD)
        if algo == "vit" and self._onnx_path:
            try:
                params = cv2.TrackerVit_Params()
            except AttributeError as e:
                raise RuntimeError(
                    f"cv2.TrackerVit_Params() unavailable in cv2 "
                    f"{cv2.__version__}; pass --sot-model only on OpenCV "
                    f"versions that expose the params object.") from e
            params.net = self._onnx_path
            self._factory = lambda: cv2.TrackerVit_create(params)
        else:
            self._factory = factory
        self.tracker = self._factory()
        self._ready = False
        # EMA-smoothed ROI size + last reported center (seeded in init())
        self._w_s: Optional[float] = None
        self._h_s: Optional[float] = None
        self._prev_center: Optional[Tuple[float, float]] = None
        # raw per-frame tracker response (the preview draws this as the
        # CURRENT ROI size - see PersonTracker.last_raw_box)
        self.last_raw_box: Optional[Tuple[float, float, float, float]] = None

    # ── SOT lifecycle ─────────────────────────────────────────
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = roi
        # keep the ROI inside the frame (cv2 requires the box to fit)
        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        # OpenCV 5.x: Tracker.init() is VOID - success = no exception,
        # invalid params (empty frame, bad ROI) raise an OpenCV exception.
        # Do NOT check the return value (4.x returned a bool we ignored).
        self.tracker.init(frame, (int(x), int(y), int(w), int(h)))
        self._ready = True
        # seed the smoothed size with the manual ROI (floored at MIN_ROI)
        self._w_s = max(float(w), float(MIN_ROI_W))
        self._h_s = max(float(h), float(MIN_ROI_H))
        self.init_w = self._w_s
        self.init_h = self._h_s
        self._prev_center = (float(x + w / 2.0), float(y + h / 2.0))
        print(f"[sot] ROI set: ({x},{y},{w}x{h}); press R in the preview "
              f"to re-select if the track is lost")
        return True

    def reset(self) -> None:
        """Fresh tracker instance (the current one can keep stale state).
        Also clears the size-EMA / center history so a re-selected ROI
        re-seeds them cleanly."""
        self.tracker = self._factory()
        self._ready = False
        self._w_s = None
        self._h_s = None
        self._prev_center = None
        self.last_raw_box = None

    def detect_all(self, frame: np.ndarray) -> List[Person]:
        if not self._ready:
            return []  # not initialized yet (identify window not done)
        ok, box = self.tracker.update(frame)
        if not ok:
            # If OpenCV lost the object, we do not reset the state;
            # we return an empty list, letting the system perform Holdover
            return []
        self.last_raw_box = tuple(float(v) for v in box)
        # rx, ry - the top-left corner of the "dirty" drifting box from OpenCV
        # rw, rh - the sizes inflated by the neural network
        rx, ry = float(box[0]), float(box[1])
        rw, rh = max(1.0, float(box[2])), max(1.0, float(box[3]))
        # Compute the CLEAN geometric center found by the transformer
        cx, cy = rx + rw / 2.0, ry + rh / 2.0
         # ====================================================================
        # ELASTIC GEOMETRY WITH HARD LIMITS (BOUNDED EMA)
        # ====================================================================
        # Use a very soft alpha (0.02 instead of 0.1) so the size changes smoothly
        # and does not react to one-second neural-network hallucinations
        SMOOTH_ALPHA = 0.02
        # Smoothly pull the current size toward the predicted one
        target_w = SMOOTH_ALPHA * rw + (1.0 - SMOOTH_ALPHA) * self._w_s
        target_h = SMOOTH_ALPHA * rh + (1.0 - SMOOTH_ALPHA) * self._h_s
        # Force-clamp the sizes into a strict safety corridor.
        # For example: the size cannot become smaller than 80% or larger than 130% of the ORIGINAL,
        # which the user drew by hand on click (we saved it in init)
        # For this we need the initial sizes. In your code at init() they are placed into self._w_s
        # But since self._w_s is overwritten, we need to preserve the initial constants.
        # If you do not want to introduce new variables in init(), you can tie them to MIN_ROI:
        MAX_ALLOWED_MULTIPLIER = 1.35  # Maximum +35% inflation
        MIN_ALLOWED_MULTIPLIER = 0.75  # Minimum -25% shrinkage
        self._w_s = max(self.init_w * MIN_ALLOWED_MULTIPLIER,
                        min(target_w, self.init_w * MAX_ALLOWED_MULTIPLIER))
        self._h_s = max(self.init_h * MIN_ALLOWED_MULTIPLIER,
                        min(target_h, self.init_h * MAX_ALLOWED_MULTIPLIER))
        # Guarantee that we do not drop below the project's absolute minimum
        self._w_s = max(float(MIN_ROI_W), self._w_s)
        self._h_s = max(float(MIN_ROI_H), self._h_s)
        w, h = self._w_s, self._h_s
        # One-line per-frame diagnostic (kept quiet: the OLD 8-line Cyrillic
        # block used to go to stdout/stderr/the fallback log on EVERY frame
        # of a CSRT run, which under OFS Process.new + cv2 5.0 Win32 flooded
        # the message queue right around tracking_stream's destroyAllWindows
        # and made the first few cv2.waitKey() calls on a brand-new window
        # return -1 - which the RDP UI (core/interface/rdp_ui.py) mis-read as
        # "window closed externally" and silently returned ("Esc / short
        # track").  The diagnostic is kept, but collapsed to one line and in
        # English to satisfy the "no Cyrillic in core/" project rule.)
        log(f"[sot-{self.algo}] raw=({rx:.1f},{ry:.1f},{rw:.1f},{rh:.1f}) "
            f"center=({cx:.2f},{cy:.2f}) "
            f"w_s={self._w_s:.1f} h_s={self._h_s:.1f} "
            f"init_w={self.init_w:.0f} init_h={self.init_h:.0f}")
        # box = re-centred on the pure motion signal with the smoothed size
        box_f = (cx - w / 2.0, cy - h / 2.0, w, h)
        point = box_center(box_f)  # (cx, cy) - the clean motion signal of the tracked zone
        box_i = (int(round(box_f[0])), int(round(box_f[1])),
                 int(round(w)), int(round(h)))
        score = 1.0
        if self._prev_center is not None:
            d = math.hypot(cx - self._prev_center[0], cy - self._prev_center[1])
            budget = max(1.0, 0.5 * min(w, h))
            if d > budget:
                score = max(0.05, 1.0 - (d - budget) / max(1.0, min(w, h)))
        self._prev_center = (cx, cy)
        return [Person(box_i, point, score, kp=None, kp_conf=None)]


