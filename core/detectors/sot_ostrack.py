"""
OSTrack (Object Tracking via Joint Generation and Appearance Modeling,
ECCV 2022, botaoye/OSTrack) - one-stream ViT-B SOT tracker on ONNX
(onnxruntime), run on a manually drawn ROI like the other SOT trackers.

The heaviest single-object tracker we ship: template+search are fed into ONE
ViT transformer (joint self-attention), which survives strong blur / partial
occlusion better than CSRT/KCF/ViT-OpenCV - the max-accuracy option for the
pelvis-zone tracking task.

ONNX interface (verified against the official repo code botaoye/OSTrack and
the l-sf/Track-trt export recipe):
    inputs  : "z" - template (1,3,128,128) or (1,3,192,192)
              "x" - search   (1,3,256,256) or (1,3,384,384)
    outputs : score_map (1,1,F,F)   - already sigmoid (clamped 1e-4..1-1e-4)
              size_map  (1,2,F,F)   - w/h in feature units (see decode math)
              offset_map(1,2,F,F)   - raw sub-cell offset
    where F = search_size / 16 (backbone stride).

Inference pipeline (mirrors the official lib/test/tracker/ostrack.py,
lib/train/data/processing_utils.sample_target, Preprocessor and the decoded
C++ reference in l-sf/Track-trt):
    1. crop a square around the last state center,
       crop_sz = ceil(sqrt(w*h) * factor), factor: 2.0 template,
       5.0 search (384 build) / 4.0 (256 build) - official TEST config;
    2. pad with BORDER_CONSTANT, resize to the network input size,
       resize_factor = net_size / crop_sz;
    3. BGR->RGB, /255, (x - mean) / std with ImageNet mean/std;
    4. argmax of the HANN-windowed score map (official motion constraint),
       decode cx/w at that cell;
    5. map_box_back into frame pixels (official formula).

Model source: NO official ONNX release (the project ships .pth.tar). The user
drops one or more exported .onnx files into the extension models/ostrack/
directory; the OFS GUI scans that folder (dir /b) and picks one, or the CLI
passes --ostrack-model.  `funscript_gen.py --list ostrack` prints the
available files (one per line; machine-readable).

Stability policy (same invariants as core.detectors.sot_opencv):
    * center = the real motion signal - taken as-is (float, subpixel);
    * size   = EMA (SCALE_EMA_ALPHA) + floor MIN_ROI_W/H - damps the
      per-frame size "breathing" (keeps the stable box the center is read from);
    * point p = the box CENTER (box_center) IN FLOATS from the unrounded
      geometry (the ROI the user drew defines the tracked zone; Person.box
      ints are for drawing/IoU only);
    * score < SCORE_MIN -> detect_all returns [] -> the existing
      HoldoverPoint (LK ghost) keeps the point alive for a few frames;
    * reset() clears the state - R in the preview re-selects the ROI.

`onnxruntime` is imported LAZILY in __init__ so the tracker registry import
stays cheap (same rule as ultralytics / mediapipe / groundingdino).
"""
import math
from typing import List, Optional, Tuple

import numpy as np

from core.config import TARGETS
from core.detectors.box import box_center
from core.detectors.person_tracker import PersonTracker
from core.detectors.sot_opencv import MIN_ROI_H, MIN_ROI_W, SCALE_EMA_ALPHA
from core.types import Person
from core.logging import log

# ── official constants (experiments/ostrack/*.yaml + lib/test/tracker) ───
#: ViT backbone stride -> feature grid = search_size / STRIDE
STRIDE = 16
#: ImageNet channel mean/std in RGB order (official Preprocessor)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
#: border fill for out-of-frame crops (official sample_target:
#: BORDER_CONSTANT, default value = 0/black)
_PAD_VALUE = 0
#: target-loss threshold on the (sigmoid) score map max - below this the
#: tracker reports "lost" (detect_all -> [] -> HoldoverPoint ghost)
SCORE_MIN = 0.3
#: crop area factor relative to the target box (official TEST config:
#: v384 -> 5.0 search / 4.0 for the v256 build); template is 2.0 everywhere
_TEMPLATE_FACTOR = 2.0
_SEARCH_FACTOR = {384: 6.0, 256: 5.0}
_SEARCH_FACTOR_DEFAULT = 4.0
#: clip_box margin: a prediction is clipped at least this far inside the edge
_FRAME_MARGIN = 10

# ── pure math (unit-testable without onnxruntime / a GPU) ─────────────────

def _hann2d(sz: int) -> np.ndarray:
    """2D Hann window exactly as the official hann1d(centered=True):
    0.5 * (1 - cos(2*pi/(sz+1) * i)), i = 1..sz; outer product per axis."""
    i = np.arange(1, sz + 1, dtype=np.float64)
    w = 0.5 * (1.0 - np.cos((2.0 * math.pi / (sz + 1)) * i))
    return np.outer(w, w)


def decode_search_maps(score_map: np.ndarray, size_map: np.ndarray,
                       offset_map: np.ndarray, search_size: int):
    fs = search_size // STRIDE
    score = np.asarray(score_map, dtype=np.float32)
    while score.ndim > 2:
        score = score.squeeze(0)
    # RETURN THE SMOOTHED HANN WINDOW:
    window_influence = 0.40  # Coefficient from the official repository
    hann = _hann2d(fs)
    response = (1.0 - window_influence) * score + window_influence * (score * hann)
    iy, ix = np.unravel_index(int(np.argmax(response)), score.shape)
    # offset/size arrive as (1, 2, F, F) or (2, F, F) or already-squeezed
    # (2, F, F): C=0 -> w / x-offset, C=1 -> h / y-offset
    off = np.asarray(offset_map, dtype=np.float32)
    siz = np.asarray(size_map, dtype=np.float32)
    while off.ndim > 3:
        off = off.squeeze(0)
    assert off.ndim == 3 and off.shape[0] == 2, f"offset_map shape {off.shape}"
    while siz.ndim > 3:
        siz = siz.squeeze(0)
    assert siz.ndim == 3 and siz.shape[0] == 2, f"size_map shape {siz.shape}"
    cx = (float(ix) + float(off[0, iy, ix])) / float(fs)
    cy = (float(iy) + float(off[1, iy, ix])) / float(fs)
    w = float(siz[0, iy, ix]) / float(fs)
    h = float(siz[1, iy, ix]) / float(fs)
    # score_map is already sigmoid (clamped 1e-4..1-1e-4) in the model
    s = float(np.clip(score[iy, ix], 1e-4, 1.0 - 1e-4))
    return cx, cy, w, h, s, iy, ix


def map_box_back(cx_s: float, cy_s: float, w_s: float, h_s: float,
                 resize_factor: float, search_size: int,
                 prev_cx: float, prev_cy: float
                 ) -> Tuple[float, float, float, float]:
    """Official map_box_back: search-normalized box -> frame pixels.
    cx_frame = cx_s * (S / resize_factor) + (prev_cx - S / (2*resize_factor))
    where S = search_size (same for the size)."""
    s = search_size / resize_factor
    half_side = 0.5 * s
    cx = cx_s * s + (prev_cx - half_side)
    cy = cy_s * s + (prev_cy - half_side)
    return cx, cy, w_s * s, h_s * s


# ── patch helpers (official sample_target preprocessing) ──────────────────

def _extract_patch(frame: np.ndarray, cx: float, cy: float,
                   target_w: float, target_h: float,
                   factor: float, output_sz: int) -> Tuple[np.ndarray, float]:
    """Square crop (area = factor^2 * target area) around (cx, cy), padded to
    the frame border, resized to output_sz x output_sz.
    Returns (patch HWC uint8, resize_factor = output_sz / crop_sz).
    Mirrors official sample_target (BORDER_CONSTANT 0, resize INTER_LINEAR).
    """
    crop_sz = int(math.ceil(math.sqrt(max(target_w, 1.0) * max(target_h, 1.0))
                            * factor))
    if crop_sz < 1:
        raise ValueError("OSTrack: target box too small "
                         f"({target_w:.1f}x{target_h:.1f})")
    x1 = int(round(cx - crop_sz * 0.5))
    y1 = int(round(cy - crop_sz * 0.5))
    H, W = frame.shape[:2]
    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(0, x1 + crop_sz - (W - 1))
    y2_pad = max(0, y1 + crop_sz - (H - 1))
    crop = frame[y1 + y1_pad:y1 + crop_sz - y2_pad,
                 x1 + x1_pad:x1 + crop_sz - x2_pad]
    if x1_pad or y1_pad or x2_pad or y2_pad:
        crop = np.pad(crop,
                      ((y1_pad, y2_pad), (x1_pad, x2_pad), (0, 0)),
                      mode="constant", constant_values=_PAD_VALUE)
    import cv2  # cv2 is already a hard dependency of the project
    patch = cv2.resize(crop, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR)
    return patch, output_sz / float(crop_sz)


def _to_tensor(patch: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> (1,3,H,W) float32, (x/255 - mean)/std, RGB order
    (official Preprocessor.process)."""
    img = patch[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    img = img / 255.0
    mean = np.asarray(_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)


# ── tracker ───────────────────────────────────────────────────────────────

class OSTrackTracker(PersonTracker):
    """OSTrack ONNX single-object tracker on a manually drawn ROI.

    kind = "sot" -> the identify window uses the shared ROI-drag mode and R
    in the preview re-initializes (reset()/init()), exactly like the other
    SOT trackers. Box-only output: point p = the box CENTER (box_center),
    SUBPIXEL float - the ROI the user drew defines the tracked zone.

    `model` = ONNX file path (models/ostrack/ directory, see
    core.detectors.models.ostrack_model_path); `onnxruntime` is imported
    lazily here so the tracker registry import stays cheap.
    """

    kind = "sot"

    def __init__(self, target: str = "mid-hips", model: Optional[str] = None,
                 score_min: float = SCORE_MIN):
        if target not in TARGETS:
            raise ValueError(f"unknown person_target: {target!r} (choose from {TARGETS})")
        from core.detectors.models import ostrack_model_path
        path = ostrack_model_path(model)
        import onnxruntime as ort  # LAZY: registry import stays cheap
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            avail = list(ort.get_available_providers())
        except Exception:
            avail = ["CPUExecutionProvider"]
        providers = [p for p in ("CUDAExecutionProvider",
                                 "OpenVINOExecutionProvider",
                                 "CPUExecutionProvider") if p in avail] \
            or ["CPUExecutionProvider"]
        print(f"[ostrack] loading ONNX: {path}")
        self._session = ort.InferenceSession(path, sess_options=opts,
                                             providers=providers)
        print(f"[ostrack] active provider: {self._session.get_providers()[0]}")

        ins = self._session.get_inputs()
        if len(ins) != 2:
            raise RuntimeError(f"OSTrack ONNX must have exactly 2 inputs "
                               f"(template, search), got {len(ins)}: "
                               f"{[i.name for i in ins]}")
        # template = the SMALLER spatial input
        def _spatial(inp):
            dims = [d for d in inp.shape if isinstance(d, int) and d > 1]
            return dims[-1] if dims else 0
        a, b = ins[0], ins[1]
        if _spatial(a) <= _spatial(b):
            self._t_in, self._s_in = a, b
        else:
            self._t_in, self._s_in = b, a
        self.template_size = max(1, int(_spatial(self._t_in)))
        self.search_size = max(1, int(_spatial(self._s_in)))
        self.feat_sz = max(1, self.search_size // STRIDE)
        print(f"[ostrack] template={self.template_size}px "
              f"search={self.search_size}px (feature grid "
              f"{self.feat_sz}x{self.feat_sz})")

        # outputs: score_map / size_map / offset_map - by name (official
        # export names), falling back to the default export order
        out_names = [o.name.lower() for o in self._session.get_outputs()]
        log(f"[DEBUG] out_names:{out_names}")
        if len(out_names) < 3:
            raise RuntimeError(f"OSTrack ONNX must expose score/size/offset "
                               f"maps, got {out_names}")
        def _pick(key: str, fallback: int) -> int:
            for i, n in enumerate(out_names):
                if key in n:
                    return i
            return fallback
        self._i_score = _pick("score", 0)
        self._i_size = _pick("size", 1)
        self._i_offset = _pick("offset", 2)
        if len({self._i_score, self._i_size, self._i_offset}) != 3:
            raise RuntimeError(f"OSTrack ONNX outputs ambiguous: {out_names}")

        self.target = target
        self.score_min = float(score_min)
        self.search_factor = _SEARCH_FACTOR.get(self.search_size,
                                                _SEARCH_FACTOR_DEFAULT)
        print(f"[ostrack] search_factor={self.search_factor} "
              f"score_min={self.score_min}")

        # ── state (subpixel floats) ─────────────────────────────
        self._ready = False
        self._template: Optional[np.ndarray] = None
        self._cx: Optional[float] = None   # last state center (frame px)
        self._cy: Optional[float] = None
        self._w = 0.0                      # EMA-smoothed box size
        self._h = 0.0
        # raw per-frame response (the preview draws this as the CURRENT ROI
        # size - see PersonTracker.last_raw_box)
        self.last_raw_box: Optional[Tuple[float, float, float, float]] = None

    # ── SOT lifecycle ───────────────────────────────────────────
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = roi
        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        cx, cy = x + w / 2.0, y + h / 2.0
        self._template, _ = _extract_patch(
            frame, cx, cy, w, h, _TEMPLATE_FACTOR, self.template_size)
        self._cx, self._cy = float(cx), float(cy)
        # seed the smoothed size with the manual ROI (floored at MIN_ROI)
        self._w = max(float(w), float(MIN_ROI_W))
        self._h = max(float(h), float(MIN_ROI_H))
        # 1. Save the initial sizes at initialization (if not yet saved in init)
        self.init_w = self._w 
        self.init_h = self._h
        self._ready = True
        print(f"[ostrack] ROI set: ({x},{y},{w}x{h}); press R in the preview "
              f"to re-select if the track is lost")
        return True

    def reset(self) -> None:
        """Fresh state (R in the preview re-selects the ROI and re-inits)."""
        self._ready = False
        self._template = None
        self._cx = self._cy = None
        self._w = self._h = 0.0
        self.last_raw_box = None

    def close(self) -> None:
        self._session = None

    # ── per-frame ───────────────────────────────────────────────
    def detect_all(self, frame: np.ndarray) -> List[Person]:
        if not self._ready or self._template is None:
            return []  # not initialized yet (identify window not done)
        box, score = self._track(frame)
        if score < self.score_min:
            self.last_raw_box = None
            print(f"[ostrack] score {score:.3f} < {self.score_min:.2f} - "
                  f"target marked lost this frame (ghost hold-over active)")
            return []  # -> HoldoverPoint ghost keeps the point alive
        # point p: the box CENTER - the ROI was drawn around the zone to
        # track, so its center is the tracked point (SUBPIXEL floats; box
        # ints are for drawing/IoU only)
        point = box_center(box)
        box_i = (int(round(box[0])), int(round(box[1])),
                 int(round(box[2])), int(round(box[3])))
        return [Person(box_i, point, float(score), kp=None, kp_conf=None)]

    def _track(self, frame: np.ndarray) -> Tuple[Box, float]:
        """One inference step -> (box_xywh in frame px, float score)."""
        if self._template is None or self._cx is None or self._cy is None:
            raise RuntimeError("OSTrack tracker used before init()")
        H, W = frame.shape[:2]
        # 1. search patch around the LAST state center (official sample_target)
        search, resize_factor = _extract_patch(
            frame, self._cx, self._cy, self._w, self._h,
            self.search_factor, self.search_size)
        # 2. inference (inputs/outputs by name - robust to export order)
        feeds = {self._t_in.name: _to_tensor(self._template),
                 self._s_in.name: _to_tensor(search)}
        outs = self._session.run(None, feeds)
        # ==========================================
        # TEMPORARY HARD DEBUG (insert here)
        # ==========================================
        log("\n" + "="*50)
        log(f"[DEBUG] Number of ONNX outputs: {len(outs)}")
        for idx, out in enumerate(outs):
            log(f"[DEBUG] Output #{idx}: shape={out.shape}, dtype={out.dtype}, min={np.min(out):.4f}, max={np.max(out):.4f}")
        score_arr = np.asarray(outs[self._i_score], dtype=np.float32)
        log(f"[DEBUG] Peak confidence in the pelvis search: {np.max(score_arr):.4f}")
        log(f"[DEBUG] self._i_score:{self._i_score} self._i_size:{self._i_size} self._i_offset:{self._i_offset}")
        log("="*50 + "\n")
        # ==========================================
        # score_arr = np.asarray(outs[self._i_score], dtype=np.float32)
        # size_arr = np.asarray(outs[self._i_size], dtype=np.float32)
        # off_arr = np.asarray(outs[self._i_offset], dtype=np.float32)
        score_arr = np.squeeze(outs[self._i_score], axis=0)   # Now strictly shape (1, 24, 24) or (24, 24)
        size_arr = np.squeeze(outs[self._i_size], axis=0)     # Now strictly shape (2, 24, 24)
        off_arr = np.squeeze(outs[self._i_offset], axis=0)    # Now strictly shape (2, 24, 24)
        # Extra squeeze for the score map, in case a channel is left over
        if score_arr.ndim == 3:
            score_arr = np.squeeze(score_arr, axis=0)         # Now strictly (24, 24)
        # 3. decode (hann-windowed argmax + subcell offset - official math)
        cx_s, cy_s, w_s, h_s, score, iy, ix = decode_search_maps(
            score_arr, size_arr, off_arr, self.search_size)
        cx, cy, w, h = map_box_back(cx_s, cy_s, w_s, h_s, resize_factor,
                                    self.search_size, self._cx, self._cy)
        # ====================================================
        # DECODER COORDINATE DEBUG
        # ====================================================
        log(f"[DECODER] Grid peak: iy={iy}, ix={ix} | Normalized center in patch: cx_s={cx_s:.4f}, cy_s={cy_s:.4f}")
        log(f"[DECODER] Absolute frame center BEFORE clipping: cx={cx:.2f}, cy={cy:.2f}")
        log(f"[DECODER] Current tracker state: self._cx={self._cx:.2f}, self._cy={self._cy:.2f}")
        # ====================================================

        # 4. Soft elastic size re-computation (EMA)
        # Alpha 0.015 means the size adapts very smoothly,
        # completely ignoring one-second spikes and skin hallucinations
        SMOOTH_ALPHA = 0.015

        adapted_w = SMOOTH_ALPHA * w + (1.0 - SMOOTH_ALPHA) * self._w
        adapted_h = SMOOTH_ALPHA * h + (1.0 - SMOOTH_ALPHA) * self._h

        # 5. Safety corridor relative to the user's initial ROI
        # Allow the pelvis to visually shrink by 15% or expand by 30%
        if hasattr(self, 'init_w') and self.init_w is not None:
            self._w = max(self.init_w * 0.85, min(adapted_w, self.init_w * 1.30))
            self._h = max(self.init_h * 0.85, min(adapted_h, self.init_h * 1.30))
        else:
            # Fallback, in case init_w was not passed through for some reason
            self._w = adapted_w
            self._h = adapted_h

        # Guarantee that we do not compress below the project's limits
        self._w = max(float(MIN_ROI_W), self._w)
        self._h = max(float(MIN_ROI_H), self._h)

        # 6. Safe center clipping based on OUR new adapted size
        cx = max(_FRAME_MARGIN + self._w / 2.0, min(cx, W - self._w / 2.0 - _FRAME_MARGIN))
        cy = max(_FRAME_MARGIN + self._h / 2.0, min(cy, H - self._h / 2.0 - _FRAME_MARGIN))

        # 7. Update the state for the next frame (_extract_patch will take the new size!)
        self._cx, self._cy = cx, cy

        box = (cx - self._w / 2.0, cy - self._h / 2.0, self._w, self._h)
        return box, score