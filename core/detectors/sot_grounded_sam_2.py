"""
Grounded SAM 2 tracker - single-object silhouette tracking via
facebookresearch/sam2's `SAM2VideoPredictor`.  The most accurate of our
SOT-family trackers: it tracks the pixel-level MASK of the zone you draw,
not a bounding box - robust to a changing motion angle (the mask IS the
silhouette, which has no axis), partial occlusion, and close-up frames
that drop box trackers.

Lifecycle (identical UX to the other SOT - `kind = "sot"`):
  * identify window - press+hold LMB to draw a ROI around the zone to track
    (e.g. pelvis); Enter/Space = init; Q = abort;
  * detect_all(frame) -> List[Person] - one Person with the subpixel center
    of mass of the silhouette mask (clipped to the ROI);
  * R in the preview window -> `_identify_sot` re-opens the ROI, `reset()`
    re-seeds the predictor state and init() re-prompts it on the new box;
  * on a "lost" frame (score < SCORE_MIN) detect_all returns [] and the
    HoldoverPoint (LK ghost) keeps the point alive, same as sot-ostrack.

Model build (one-time, in __init__ - torch/sam2 imported LAZILY so the
registry stays cheap):
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(config, ckpt, device=device)
`build_sam2_video_predictor` returns a `SAM2VideoPredictor` whose methods
already carry `@torch.inference_mode()`.  On a Blackwell GPU we additionally
run inference under torch.autocast(bfloat16) (bf16 tensor cores, same
numerical range as fp32) - see `_call_autocast` below.

Streaming (per-frame) API - verified against `sam2/sam2_video_predictor.py`
(main branch) and the official `tools/vos_inference.py` loop:
    st = predictor.init_state(video_path=video, offload_video_to_cpu=False)
    predictor.add_new_mask(st, 0, obj_id, mask=roi_bool_mask)  # box-ROI prompt
    gen = predictor.propagate_in_video(st, start_frame_idx=0)
    for out_frame_idx, out_obj_ids, out_mask_logits in gen:
        # out_mask_logits is a LIST - one full-res (H,W) LOGIT tensor per
        # obj_id; we use out_mask_logits[0] (our single object, obj_id=1).
        mask = (out_mask_logits[0] > 0).detach().cpu().numpy().astype(bool)
        ...
`init_state` NEEDS a concrete `video_path` (an .mp4 file or a dir of JPEG
frames - see `load_video_frames` in `sam2/utils/misc.py`).  So this tracker
takes a `video_path` argument at construction (funscript_gen forwards
`args.video`).  The ROI you draw becomes a bool mask (1 inside, 0 outside)
that seeds frame 0 as the conditioning - no point clicks involved.

Score (Person.score, 0..1): |mask ∩ ROI| / |mask| - 1.0 when the silhouette
stays inside the ROI; falls off as it drifts outside.  Below SCORE_MIN the
tracker returns [] and the HoldoverPoint ghost takes over.

`--point` (core/config.TARGETS) is accepted for CLI / registry validation
but NOT applied - the ROI you draw defines the tracked zone; point p is the
subpixel center of mass of the mask inside that ROI (same rule as the other
SOT ROI trackers).

Frames backend (`--sam2-frames`, default `lazy`):
  * `lazy` (RECOMMENDED - constant RAM) - the SAM 2 `inference_state["images"]`
    is swapped for a `_LazyVideoFrames` adapter that decodes frames ON-DEMAND
    via decord (per-index random access). This removes SAM 2's "whole clip in
    one CPU tensor" upfront load - the 47 GB RAM spike a ~2 min 1080p clip hits
    before the GPU starts working - and keeps the GPU continuously fed over
    PCIe (SAM 2's documented `offload_video_to_cpu` streaming mode, same as
    tools/vos_inference.py).
  * `jpeg` (RAM grows with clip length) - the whole video is pre-exported to a
    JPEG folder and fed through SAM 2's built-in `AsyncVideoFrameLoader`.
    Useful when decord is slow for a particular codec, or as a comparison
    baseline. Note: `AsyncVideoFrameLoader` never frees already-decoded
    frames, so RAM ~ N × 12.58 MB (same order as the default MP4 path).
"""
import math
import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import List, Optional, Tuple

import numpy as np

from core.config import TARGETS
from core.detectors.person_tracker import PersonTracker
from core.types import Person
from core.logging import log


# ── lazy video adapter (backend 1: constant RAM, on-demand decord decode) ──
class _LazyVideoFrames:
    """Drop-in replacement for the `[N, 3, image_size, image_size]` CPU tensor
    SAM 2's `init_state` builds for MP4 videos (`sam2/utils/misc.py`
    `load_video_frames_from_video_file` stacks the WHOLE clip into one float32
    tensor: 12.58 MB/frame at image_size=1024 - ~47 GB for a 127 s / 30 fps
    1080p clip BEFORE the GPU starts working).

    How SAM 2 consumes the buffer (`sam2_video_predictor.py`):
        image = inference_state["images"][frame_idx].to(device).float()
                .unsqueeze(0)          # _get_image_feature, line ~713
        num_frames = inference_state["num_frames"]   # set from len(images)
    So `__len__` + `__getitem__(int)` is the entire contract. We return a CPU
    float32 `[3, sz, sz]` tensor normalized with the SAME ImageNet mean/std as
    sam2's own loaders (`misc.py`, both the MP4 and JPEG paths):
        x = (x / 255.0 - mean) / std,  mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),  channels in RGB order.
    decord (a sam2 hard dep, in requirements.txt) provides per-index random
    access - only the requested frame is decoded.  Keeps a 1-frame LRU for the
    warm R-rebind / seed-frame hot path.
    """

    #: ImageNet constants - MUST match sam2/utils/misc.py (applied AFTER /255.0).
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)

    def __init__(self, video_path: str, total: int, image_size: int):
        import torch  # lazy: keep this module importable without torch loaded
        # Try decord FIRST (its VideoReader(width, height) does the
        # decode+resize in C - fastest).  On ANY failure, fall back to
        # cv2.VideoCapture (OpenCV's FFmpeg) + F.interpolate on CPU.
        # The identify window proves OpenCV already reads this file,
        # so the fallback is guaranteed to work for every file the GUI
        # can preview.
        self._T = torch
        self._image_size = int(image_size)
        self._total = int(total)
        self._mean = torch.tensor(list(self.IMG_MEAN),
                                  dtype=torch.float32).reshape(3, 1, 1)
        self._std = torch.tensor(list(self.IMG_STD),
                                 dtype=torch.float32).reshape(3, 1, 1)
        self._kind: Optional[str] = None
        self._reader = None        # decord.VideoReader when kind == "decord"
        self._cap = None           # cv2.VideoCapture  when kind == "cv2"
        self._cache: "OrderedDict[int, object]" = OrderedDict()
        self._cache_cap = 1
        try:
            import decord  # lazy: already a sam2 hard dependency
            decord.bridge.set_bridge("torch")
            self._reader = decord.VideoReader(
                str(video_path), width=image_size, height=image_size)
            self._kind = "decord"
        except Exception as _de:
            self._reader = None
            import cv2 as _cv2
            cap = None
            for _backend in (_cv2.CAP_FFMPEG, _cv2.CAP_ANY):
                _c = _cv2.VideoCapture(str(video_path), _backend)
                if _c is not None and _c.isOpened():
                    cap = _c
                    break
                if _c is not None:
                    _c.release()
            if cap is None:
                raise OSError(
                    f"[sam2 lazy] neither decord nor cv2 could open "
                    f"{str(video_path)!r} (decord: {_de})"
                )
            self._cap = cap
            self._kind = "cv2"
            # cv2 does NOT resize for us - we F.interpolate on-demand.
            n = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            if n > 0:
                self._total = n
            print(f"[sam2 lazy] adapter fell back to cv2 "
                  f"(decord: {_de}); frames={self._total}", flush=True)

    def _decode_one(self, idx: int):
        """Read one frame normalized (RGB), returning [3, sz, sz] float32."""
        T = self._T
        if self._kind == "decord":
            frame = self._reader[idx]  # uint8 HWC RGB, already sz x sz
            t = frame.permute(2, 0, 1).float()
        else:
            import cv2 as _cv2
            import torch.functional as F
            cap = self._cap
            if cap is None:
                raise RuntimeError("cv2 capture was closed")
            if not cap.set(_cv2.CAP_PROP_POS_FRAMES, int(idx)):
                cur = int(cap.get(_cv2.CAP_PROP_POS_FRAMES))
                if idx < cur:
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
                while cur < idx:
                    ok, _f = cap.read()
                    if not ok:
                        raise OSError(
                            f"[sam2 lazy] cv2 seek failed at index {idx}"
                        )
                    cur += 1
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"[sam2 lazy] cv2 read failed at index {idx}")
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            t = T.from_numpy(np.ascontiguousarray(rgb)).float()
            t = t.permute(2, 0, 1).contiguous()
            t = F.interpolate(
                t[None], size=(self._image_size, self._image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )[0]
        t = t / 255.0
        t = t - self._mean
        t = t / self._std
        return t

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        cache = self._cache
        t = cache.get(idx)
        if t is not None:
            cache.move_to_end(idx)
            return t
        t = self._decode_one(idx)
        cache[idx] = t
        while len(cache) > self._cache_cap:
            cache.popitem(last=False)
        return t

    def close(self) -> None:
        self._cache.clear()
        self._reader = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


# ── JPEG export (backend 2: pre-extract, SAM 2 built-in async loader) ──────
def _export_jpeg_dir(video_path: str, dir_path: str) -> int:
    """Decode the whole video into `<dir>/<NNNNN>.jpg` (0-indexed, 5-digit
    zero-padded) - the exact format `sam2.utils.misc.load_video_frames_from_jpg_
    images` expects.  Returns the frame count.

    Used only when `frames_backend == "jpeg"` (the "comparison" mode).  The
    upfront decode cost is spread over a SAM 2 background thread, but RAM still
    grows with clip length (AsyncVideoFrameLoader never frees old frames).
    """
    import cv2  # already a hard dependency
    p = str(dir_path)
    os.makedirs(p, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"[sam2] jpeg backend: cannot open video for decode: "
            f"{video_path!r}")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(p, f"{n:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(
            f"[sam2] jpeg backend: 0 frames decoded from {video_path!r}")
    return n

#: default variant: best accuracy/size balance (see core.detectors.models.
#: SAM2_VARIANTS). `sam2.1-hiera-large` = max accuracy (fits RTX 5090 32GB);
#: `sam2.1-hiera-tiny` = fastest (~1.4 GB).
DEFAULT_VARIANT = "sam2.1-hiera-base-plus"

#: target-loss threshold: at least this fraction of the silhouette must sit
#: inside the ROI to count as "on target" (same role as OSTrack SCORE_MIN).
SCORE_MIN = 0.35

#: a mask smaller than this (px) is treated as noise -> [] -> ghost hold-over.
MIN_MASK_PX = 25

#: valid `frames_backend` values (--sam2-frames). Order = CLI choices list.
#:   "lazy" - decord on-demand decode, constant RAM (recommended, default)
#:   "jpeg" - pre-export to a JPEG folder + SAM 2 async loader (RAM grows)
FRAMES_BACKENDS = ("lazy", "jpeg")

#: EMA blend factor over the raw SAM 2 logit map (see detect_all).
#: `alpha` = weight on the PREVIOUS frame; `(1 - alpha)` = weight on the
#: current frame.  alpha=1.0 disables the smoothing (legacy); alpha=0.5
#: halves the weight of a single noisy frame.  The logit map IS the SAM 2
#: foreground score; smoothing it before the `> 0` binarisation removes
#: the ~10-30 px "breathing" of the silhouette boundary that otherwise
#: shows up as per-frame jitter in the centroid for large ROIs (pelvis).
DEFAULT_EMA_ALPHA = 0.5


# ── pure math (no torch / no model - unit-testable in _smoke_refactor) ────
def _mask_centroid_2d(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Subpixel (x, y) center of mass of a 2-D bool/float mask.

    O(N) - a single weighted mean over the foreground pixel coords.  This is
    the rotation-invariant tracked point (the silhouette itself has no axis),
    so a changing camera angle / body posture does NOT shift it.
    """
    ys, xs = np.nonzero(mask > 0)
    if xs.size < 1:
        return None
    return (float(xs.mean(dtype=np.float64)), float(ys.mean(dtype=np.float64)))


def _mask_bbox_center_2d(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Subpixel (x, y) center of the mask's bounding box (the "bbox" debug
    axis_mode - same rule as the other SOT ROI trackers). None if empty."""
    ys, xs = np.nonzero(mask > 0)
    if xs.size < 1:
        return None
    return (0.5 * (float(xs.min()) + float(xs.max())),
            0.5 * (float(ys.min()) + float(ys.max())))


def _mask_inside_roi_ratio(mask: np.ndarray,
                           roi: Tuple[int, int, int, int]) -> float:
    """|mask ∩ ROI| / |mask| - the tracker's "target-loss" quality score.
    1.0 = the silhouette is fully inside the ROI; falls off as it grows beyond
    / shrinks inside the box.  ROI is (x, y, w, h) in FRAME pixels."""
    if mask.size == 0:
        return 0.0
    total = int(mask.sum())
    if total == 0:
        return 0.0
    H, W = mask.shape
    x0 = max(0, int(roi[0]))
    y0 = max(0, int(roi[1]))
    x1 = min(W, int(roi[0]) + int(roi[2]))
    y1 = min(H, int(roi[1]) + int(roi[3]))
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inside = int(mask[y0:y1, x0:x1].sum())
    return float(inside) / float(total)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, w, h) int bbox of a mask (for the preview's raw-ROI box)."""
    ys, xs = np.nonzero(mask > 0)
    if xs.size < 1:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _ema_blend(prev, cur, alpha: float):
    """Frame-over-frame EMA blend on the raw SAM 2 logit tensor.

    Returns a new tensor; `prev` and `cur` are left alone.  alpha=1.0 (or
    prev=None) returns `cur` unchanged (legacy path - no smoothing);
    alpha=0.5 halves the weight of any single noisy frame and is what kills
    the ~10-30 px silhouette boundary breathing visible in the logs.
    """
    if alpha >= 1.0 or prev is None:
        return cur
    return alpha * prev + (1.0 - alpha) * cur


def _honest_score(peak: float,
                  peak_ref: Optional[float],
                  area_ratio: float) -> float:
    """Quality score that CAN degrade - replaces the legacy
    `|mask \u2229 ROI| / |mask|` self-comparison that read 1.0 every frame.

    Two independent signals, equal-weighted, both in [0, 1]:
      * sigmoid(peak - peak_ref)
          - peak_ref is the logit peak recorded on the seed frame (init()).
          - a frame with the same "objectness" as the seed scores 0.5
            (the sigmoid midpoint) - the score is a RELATIVE signal, not an
            absolute probability.
          - if peak drops substantially below peak_ref, the score falls.
      * area gate
          - mask_area / seed_roi_area, symmetric (ratio r and 1/r score the
            same).  r=1 -> 1.0, r=2 -> 0.25, r=10 -> ~0.  Catches a mask
            that has ballooned to a third of the frame (object lost) or
            collapsed to a small blob.

    Final = max(peak_score, area_score) so a strong signal on either axis
    still produces a valid score; below = both axes are weak -> drop.
    """
    if peak_ref is None:
        peak_score = 0.5
    else:
        # scale 0.5: SAM 2 peaks are usually in +3..+8 range on hits,
        # -3..0 on misses; a 3-logit drop = sigmoid(-1.5) ~ 0.18
        peak_score = 1.0 / (1.0 + math.exp(-0.5 * (peak - peak_ref)))
    if area_ratio <= 0.0:
        area_score = 0.0
    else:
        r = min(area_ratio, 1.0 / area_ratio)
        area_score = max(0.0, min(1.0, (r - 0.1) / (1.0 - 0.1)))
    return max(0.0, min(1.0, max(peak_score, area_score)))


class GroundedSam2Tracker(PersonTracker):
    """SAM 2 video-predictor on a manually drawn ROI (single object).

    kind = "sot" - identify window / R-rebind / holdover / preview share the
    exact code path used for the other SOT trackers.

    point p = subpixel center of mass of the silhouette mask (or mask's
    bounding-box center depending on axis_mode).  Rotation-invariant by
    construction - pixel-level tracking.

    `video_path` is REQUIRED: SAM 2's `init_state(video_path=...)` needs a
    concrete video source (either an MP4 file or a dir of JPEG frames —
    see sam2/utils/misc.py load_video_frames).  Pass the same path the
    identify / preview / streaming loops use (funscript_gen._run's
    `args.video`)."""

    kind = "sot"

    def __init__(self,
                 target: str = "mid-hips",
                 video_path: str = "",
                 variant: str = DEFAULT_VARIANT,
                 device: Optional[str] = None,
                 axis_mode: str = "cos",
                 score_min: float = SCORE_MIN,
                 min_mask_px: int = MIN_MASK_PX,
                 start_frame: int = 0,
                 frames_backend: str = FRAMES_BACKENDS[0],
                  ema_alpha: float = DEFAULT_EMA_ALPHA):
        if target not in TARGETS:
            raise ValueError(
                f"unknown person_target: {target!r} (choose from {TARGETS})")
        if axis_mode not in ("cos", "principal", "bbox"):
            raise ValueError(
                f"unknown axis_mode {axis_mode!r} (cos | principal | bbox)")
        if frames_backend not in FRAMES_BACKENDS:
            raise ValueError(
                f"unknown frames_backend {frames_backend!r} "
                f"(choose from {list(FRAMES_BACKENDS)})")
        if not video_path:
            raise ValueError(
                "video_path is required for SAM 2 (init_state needs it); "
                "funscript_gen must forward args.video here")
        from core.detectors.models import (SAM2_VARIANTS,
                                           sam2_config_path,
                                           sam2_weights_path)
        self.target = target
        self.variant = variant if variant in SAM2_VARIANTS else DEFAULT_VARIANT
        self.video_path = str(video_path)
        self.axis_mode = axis_mode
        self.frames_backend = frames_backend
        self.score_min = float(score_min)
        self.min_mask_px = int(min_mask_px)
        self.ema_alpha = float(ema_alpha)
        # SAM 2 tracks along temporal order of the video; if `--start 5:27`
        # is set, the identify window opens at frame start_frame and we
        # condition the mask THERE.  Propagating forward from there avoids
        # having to re-trace every frame before start_s (a real waste in
        # long videos).  start_frame=0 is the default (full video).
        self.start_frame = max(0, int(start_frame))

        import torch  # lazy: keep the registry import cheap (heavy)
        self._T = torch
        cfg_path = sam2_config_path(self.variant)
        ckpt_path = sam2_weights_path(self.variant)
        torch_dev = self._pick_device(torch, device)
        self._torch_device = torch_dev
        if torch_dev.type == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                print(f"[sam2] CUDA capability {cap}")
            except Exception:
                pass
        print(f"[sam2] building variant={self.variant} device={torch_dev}\n"
              f"       config={cfg_path}\n       ckpt  ={ckpt_path}")
        with torch.inference_mode():
            from sam2.build_sam import build_sam2_video_predictor
            self._predictor = build_sam2_video_predictor(
                config_file=cfg_path, ckpt_path=ckpt_path, device=torch_dev)
        self._predictor.eval()

        # bf16 autocast (Blackwell tensor cores) for the per-frame forward
        # pass.  Wrapped by _autocast() at the inference hot path; `build`
        # itself runs in fp32 (that's fine, one-time cost, not hot).
        self._use_bf16 = (torch_dev.type == "cuda"
                          and torch.cuda.is_bf16_supported())
        if self._use_bf16:
            print("[sam2] using torch.autocast(bfloat16) in detect_all")

        # streaming state (session dict created in init(); generator in _gen)
        self._state = None
        self._gen: Optional[object] = None   # propagate_in_video generator
        self._ready = False
        self._frame_seen = -1                 # last out_frame_idx we consumed
        self._obj_id = 1
        self._roi: Optional[Tuple[float, float, float, float]] = None
        self.last_raw_box: Optional[Tuple[float, float, float, float]] = None
        self._last_point: Optional[Tuple[float, float]] = None
        self.last_score = None                # float 0..1 or None

        # Backend scratch (created in init(), freed in reset()/close()):
        #   lazy - the _LazyVideoFrames adapter installed over state["images"]
        #          (holds the decord reader);  jpeg - a JPEG export folder.
        self._lazy_frames: Optional[object] = None
        self._jpeg_dir: Optional[str] = None

        # Set by the identify window before init() - the video-frame index
        # of the frame the user is drawing the ROI on.  `init()` pops it (or
        # uses self.start_frame as a fallback).  The streaming loop does NOT
        # set it (the identify window already did).  Optional on other
        # trackers (which ignore it).
        self.pending_seed_frame: Optional[int] = None
        self._orig_hw: Optional[Tuple[int, int]] = None
        # EMA over the raw logit map (None = not initialized; reset() clears
        # it; init() re-records as "prev" on the first detect_all frame).
        self._logit_prev = None
        # Reference peak on the seed frame - recorded in init() on the FIRST
        # mask that the predictor actually commits (the cond_frame_outputs).
        # Used by _honest_score().  None until the first detect_all succeeds.
        self._peak_ref: Optional[float] = None
        print(f"[sam2] tracker ready (score_min={self.score_min}, "
              f"axis={axis_mode}, frames_backend={self.frames_backend})")

    @staticmethod
    def _pick_device(torch, request: Optional[str]):
        """Resolve 'auto' / 'cuda' / 'cpu' to a torch.device.  'auto' =
        cuda:0 when torch.cuda.is_available() else cpu."""
        r = (request or "auto").lower()
        if r == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda", 0)
            return torch.device("cpu")
        if r == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is "
                "False. Install the CUDA build of torch (see "
                "requirements.txt) or pass device='cpu' to force CPU.")
        if r in ("cpu", "cuda"):
            return torch.device(r)
        raise RuntimeError(f"unknown device {request!r} (auto | cpu | cuda)")

    def _autocast(self):
        """A CM that wraps the call under torch.autocast(bfloat16) on CUDA
        when supported (see __init__ for the rationale).  A plain
        nullcontext on CPU or when build lacks bf16 support."""
        T = self._T
        if self._use_bf16:
            return T.autocast(device_type="cuda", dtype=T.bfloat16)
        return nullcontext()

    def _close_generator(self):
        """Defensive: close the generator (if SAM 2's `yield` frame is ever
        long-lived - e.g. when the tracker is reset before the video
        ends - `gen.close()` releases the SAM-side per-frame cache)."""
        if self._gen is not None:
            try:
                self._gen.close()
            except Exception:
                pass
            self._gen = None

    def _release_backend_scratch(self) -> None:
        """Free whatever the active frames_backend allocated in init():
        the lazy adapter (decord reader) or the JPEG export folder."""
        if self._lazy_frames is not None:
            try:
                self._lazy_frames.close()
            except Exception:
                pass
            self._lazy_frames = None
        if self._jpeg_dir is not None:
            import shutil
            try:
                shutil.rmtree(self._jpeg_dir, ignore_errors=True)
            except Exception:
                pass
            self._jpeg_dir = None

    # ── SOT lifecycle ───────────────────────────────────────────────────
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """Create the SAM 2 inference state and seed it with the ROI mask.

        Two frames backends (see module docstring / FRAMES_BACKENDS):
          * "lazy" (RECOMMENDED) - SAM 2 init_state on a 1-frame seed dir,
            then swap `state["images"]` for a `_LazyVideoFrames` adapter
            that decodes frames ON-DEMAND via decord.  Constant RAM.
          * "jpeg" - SAM 2 init_state on a full-video JPEG export (its
            built-in AsyncVideoFrameLoader streams from there; RAM grows
            with clip length).

        Both branches end with the same seed:
          1. `add_new_points_or_box` (a box prompt + a single center
             point prompt at vfi) - the strongest SAM 2 conditioning
             for the same ROI; a rectangle-only `add_new_mask` is a
             degraded case (the memory bank encodes "this region"
             rather than "this object at this region").
          2. `propagate_in_video` forward from vfi

        Fallback: if the SAM 2 build does not expose
        `add_new_points_or_box` (extremely old versions), we fall back
        to the legacy `add_new_mask` rectangle prompt.

        ROI is (x, y, w, h) in FRAME pixels (see core/interface/identify.
        py::_identify_sot).  Only the frame SHAPE is used for the cond
        mask (SAM 2's add_new_mask internally resizes it to
        `predictor.image_size`) - the live pixels of `frame` are
        redundant.

        Seed frame: `self.pending_seed_frame` if set (identify window),
        else `self.start_frame`.  This is duck-typed - other SOT trackers
        don't have either attribute, so the same `init(frame, roi)`
        signature works for all SOT.
        """
        H, W = frame.shape[:2]
        self._orig_hw = (H, W)
        x, y, w, h = roi
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(1, min(int(w), W - x))
        h = max(1, min(int(h), H - y))
        vfi = int(self.start_frame) if self.pending_seed_frame is None \
            else max(0, int(self.pending_seed_frame))

        # Branch on frames backend - the two strategies differ in how
        # the "images" buffer is populated, not in the seed mask or the
        # final propagate_in_video call.
        if self.frames_backend == "lazy":
            self._init_lazy()
        else:  # "jpeg"
            self._init_jpeg()

        # Common to both backends: seed the ROI at video frame vfi with a
        # STRONG prompt - a box AND a single center point (label=1,
        # positive).  This is the prompt style SAM 2 was TRAINED on
        # (`add_new_points_or_box`, see sam2/
        # sam2_video_predictor.py); a raw rectangle-only mask (the old
        # `add_new_mask` path) is a degraded case where the memory bank
        # encodes "this region" rather than "this object IN this region",
        # which is what caused the silhouette to breathe / fail to follow
        # a moving hand (see AGENTS.md gsam2 notes).
        #
        # The positive click is the ROI CENTER (cx, cy) - this is the
        # default the SAM 2 UI uses when you drag a box (it emits the box
        # AND a positive click at the last mouse position; we use the box
        # center for a stable, deterministic seed).
        #
        # Coordinates are expected in the SAME space as video frames:
        # add_new_points_or_box with normalize_coords=True divides by
        # state[video_height]/[video_width] - that matches the raw BGRH
        # pixels we receive in this init().
        try:
            T = self._T
            points_arr = T.tensor(
                [[[(x + 0.5 * w), (y + 0.5 * h)]]], dtype=T.float32)
            labels_arr = T.tensor([[1]], dtype=T.int32)
            box_arr = T.tensor([[x, y, x + w, y + h]], dtype=T.float32)
            with self._autocast():
                self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=vfi,
                    obj_id=self._obj_id,
                    points=points_arr,
                    labels=labels_arr,
                    box=box_arr,
                    clear_old_points=True,
                )
            print(f"[sam2] seeded via add_new_points_or_box (box + center "
                  f"point) at frame {vfi}")
        except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
            # Legacy SAM 2 build or unexpected arg contract - use the
            # rectangle-only prompt path (the old behavior).
            print(f"[sam2] add_new_points_or_box unavailable ({exc!r}); "
                  f"falling back to add_new_mask (rectangle-only prompt)")
            with self._autocast():
                cond_mask = np.zeros((H, W), dtype=bool)
                cond_mask[y:y + h, x:x + w] = True
                self._predictor.add_new_mask(
                    inference_state=self._state,
                    frame_idx=vfi,
                    obj_id=self._obj_id,
                    mask=cond_mask,
                )

        self._roi = (float(x), float(y), float(w), float(h))
        self._frame_seen = vfi
        self._gen = self._predictor.propagate_in_video(
            self._state, start_frame_idx=vfi)
        self._ready = True
        self.last_raw_box = (float(x), float(y), float(w), float(h))
        self._last_point = (float(x + w / 2.0), float(y + h / 2.0))
        self.last_score = 1.0
        # consume the pending seed (init() is the only place that uses it;
        # re-init after reset() will re-pick it from the identify window)
        self.pending_seed_frame = None
        print(f"[sam2] ROI set: ({x},{y},{w}x{h}) @ video_frame={vfi}; "
              f"backend={self.frames_backend}; "
              f"press R in the preview to re-select if the track is lost")
        return True

    def _init_lazy(self) -> None:
        """`frames_backend=="lazy"` - SAM 2 init_state on a 1-frame seed
        JPEG dir (12.58 MB only), then swap `state["images"]` for a
        `_LazyVideoFrames` adapter.  Decodes frames ON-DEMAND via decord
        (per-index random access) - constant RAM, no O(N) upfront tensor.

        SAM 2 only ever reads `state["images"][i]` and `state["num_frames"]`
        (see sam2/sam2_video_predictor.py `_get_image_feature` and
        `propagate_in_video`), so a `__len__`/`__getitem__` object is a
        valid stand-in.  The 1-frame seed still runs the frame-0 image
        feature warm-up inside init_state (harmless, cached)."""
        import tempfile
        import decord
        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(self.video_path)
        total = len(reader)
        if total == 0:
            raise RuntimeError(
                f"[sam2] lazy backend: 0 frames in {self.video_path!r}")
        image_size = int(getattr(self._predictor, "image_size", 1024))
        # 1) 1-frame seed dir so init_state has a valid video source.
        seed_dir = tempfile.mkdtemp(prefix="sam2_lazy_seed_")
        try:
            seed = reader[0].numpy()           # uint8 HWC RGB
            from PIL import Image
            Image.fromarray(seed).save(os.path.join(seed_dir, "00000.jpg"))
            with self._autocast():
                self._state = self._predictor.init_state(
                    video_path=seed_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )
        except BaseException:
            import shutil
            shutil.rmtree(seed_dir, ignore_errors=True)
            self._state = None
            raise
        # 2) Swap the 1-frame buffer for the full-video lazy adapter.
        #    SAM 2 reads state["images"][i] and state["num_frames"];
        #    our adapter satisfies both.
        adapter = _LazyVideoFrames(
            self.video_path, total=total, image_size=image_size)
        self._state["images"] = adapter
        self._state["num_frames"] = total

        # 3) Robust state patch: SAM 2.1 init_state pre-allocates lists and
        #    tensors based on num_frames=1. We must expand them to `total`
        #    otherwise add_new_mask/propagate will IndexError at frame > 0.
        s = self._state
        for k in ("point_inputs", "mask_inputs", "synced_mask_inputs"):
            if k in s and isinstance(s[k], list):
                s[k] = [None] * total
        import torch
        for k in ("frames_already_compressed", "frames_uncompressed_at_feats_res"):
            if k in s and isinstance(s[k], torch.Tensor):
                s[k] = torch.zeros((total,), dtype=s[k].dtype, device=s[k].device)

        # 4) The 1-frame seed dir is no longer needed - free it.
        import shutil
        shutil.rmtree(seed_dir, ignore_errors=True)

    def _init_jpeg(self) -> None:
        """`frames_backend=="jpeg"` - pre-export the whole video to a JPEG
        folder, then SAM 2's built-in AsyncVideoFrameLoader streams from
        it.  Upfront cost is spread across SAM 2's background thread, but
        RAM still grows with clip length (AsyncVideoFrameLoader never
        frees old frames).  Useful when decord is slow for a codec, or as
        a direct comparison baseline against the lazy backend."""
        import tempfile
        jpeg_dir = tempfile.mkdtemp(prefix="sam2_jpeg_")
        try:
            total = _export_jpeg_dir(self.video_path, jpeg_dir)
            with self._autocast():
                self._state = self._predictor.init_state(
                    video_path=jpeg_dir,
                    offload_video_to_cpu=True,
                    # offload_state_to_cpu=True: the AsyncVideoFrameLoader
                    # already keeps ALL frames in RAM (RAM ~ N * 12.58 MB
                    # for the video), so keep the per-frame memory features
                    # on CPU too.  Cost: ~10 % slower per frame (see the
                    # SAM 2 predictor docstring).
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )
            print(f"[sam2] jpeg backend: {total} frames exported to "
                  f"{jpeg_dir}")
        except BaseException:
            import shutil
            shutil.rmtree(jpeg_dir, ignore_errors=True)
            self._state = None
            raise
        self._jpeg_dir = jpeg_dir

    def reset(self) -> None:
        """Discard the current inference state + generator.  The tracker is
        still bound to the same `video_path`; the next init() re-creates
        (called by _identify_sot on Enter-failure / R-rebind)."""
        self._close_generator()
        if self._state is not None and self._predictor is not None:
            try:
                self._predictor.reset_state(self._state)
            except Exception:
                pass
        self._state = None
        self._release_backend_scratch()
        self._ready = False
        self._frame_seen = -1
        self._roi = None
        self._orig_hw = None
        self.last_raw_box = None
        self._last_point = None
        self.last_score = None
        # EMA / score reference state - a re-init from the identify window
        # means a FRESH object; the old peak reference is no longer valid.
        self._logit_prev = None
        self._peak_ref = None

    def close(self) -> None:
        self.reset()
        self._predictor = None
        if self._T is not None and getattr(self._T, "cuda", None) \
                and self._T.cuda.is_available():
            try:
                self._T.cuda.empty_cache()
            except Exception:
                pass

    # ── per-frame ────────────────────────────────────────────────────────
    def detect_all(self, frame: np.ndarray) -> List[Person]:
        """Read the NEXT frame from the SAM 2 generator and build a Person
        with the subpixel center of mass of the silhouette mask.

        The `frame` argument is NOT used (SAM 2 pre-loads the entire video
        in init_state; decord feeds the same bytes per index).

        One subtlety (R-rebind):  the streaming loop's cap.read() and the
        SAM 2 generator are BOTH aligned to the same video frame index —
        so for a mid-stream R-rebind (identify seeds on the CURRENT
        frame), the very FIRST yield equals the seed frame.  We detect
        that via `self._gen_peeked_seed` and SKIP the first yield when it
        matches - otherwise detect_all would emit the seed frame twice.
        """
        if not self._ready or self._gen is None or self._orig_hw is None:
            return []  # not initialized yet (identify window not done)
        # Pull the NEXT frame from the generator.  Each yield returns
        # (out_frame_idx, out_obj_ids, out_mask_logits) where:
        #   * out_frame_idx      - the frame's index in the video
        #   * out_obj_ids        - list of client obj ids (we have just one)
        #   * out_mask_logits    - list of full-res LOGIT tensors, same
        #                         order as out_obj_ids; use [0] (obj_id=1)
        # `>0` binarises the logit to the foreground mask.
        try:
            out_frame_idx, out_obj_ids, out_mask_logits = next(self._gen)
        except StopIteration:
            print(f"[sam2] generator exhausted (end of SAM's known frames); "
                  f"last consumed frame = {self._frame_seen}")
            self.last_raw_box = None
            self.last_score = None
            return []
        except Exception as e:
            print(f"[sam2] propagate_in_video error: "
                  f"{type(e).__name__}: {e}")
            self.last_raw_box = None
            self.last_score = None
            return []
        self._frame_seen = out_frame_idx

        # SAM 2.1 yields out_mask_logits as EITHER a 4-D tensor (N_obj, 1, H, W)
        # or a list of per-obj (H, W). Normalize to 2-D (H, W).
        try:
            if isinstance(out_mask_logits, (list, tuple)):
                m = out_mask_logits[0]
            else:  # torch.Tensor
                m = out_mask_logits[0]
                if getattr(m, "ndim", 2) == 3 and m.shape[0] == 1:
                    m = m[0]            # drop the [1, ...] channel dim
        except Exception as e:
            print(f"[sam2] logits unpack failed: {e}")
            self.last_raw_box = None
            self.last_score = 0.0
            return []
        if m is None:
            self.last_raw_box = None
            self.last_score = 0.0
            return []
        # EMA-blend the RAW LOGIT MAP before the >0 binarisation.  This is
        # what kills the per-frame silhouette-boundary "breathing" we see
        # in the logs (~10-30 px oscillation of the mask bbox / centroid
        # even when the target is stationary).  A single noisy frame's
        # contribution is damped by (1 - alpha); the smoothed map is what
        # gets thresholded and what the centroid / bbox / score read from.
        # alpha=1.0 disables this (legacy behaviour).
        try:
            cur_logits = m.detach().float()
        except Exception as e:
            print(f"[sam2] tensor cast failed: {e}")
            self.last_raw_box = None
            self.last_score = 0.0
            return []
        if self.ema_alpha >= 1.0:
            blended = cur_logits
        else:
            blended = _ema_blend(self._logit_prev, cur_logits, self.ema_alpha)
        try:
            mask = (blended > 0.0).detach().cpu().numpy().astype(bool)
        except Exception as e:
            print(f"[sam2] mask decode failed: {e}")
            self.last_raw_box = None
            self.last_score = 0.0
            return []
        # Record the peak BEFORE the mask.size check, so the peak_ref
        # reference is still calibrated even on a "mask too small" frame.
        try:
            cur_peak = float(blended.max().item())
        except Exception:
            cur_peak = float(cur_logits.max().item())

        # Resolution scaling: SAM 2 operates in its internal image_size space
        # (usually 1024x1024). To match the original frame ROI and return
        # correct UI coordinates, we scale the ROI DOWN for the score, and
        # scale the POINT/BBOX UP for the result.
        H_orig, W_orig = self._orig_hw
        H_mask, W_mask = mask.shape
        sx, sy = W_orig / float(W_mask), H_orig / float(H_mask)

        if int(mask.sum()) < self.min_mask_px:
            print(f"[sam2] frame {out_frame_idx}: mask too small "
                  f"({int(mask.sum())} px) - treating as lost")
            self.last_raw_box = None
            self.last_score = 0.0
            return []

        # 1) Point p: calculate in mask space, scale up to orig space.
        if self.axis_mode == "bbox":
            point_m = _mask_bbox_center_2d(mask)
        else:
            point_m = _mask_centroid_2d(mask)
        if point_m is None:
            self.last_raw_box = None
            self.last_score = 0.0
            return []
        point = (point_m[0] * sx, point_m[1] * sy)

        # 2) BBox: calculate in mask space, scale up to orig space.
        bbox_m = _mask_bbox(mask)
        if bbox_m is not None:
            bbox = (bbox_m[0] * sx, bbox_m[1] * sy, bbox_m[2] * sx, bbox_m[3] * sy)
            self.last_raw_box = bbox
            box_i = tuple(int(v) for v in bbox)
        else:
            self.last_raw_box = (float(point[0]), float(point[1]), 1, 1)
            box_i = (int(point[0]), int(point[1]), 1, 1)

        # 3) Score - an honest, RELATIVE signal (previously
        #    `|mask ∩ last_bbox| / |mask|` read ~1.0 every single frame, so
        #    the score could NEVER trigger the ghost hold-over; the loss
        #    detector was dead).  Two signals, both independent of the
        #    moving box:
        #      * peak = max of the EMA-blended logit map this frame
        #      * peak_ref = the peak recorded on the FIRST frame after init
        #        (the seed frame).  Relative drop => the model lost the object.
        #      * area = |this frame's mask| vs |the seed ROI area|; a mask
        #        that balloons to / collapses to a fraction of the ROI is
        #        a different-sized object (or lost).
        # The EMA state is advanced only on ACCEPTED frames (score >= min)
        # so a bad frame does not poison the smoothed baseline.
        seed_area = 0.0
        if self._roi is not None:
            seed_area = float(self._roi[2]) * float(self._roi[3])
        mask_area = int(mask.sum())
        area_ratio = (float(mask_area) / seed_area) if seed_area > 0.0 else 1.0
        if self._peak_ref is None:
            # First real frame since init - calibrate the reference peak.
            self._peak_ref = cur_peak
        score = _honest_score(cur_peak, self._peak_ref, area_ratio)

        if score < self.score_min:
            print(f"[sam2] frame {out_frame_idx}: "
                  f"score {score:.3f} < {self.score_min:.2f} (peak={cur_peak:.2f}"
                  f" ref={self._peak_ref:.2f} area_ratio={area_ratio:.2f}) - "
                  f"ghost hold-over active")
            self.last_raw_box = None
            self.last_score = float(score)
            return []

        # Accepted: commit this frame's logits as the next-frame EMA baseline.
        self._logit_prev = blended
        log(f"[sam2] frame {out_frame_idx} score:{score:.3f} peak:{cur_peak:.2f} "
            f"ref:{self._peak_ref:.2f} area_ratio:{area_ratio:.2f} "
            f"ema_alpha:{self.ema_alpha:.2f}\n"
            f"point:{point} bbox:{bbox} box_i:{box_i}\n"
            f"point_m:{point_m} bbox_m:{bbox_m}\n")

        self._last_point = point
        self.last_score = float(score)
        return [Person(box_i, point, float(score), kp=None, kp_conf=None)]


    # ── diagnostics ──────────────────────────────────────────────────────
    @property
    def last_frame_index(self) -> int:
        """The video-frame index of the last mask we decoded (-1 if none)."""
        return max(-1, int(self._frame_seen))
