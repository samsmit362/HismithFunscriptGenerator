"""
Tracker registry / factory.

Tracker names exposed to the CLI (-tracker) and the OFS GUI combo:
    yolo             - YOLO-pose (v8 / v11 / v26), the default
    mediapipe        - MediaPipe Pose (33 landmarks)
    groundingdino    - text-prompt boxes (pip install groundingdino)
    sot-vit / sot-mil / sot-csrt / sot-kcf
                     - OpenCV SOT trackers on a manually drawn ROI
    ostrack          - OSTrack (ONE-stream ViT, max-accuracy SOT) on the
                       ONNX weights from models/ (--ostrack-model), via
                       onnxruntime (lazy import, CUDA provider preferred)
    ollama           - Ollama VLM generation mode (NOT a per-frame tracker):
                       the user draws a fixed ROI, the model returns
                       per-sample depth_pct in strict JSON, and the raw
                       (t_sec, pos) track is produced.  Driven by
                       core/postpass/llm_gen.py (see the ``kind == "llm"``
                       branch in funscript_gen._run).

Heavy third-party imports are LAZY (inside each tracker's __init__), so
importing this registry does not load ultralytics / mediapipe / groundingdino.
"""
from typing import Type

from core.detectors.person_tracker import PersonTracker

#: name -> (module dotted path, class attribute)
_REGISTRY = {
    "yolo": ("core.detectors.yolo", "YoloTracker"),
    "mediapipe": ("core.detectors.media_pipe", "MediaPipeTracker"),
    "groundingdino": ("core.detectors.groundingdino", "GroundingDinoTracker"),
    "sot-vit": ("core.detectors.sot_opencv", "SotOpenCVTracker"),
    "sot-mil": ("core.detectors.sot_opencv", "SotOpenCVTracker"),
    "sot-csrt": ("core.detectors.sot_opencv", "SotOpenCVTracker"),
    "sot-kcf": ("core.detectors.sot_opencv", "SotOpenCVTracker"),
    "ostrack": ("core.detectors.sot_ostrack", "OSTrackTracker"),
    "gsam2": ("core.detectors.sot_grounded_sam_2", "GroundedSam2Tracker"),
    "ollama": ("core.detectors.ollama_tracker", "OllamaTracker"),
}

SOT_ALGOS = {"sot-vit": "vit", "sot-mil": "mil",
             "sot-csrt": "csrt", "sot-kcf": "kcf"}

TRACKERS = tuple(_REGISTRY.keys())


class UnknownTrackerError(ValueError):
    pass


def create_tracker(name: str, target: str = "mid-hips", **kw) -> PersonTracker:
    """Instantiate a tracker by registry name (lazy import).

    `target` = the body zone / prompt target for the point p:
      * pose trackers (yolo, mediapipe)  - core.config.TARGETS
      * groundingdino                   - core.detectors.groundingdino.DINO_TARGETS
                                          (independent of TARGETS by design:
                                           targets are free-text prompts)
      * SOT trackers                    - any string from TARGETS
                                          (used for logging only; point p
                                           = box CENTER)
    Each class validates its own target in __init__ (ValueError on mismatch),
    so this factory does a NO-OP check here.

    `kw` may carry: model (yolo), imgsz (yolo), fps (mediapipe),
                    variant / prompt (groundingdino),
                    onnx_file (sot-vit: explicit ONNX weights file;
                               sot-mil / csrt / kcf ignore it),
                    model (ollama: the Ollama model name for the generate call).
    """
    if name not in _REGISTRY:
        raise UnknownTrackerError(f"unknown tracker {name!r} (choose from {TRACKERS})")
    import importlib
    module_path, cls_attr = _REGISTRY[name]
    cls: Type[PersonTracker] = getattr(importlib.import_module(module_path), cls_attr)
    if name in SOT_ALGOS:
        return cls(algo=SOT_ALGOS[name], target=target, onnx_file=kw.get("onnx_file"))
    if name == "ollama":
        # The LLM mode has NO target: the crop ROI defines the zone.
        # ``model`` is forwarded so the tracker's __init__ can hold it.
        return cls(target=target, model=kw.get("model", ""), **{k: v for k, v in kw.items() if k != "model"})
    return cls(target=target, **kw)
