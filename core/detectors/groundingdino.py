"""
GroundingDINO: text-prompt open-vocabulary detector.

No pose landmarks - the point p is a fixed vertical position inside the
detection box (TARGET_BOX_FRAC, same table as SOT).  The text prompt per
person_target is human-readable (the model reads it) and is deliberately
written as a VISIBLE SILHOUETTE / CONTOUR phrase ("the lower pelvic contour
of a person") instead of an abstract body zone ("pelvis area") - the
detector locks onto a crisp visible edge, not a fuzzy interior region.
Edit TARGET_PROMPT to aim a different phrase at any target.

Two official checkpoints, selected by the `variant` kwarg (CLI: --dino-model):
    swinb  - GroundingDINO-B / Swin-B, 56.7 zero-shot AP (default, most
             accurate; trained on RefCOCO / OpenImages / ODinW-35)
    swint  - GroundingDINO-T / Swin-T, 48.4 zero-shot AP (lighter)
The architecture config ships inside the installed `groundingdino`
package; only the weights (*.pth) are downloaded into the extension
models/groundingdino/ directory (see core.config.MODEL_DIRS).

Point math (subpixel): the model output is de-normalized to float pixel
coordinates; the point p is computed IN FLOATS from the raw box edges
before any rounding. Only Person.box (drawing / IoU continuity) is rounded
to ints. This removes the 1-2 px per-frame Y jitter the old
cv2.boxPoints(...).astype(int) path introduced.

The package ships as `rf-groundingdino` on PyPI (Roboflow's pure-Python
build, import name stays `groundingdino`) - all heavy imports are lazy so
the tracker registry import stays cheap.
"""
from typing import List, Optional, Tuple

import numpy as np

from core.detectors.models import (DEFAULT_GNDINO_VARIANT, GNDINO_VARIANTS,
                                   gndino_config_path, gndino_weights_path)
from core.detectors.person_tracker import PersonTracker
from core.types import Person
from core.logging import log

#: GroundingDINO targets = detection prompts.  This table is DINO's OWN
#: target list - it is deliberately INDEPENDENT of core.config.TARGETS
#: (the pose-tracker body zones), per design: DINO targets are free text
#: prompts, not keypoint zones.  The OFS GUI (main.lua) asks Python for
#: this list via `funscript_gen.py --list dino` and the CLI takes the
#: choice as --dino-target.  Single source of truth for both.
#: Phrases are silhouette/contour-oriented on purpose (see module
#: docstring); one entry per target = the single place to re-aim the
#: detection prompt for that target.
#
# Note: the dot at the end of a word ("pelvis .") is the official
# GroundingDINO tokenizer syntax that helps the model strictly separate
# class boundaries from the background
DINO_TARGETS = {
    "guy": "guy .",
    "girl": "girl .",
    "pelvis": "pelvis .",
    "butt": "butt .",
    "ass": "ass .",
    "crotch": "crotch .",
    "hip": "hip .",
    "thigh": "thigh .",
    "torso": "torso .",
    "head": "head .",
    "head": "head .",
    "left-hip": "the left outer hip curve of a body",
    "right-hip": "the right outer hip curve of a body",
    "thigh": "the visible upper thigh silhouette",
    "torso": "the side torso profile line of a person",
    "shoulders": "the shoulder line of a person",
    "head": "the outline of a person's head and hair",
    "ankle": "the visible ankle and lower shin of a person",
}

#: default DINO target (also used as the CLI --dino-target default and
#: the FIRST entry printed by `--list dino`, so GUI combo index 1 = it).
DEFAULT_DINO_TARGET = "pelvis"

#: back-compat alias (older imports / _smoke_refactor.py)
TARGET_PROMPT = DINO_TARGETS


def _round_box(x1: float, y1: float, x2: float, y2: float):
    """Round a float xyxy rect to a positive-size int (x, y, w, h)."""
    bx1, by1 = int(np.floor(x1)), int(np.floor(y1))
    bx2, by2 = int(np.ceil(x2)), int(np.ceil(y2))
    return (bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1))



class GroundingDinoTracker(PersonTracker):
    """Text-prompt open-vocabulary detector.

    `variant` selects the checkpoint (swinb default / swint) - see
    core.detectors.models.GNDINO_VARIANTS. Box-only (no landmarks): the
    point p is a user-selected percent position INSIDE the box
    (point_uv = (u_percent, v_percent), default (50, 50) = box center —
    "the CENTER of the ROI"; the identify window / _identify_dino lets
    the user point to an exact location by clicking inside the detection
    box), computed IN FLOATS from the unrounded box edges (subpixel, no
    per-frame Y jitter). Person.box carries the rounded int rect
    (drawing / IoU).

    Point p follows the box: on EVERY frame, once GroundingDINO has
    detected the box (wherever it has moved), p is placed at the same
    (u, v) percent position inside that box - the user's selection is
    RELATIVE, not absolute in the video frame (this is exactly what
    "the user can re-aim it" means: pick a different point once, it
    follows from then on).
    """

    kind = "dino"

    def __init__(self, target: str = DEFAULT_DINO_TARGET,
                 prompt: Optional[str] = None,
                 variant: str = DEFAULT_GNDINO_VARIANT,
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25):
        if target not in DINO_TARGETS:
            raise ValueError(f"unknown dino target: {target!r} "
                             f"(choose from {tuple(DINO_TARGETS)})")
        if variant not in GNDINO_VARIANTS:
            raise ValueError(f"unknown GroundingDINO variant {variant!r} "
                             f"(choose from {tuple(GNDINO_VARIANTS)})")
        from transformers import BertConfig, BertModel
        from transformers.utils.logging import disable_progress_bar
        from groundingdino.util.inference import Model  # lazy: pip install rf-groundingdino
        # Disable the "Loading weights" tqdm progress bar: under OFS
        # Process.new the stderr pipe has a closed parent end, and tqdm's
        # status_printer calls sys.stderr.flush() -> OSError [Errno 22]
        # while BertModel.from_pretrained runs.  This switches transformers
        # to the no-op EmptyTqdm so the build never touches stderr.
        disable_progress_bar()
        # transformers >= 5 removed BertModel.get_head_mask and changed the
        # positional signature of get_extended_attention_mask (the
        # rf-groundingdino BertModelWarper calls it with `device` as the 3rd
        # POSITIONAL arg).  Both are incompatible with the warper, which copies
        # these methods off the BertModel class at construction time - so we
        # patch once, before the model is built.
        if not getattr(BertModel, "_groundingdino_compat_shim", False):
            if not hasattr(BertModel, "get_head_mask"):
                def _get_head_mask(self, head_mask=None, num_hidden_layers: int = 12):
                    if head_mask is None:
                        return None
                    return head_mask
                BertModel.get_head_mask = _get_head_mask
            _orig_mask = BertModel.get_extended_attention_mask

            def _get_extended_attention_mask(self, attention_mask, input_shape, *args,
                                             **kwargs):
                import torch
                if args and isinstance(args[0], torch.device):
                    # the warper passes `device` positionally; the current
                    # transformers signature has no device param (the mask's
                    # device is derived from the tensor) - drop it
                    args = args[1:]
                return _orig_mask(self, attention_mask, input_shape,
                                  *args, **kwargs)

            BertModel.get_extended_attention_mask = _get_extended_attention_mask
            try:
                import dataclasses
                for f_ in dataclasses.fields(BertConfig):
                    if f_.name == "attention_head_mask" and f_.init:
                        f_.init = False  # suppress the "deprecated" ctor warning
            except Exception:
                pass
            BertModel._groundingdino_compat_shim = True
        self.target = target
        # Custom prompt overrides the target's built-in phrase (GUI
        # "Custom GroundingDINO prompt" text field -> CLI --dino-prompt);
        # an empty string falls back to DINO_TARGETS[target]
        custom = (prompt or "").strip()
        self.prompt = custom if custom else DINO_TARGETS[target]
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        # Point p inside the detection box, as PERCENTS (u along the box
        # width, v along the box height; 0 = left/top edge, 100 =
        # right/bottom edge).  Default = (50.0, 50.0) = the box CENTER
        # ("the CENTER of the ROI" - the chosen default).  The identify
        # window (_identify_dino) updates it via set_point_percent() - the
        # user clicks inside the box to place p exactly where wanted.
        self.point_percent: Tuple[float, float] = (50.0, 50.0)
        config = gndino_config_path(variant)
        weights = gndino_weights_path(variant)
        print(f"[dino] GroundingDINO '{variant}' - loading ...")
        print(f"[dino]   config  : {config}")
        print(f"[dino]   weights : {weights}")
        self.model = Model(model_config_path=config,
                           model_checkpoint_path=weights,
                           device="cuda")
        self.variant = variant

    def set_point_percent(self, u: float, v: float) -> None:
        """Place p at (u, v) PERCENTS inside the detected box (clamped 0..100).
        Called by the ROI window (identify._identify_dino) when the user
        clicks inside the detection box; detect_all() then uses this
        position on every subsequent frame (p follows the box)."""
        self.point_percent = (float(min(100.0, max(0.0, u))),
                              float(min(100.0, max(0.0, v))))

    def point_from_box(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """Point p (SUBPIXEL floats) at percent position inside a float xyxy rect."""
        u, v = self.point_percent
        return (x1 + (u / 100.0) * (x2 - x1),
                y1 + (v / 100.0) * (y2 - y1))

    def detect_all(self, frame: np.ndarray) -> List[Person]:

        # Step 1: GroundingDINO inference
        detections, _phrases = self.model.predict_with_caption(
            frame, self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold)

        xyxy = np.asarray(detections.xyxy, dtype=np.float64)
        conf = np.asarray(detections.confidence, dtype=np.float64) \
            if getattr(detections, "confidence", None) is not None \
            else np.ones(len(xyxy), dtype=np.float64)

        # ====================================================================
        # HARDCORE DEBUG GROUNDINGDINO
        # ====================================================================
        log("\n" + "="*60)
        log(f"[DINO-DEBUG] Current prompt: '{self.prompt}'")
        log(f"[DINO-DEBUG] Number of boxes found: {len(xyxy)}")
        log(f"[DINO-DEBUG] Text phrases (what the network recognized): {_phrases}")

        if len(xyxy) == 0:
            log("[DINO-DEBUG] WARNING: The model produced 0 detections! Object lost (dropped below box_threshold).")
        else:
            for idx in range(len(xyxy)):
                phrase_text = _phrases[idx] if idx < len(_phrases) else "N/A"
                log(f"  Box #{idx}:")
                log(f"    Coordinates XYXY: [{xyxy[idx,0]:.1f}, {xyxy[idx,1]:.1f}, {xyxy[idx,2]:.1f}, {xyxy[idx,3]:.1f}]")
                log(f"    Confidence: {conf[idx]:.4f} (Threshold: {self.box_threshold})")
                log(f"    Associated text: '{phrase_text}'")
        log("="*60 + "\n")
        # ====================================================================

        out: List[Person] = []
        for i in range(len(xyxy)):
            x1 = float(xyxy[i, 0]); y1 = float(xyxy[i, 1])
            x2 = float(xyxy[i, 2]); y2 = float(xyxy[i, 3])
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            # point p at the user-selected PERCENT position inside the
            # (unrounded, float) box - default (50, 50) = box center
            p = self.point_from_box(x1, y1, x2, y2)
            box = _round_box(x1, y1, x2, y2)
            out.append(Person(box, p, float(conf[i]),
                              kp=None, kp_conf=None))
        return out
