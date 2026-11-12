"""Lightweight tracker for the LLM (Ollama) generation mode.

The ``OllamaTracker`` is NOT a per-frame detector (unlike YOLO / SOT /
DINO).  Its only role in the pipeline is to carry a ``kind="llm"`` field
so the branching code in ``funscript_gen._run()`` can tell "use the
Ollama VLM path" apart from the usual "tracking path".

Lifecycle:
  * ``kind == "llm"`` — the identify window becomes the ROI picker
    (``core/interface/llm_roi.py::select_llm_roi``);
  * NO ``init()`` / ``detect_all()`` / ``reset()`` — the frame is never
    passed through the tracker (the LLM sees the raw cropped ROI);
  * ``close()`` is a no-op.

The actual generation is driven by ``core/postpass/llm_gen.generate_track``
in ``funscript_gen._run()`` (the ``tracker.kind == "llm"`` branch).
"""
from typing import List, Optional, Tuple

import numpy as np

from core.detectors.person_tracker import PersonTracker
from core.types import Person


class OllamaTracker(PersonTracker):
    """Sentinel tracker: kind = "llm", no per-frame detection.

    ``target`` is ignored (there is no body-zone concept in LLM mode - the
    crop ROI defines the zone to analyze).  ``kw`` is also ignored.
    """
    kind = "llm"

    def __init__(self, target: str = "", **kw):
        self.target = target  # store for logging
        # optional: the user-selected Ollama model name (for the LLM call)
        self.model: str = kw.get("model", "") or ""
        # optional: the fixed crop ROI (x, y, w, h) in frame pixels.
        # set by the caller AFTER the ROI selection window closes.
        self.roi: Optional[Tuple[int, int, int, int]] = None

    # --- PersonTracker contract (never called in the LLM path) ---
    def detect_all(self, frame: np.ndarray) -> List[Person]:
        raise RuntimeError(
            "OllamaTracker.detect_all is not supported - the LLM mode "
            "bypasses per-frame detection entirely.  Use "
            "core.postpass.llm_gen.generate_track() instead.")
