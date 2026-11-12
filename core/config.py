from dataclasses import dataclass

MOTION_MODES = ("straight", "arc", "parabola", "complex")
NORMALIZE_MODES = ("per-stroke", "global-max")

# ── models/ subdirectory map ──────────────────────────────────────
# Single source of truth for the folder layout under <ext_root>/models/.
# Each weight family lives in its own sub-directory so the folder can be
# safely extended over time with new backends without breaking the
# resolvers (core/detectors/models.py), the CLI (--list, --sot-model,
# --ostrack-model) or the OFS GUI (main.lua) - they all read from here.
# Only "ostrack" and "vittrack" expose ONNX weight pickers in the GUI;
# the other entries are auto-download caches (no user selection).
MODEL_DIRS = {
    "yolo": "yolo",
    "mediapipe": "mediapipe",
    "groundingdino": "groundingdino",
    "ostrack": "ostrack",       # OSTrack ONNX (user-provided .onnx)
    "vittrack": "vittrack",     # sot-vit ONNX (TrackerVit weights)
    "sam2": "sam2",             # SAM 2 video-predictor checkpoints (.pt,
                                # auto-downloaded - gsam2 tracker)
    # NOTE: sot-mil has NO model file (cv2 5.x MIL trains its own classifier
    # online, multiple-instance learning) - deliberately absent here.
}

# --point choices: the body zone tracked as the point p (a.k.a. person_target)
# left/right = the person's own left/right (COCO 11/12, mediapipe 23/24)
TARGETS = ("mid-hips", "left-hip", "right-hip", "thigh", "torso",
           "shoulders", "head", "ankle")

# ── LLM (Ollama) generator mode ────────────────────────────────────
# A VLM (multimodal Ollama model) replaces the tracking pipeline:
# the user draws a SINGLE fixed ROI on one frame, we sample every
# ~clip_s of the selected span, crop that ROI out of every sampled
# frame, shrink to thumb, base64-encode, send in ONE generate() call,
# and parse a STRICT JSON answer of the form
#     [{"frame": N, "depth_pct": X}, ...]
# where N is the 0-based index of the image in the batch and X is the
# depth of penetration in the PROJECT's terms (0% = all INSIDE = pos 0,
# 100% = all OUTSIDE = pos 100).  We compute the (t_sec, pos) track
# client-side: `t = start_s + N * clip_s`, `pos = depth_pct`.
# Direct mapping (no invert) keeps the prompt and the pos semantics on
# ONE source of truth.
LLM_DEFAULT_CLIP_S    = 0.5     # seconds per sampled image (GUI: "clip duration")
LLM_DEFAULT_NUM_CTX   = 200000  # explicit context budget (VRAM) for the VLM
LLM_DEFAULT_NUM_PRED  = 32000   # max response length (thoughts + JSON)
LLM_DEFAULT_NUM_BATCH = 2048    # batch size for image encoding (faster on GPU)
LLM_DEFAULT_THUMB     = 1920    # PIL thumbnail max side (keep aspect)
LLM_JPEG_QUALITY      = 90      # base64 size vs. detail compromise
LLM_TEMPERATURE       = 0.1     # conservative: this is structured output
LLM_REPEAT_PENALTY    = 1.1     # (same values the reference implementation used)
LLM_DEFAULT_PROMPT = (
"""
    You are analyzing a sequence of video frames in chronological order. The frames are consecutive: the time gap between any two neighboring frames is approximately 30 ms.

    Your task is to estimate the **thrust cycle stage** for each frame from the perspective of the **penetrating partner**.

    Imagine a full thrust cycle consists of two movements:
    1.  **Forward motion** (deeper) — the penetrating partner pushes inward.
    2.  **Backward motion** (shallower) — the penetrating partner retracts.

    In this context, “penetration depth” is not an absolute distance, but a **motion index**.

    Estimate the motion index as a percentage, where:
    *   **0%** = The tip is at the deepest point (maximum penetration, the penetrating partner is at the bottom of their cycle).
    *   **50%** = The tip is halfway between the deepest and shallowest points.
    *   **100%** = The tip is at the shallowest point (full retraction, the penetrating partner is at the top of their cycle).

    Thus, the **lower** the number, the **deeper** the penetration at that moment; the **higher** the number, the **shallower**.

    Note: two thrusts in ~0.5 seconds is a very fast tempo, so sudden jumps in depth are unlikely.

    The response must be in strict JSON format, only one array of objects without any additional comments:
    `[{"frame": N, "depth_pct": X}, ...]`
    where N is the frame index (starting from 0) and X is a whole integer from 0 to 100.
"""
)

# GroundingDINO (prompt boxes, no manually drawn ROI) has no landmarks, so p
# is a fixed position INSIDE the detection box: x = box center, y = top +
# TARGET_BOX_FRAC * box height.  0 = top edge, 1 = bottom edge.
# (SOT ROI trackers do NOT use this table - they track the ROI box center,
# see core.detectors.box.box_center.)
TARGET_BOX_FRAC = {
    "head": 0.15,
    "shoulders": 0.25,
    "torso": 0.45,
    "mid-hips": 0.65,
    "left-hip": 0.65,
    "right-hip": 0.65,
    "thigh": 0.80,
    "ankle": 0.95,
}


@dataclass
class Config:
    fps: float = 30.0
    keyframe_interval_ms: int = 42
    motion: str = "complex"                    # straight | arc | parabola | complex(raw path)
    normalize: str = "global-max"            # per-stroke=100% | global-max (default)
    output_range: int = 100
    inverted: bool = False
    latency_ms: int = 0
    point: str = "mid-hips"                   # the tracked person_target (see TARGETS)
