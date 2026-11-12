# Hismith Funscript Generator

An OpenFunscripter (OFS) extension that
generates FunScript keyframes from a video using AI pose tracking. The generated script
drives a Hismith sex machine:

- `pos = 0`   → maximum forward extension (fully inserted)
- `pos = 100` → maximum backward extension (fully retracted)

The extension tracks the moving **point p** of the *active* partner (selected
zone) frame-by-frame, segments the motion into forward/back strokes, and maps each frame
to a 0..100 position. The result is imported into the current OFS timeline, where the
points can be further edited by hand.

## How it works

1. **Identify partners** — a window on the first frame lets you pick the active and
   passive partner (pose / mediapipe trackers), or draw an ROI (SOT / OSTrack), or pick
   a text-prompt detection (GroundingDINO).
2. **Thrust calibration** — mark `b` (full retraction, `pos = 100`) and `f` (full insertion,
   `pos = 0`) on a frame of the active partner; these two points define the thrust axis.
3. **Streaming collect** — every frame is previewed (960x540, OpenCV) and the raw
   per-frame track of point p is collected (with optional optical-flow refinement and
   identity hold-over).
4. **Post-pass** — the track is segmented into strokes, an adaptive axis is fit to each
   stroke, and a position per frame is computed (motion mode + normalization).
5. **RDP smoothing** — an interactive window lets you tune the epsilon of the
   Ramer–Douglas–Peucker simplification.
6. **FunScript output** — one keyframe per `(time, pos)` pair is written to
   `ofs_result.json` and imported into the OFS timeline.

## Tracker backends

| Tracker          | Kind | Description                                                    |
|------------------|------|----------------------------------------------------------------|
| `yolo`           | pose   | YOLOv8 / v11 / v26 pose (COCO-17 keypoints)                   |
| `mediapipe`      | pose   | MediaPipe pose (33 landmarks)                                 |
| `groundingdino`  | dino   | GroundingDINO — open-vocabulary text-prompt detection         |
| `sot-vit`        | sot    | OpenCV `TrackerVit` (ONNX) on a drawn ROI                     |
| `sot-mil`        | sot    | OpenCV `TrackerMIL` (online multiple-instance learning)       |
| `sot-csrt`       | sot    | OpenCV `TrackerCSRT` on a drawn ROI                           |
| `sot-kcf`        | sot    | OpenCV `TrackerKCF` on a drawn ROI                            |
| `ostrack`        | sot    | OSTrack — one-stream ViT SOT on ONNX (highest accuracy)       |

## Requirements

- **Python** (in PATH, or set the "Python cmd" field in the GUI).
- Install the dependencies:

  ```
  # Install Python (checked on 3.14.6):
  py --version
  # Python 3.14.6

  # Create Python virtual environment directory:
  py -m venv {VENV_DIR_NAME}
  
  # Set "Python cmd" in GUI to your real full path to python with virtual environment:
  {VENV_DIR_FULL_PATH}\Scripts\python.exe

  # Minimal requirements setup:
  {VENV_DIR_FULL_PATH}\Scripts\python.exe -m pip install opencv-contrib-python numpy simplification

  # Full requirements setup:
  {VENV_DIR_FULL_PATH}\Scripts\python.exe -m pip install -r requirements.txt
  ```

  See [`requirements.txt`](requirements.txt) — the single source of truth for the
  environment (OpenCV-contrib, NumPy, SciPy, and the tracker backends: ultralytics,
  mediapipe, `rf-groundingdino` + torch/torchvision, onnxruntime, etc.).
- The model weights (YOLO / MediaPipe / GroundingDINO / SAM2 / OSTrack ONNX) are
  **auto-downloaded** into `models/` on first run.

## Extension folder layout

```
Hismith Funscript Generator\
├── main.lua            (OFS extension: GUI, process management, result import)
├── funscript_gen.py    (thin CLI: argparse + _run() pipeline entry point)
├── requirements.txt    (single source of truth: pip install -r requirements.txt)
├── core/               (all generation logic)
│   ├── config.py       (TARGETS, MOTION_MODES, NORMALIZE_MODES, Config)
│   ├── types.py        (Person, ThrustCalibration)
│   ├── matching.py     (match_active, box_iou)
│   ├── logging.py      (log, log_file)
│   ├── utilities.py    (parse_time, format_time)
│   ├── detectors/      (yolo, mediapipe, groundingdino, sot-*, ostrack, box, models)
│   ├── tracking/       (scene_warp, streaming, seek, flow_refiner, holdover, scaling)
│   ├── postpass/       (strokes, positions, rdp_core, funscript, vlm)
│   └── interface/      (identify, calibrate, rdp_ui, preview, win32)
├── 3rdParty/           (vendored third-party code)
│   └── rxi/
│       └── json.lua    (JSON library - needed to parse the result)
├── models/             (auto-downloaded model weights)
└── ofs_result.json     (output consumed by main.lua)
```

## This project use next 3rdParty projects:

- [`rxi/json.lua`](https://github.com/rxi/json.lua) — a lightweight pure-Lua JSON
  library (v0.1.2, by Ryan Phillips). Vendored under
  [`3rdParty/rxi/json.lua`](3rdParty/rxi/json.lua) and required by
  [`main.lua`](main.lua) as `3rdParty.rxi.json`. Licensed under the
  [MIT License](https://github.com/rxi/json.lua/blob/master/LICENSE); the full
  copyright and permission notice is kept in the header of the vendored file.
