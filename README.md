# Hismith Funscript Generator

An OpenFunscripter (OFS) extension that generates FunScript keyframes from a video using AI pose tracking. The generated script drives a Hismith sex machine:

* `pos = 0`   → maximum forward extension (fully inserted)
* `pos = 100` → maximum backward extension (fully retracted)

The extension tracks the moving **point p** of the *active* partner (selected zone) frame-by-frame, segments the motion into forward/back strokes, and maps each frame to a 0..100 position. The result is imported into the current OFS timeline, where the points can be further edited by hand.

## How it works

1. **Scene setup** — configure scene rotation, offset, `1x`–`3x` zoom, and final crop region.
2. **ROI / active partner selection** *(skipped for `groundingdino` and `ollama`, which locate targets automatically via prompts)*:
   - **`sot-*` / `ostrack` / `gsam2`:** draw the active partner ROI and, optionally, a secondary ROI on a passive partner or static background area to reduce camera movement effects. You can also adjust the internal resolution of the selected scene via a second `1x`–`3x` zoom to improve tracking accuracy.
   - **`yolo` / `mediapipe`:** switch between auto-detected people using the number keys (`1`, `2`, ...)—optionally pressing `S` to preview the selected character's skeleton—then press `A` to assign the active partner and, optionally, `P` to assign the passive partner.
3. **Thrust calibration** — define the initial thrust direction (`b` = full retraction, `pos = 100`; `f` = full insertion, `pos = 0`) using either method:
   - **Manual line drag:** draw a line segment directly with the mouse (`LMB press` = `b`, `LMB release` = `f`). Exact point positions do not matter—only the vector direction (angle).
   - **Frame-by-frame keypoint picking:** scrub through frames using the `<` (Left Arrow) and `>` (Right Arrow) keys and mark the auto-detected tracking points at full retraction (`b`) and full insertion (`f`).
4. **Streaming collect** — frames are processed and previewed (`960x540`, OpenCV) to collect the raw per-frame trajectory of point `p`, applying optional scene/camera drift compensation (`scene_warp`), optical-flow refinement (`flow_refiner`), and identity hold-over (`holdover`).
5. **Post-pass** — the trajectory is segmented into strokes with adaptive axis tracking (handling changing thrust angles and auto-detecting `b`/`f` endpoints), and each frame's `pos` is computed along the curved path (`positions` + motion mode & normalization).
6. **RDP smoothing** — an interactive preview window lets you tune the epsilon of the Ramer–Douglas–Peucker simplification (`rdp_ui`).
7. **FunScript output** — keyframes (`time`, `pos`) are written to `ofs_result.json` and automatically imported into the active OFS timeline.

## Tracker backends

| Tracker | Kind | Description |
|----|----|----|
| `sot-csrt` | sot | OpenCV `TrackerCSRT` on a drawn ROI (max accuracy) |
| `sot-kcf` | sot | OpenCV `TrackerKCF` on a drawn ROI |
| `sot-vit` | sot | OpenCV `TrackerVit` (ONNX) on a drawn ROI |
| `sot-mil` | sot | OpenCV `TrackerMIL` (online multiple-instance learning) |
| `ostrack` | sot | OSTrack — one-stream ViT SOT on ONNX |
| `yolo` | pose | YOLOv8 / v11 / v26 pose (COCO-17 keypoints) |
| `mediapipe` | pose | MediaPipe pose (33 landmarks) |
| `groundingdino` | dino | GroundingDINO — open-vocabulary text-prompt detection |
| `gsam2` | mask | Grounded SAM 2, pixel-mask silhouette |
| `ollama` | LLM | LLM: draw a ROI, the model reads all sampled frames (slow, for hard clips) |

This project includes several experimental tracking methods, with **`sot-csrt`** currently proving to be the most effective and stable option for FunScript generation tasks.

While the workflow with `sot-csrt` is conceptually similar to [MTFG (Python-Funscript-Editor)](https://github.com/michael-mueller-git/Python-Funscript-Editor), it offers significantly higher stability and precision:

* **Superior grip on skin textures:** Unlike fast correlation trackers (such as KCF/MOSSE commonly used in MTFG) that rely on rigid rectangular bounding boxes and easily drift on low-contrast surfaces, CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) uses spatial reliability maps and multi-channel features (HOG + ColorNames). This allows it to weight foreground pixels higher than the background, preventing drift on smooth skin, partial occlusions, or non-rigid body deformations.
* **Outperforms neural/pose models on micro-movements:** Unlike skeleton-based pose estimators (YOLO, MediaPipe) or heavy segmentation models—which suffer from keypoint jitter and cannot lock onto arbitrary skin patches—`sot-csrt` tracks exact surface textures frame-by-frame with sub-pixel consistency and zero keypoint snapping.

### Key Capabilities

* **Dynamic direction tracking:** Automatically adapts to changes in the thrust angle. You only need to specify an approximate initial direction—the program will continue to accurately detect the `b` and `f` endpoints even as the angle of motion shifts throughout the scene.
* **Camera motion compensation:** In addition to tracking the active partner's ROI, you can select a secondary ROI on a static background area or the passive partner. This helps reduce the impact of camera movement, yielding significantly more accurate and realistic stroke results.
* **True path-length calculation:** Computes `pos` as the exact percentage of total real displacement along complex, curved trajectories between the `b` and `f` endpoints rather than simple linear projection.
* **Fast scene & resolution controls:** Quickly configure offset, rotation, scene crop region, and internal processing resolution.
* **Dual-stage zoom selection:** Features two independent zoom controls (`1x`–`3x`)—one in the **Scene ROI** selection window and another in the **Tracking ROI** selection window—allowing you to rapidly isolate the region of interest and upscale its internal resolution for maximum `sot-csrt` tracking accuracy.

## Requirements

* **Python** (in PATH, or set the "Python cmd" field in the GUI).

* Install the dependencies:

  ```
  # Install Python (checked on 3.14.6):
  py --version
  # Python 3.14.6

  # Create Python virtual environment directory:
  py -m venv {VENV_DIR_NAME}

  # Set "Python cmd" in GUI to your real full path to python with virtual environment:
  {VENV_DIR_FULL_PATH}\Scripts\python.exe

  # Minimal requirements setup:
  {VENV_DIR_FULL_PATH}\Scripts\python.exe -m pip install opencv-contrib-python numpy simplification av

  # Full requirements setup:
  {VENV_DIR_FULL_PATH}\Scripts\python.exe -m pip install -r requirements.txt
  ```

  See [`requirements.txt`](requirements.txt) — the single source of truth for the
  environment (OpenCV-contrib, NumPy, SciPy, and the tracker backends: ultralytics,
  mediapipe, `rf-groundingdino` + torch/torchvision, onnxruntime, etc.).

* The model weights (YOLO / MediaPipe / GroundingDINO / SAM2 / OSTrack ONNX) are
  **auto-downloaded** into `models/` on first run.

## Extension folder layout

```
Hismith Funscript Generator\
├── main.lua            (OFS extension: GUI, process management, result import)
├── tools.lua           (auxiliary OFS tools: scale / move / align / stretch / insert)
├── funscript_gen.py    (thin CLI: argparse + _run() pipeline entry point)
├── requirements.txt    (single source of truth: pip install -r requirements.txt)
├── core/               (all generation logic)
│   ├── config.py       (TARGETS, MOTION_MODES, NORMALIZE_MODES, Config)
│   ├── types.py        (Person, ThrustCalibration)
│   ├── matching.py     (match_active, box_iou)
│   ├── logging.py      (log, log_file)
│   ├── utilities.py    (parse_time, format_time)
│   ├── video_reader.py (BaseVideoReader / PyAV reader: precise PTS-based frame times)
│   ├── detectors/      (yolo, mediapipe, groundingdino, sot-*, ostrack, gsam2, ollama, box, models)
│   ├── tracking/       (scene_warp, streaming, seek, flow_refiner, holdover, scaling)
│   ├── postpass/       (strokes, positions, rdp_core, funscript, vlm, llm_gen, llm_stats)
│   └── interface/      (identify, calibrate, rdp_ui, preview, win32, llm_roi, llm_thinking)
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
