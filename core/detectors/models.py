"""
Model downloads and resolvers for the Hismith Funscript Generator.

Storage layout - every weight family has its own subdirectory under
<ext_root>/models/ (the single source of truth for these names is
core.config.MODEL_DIRS so the tree can be extended with new backends
without breaking CLI / GUI):

    models/
      yolo/            YOLO-pose .pt weights (auto-downloaded)
      mediapipe/       pose_landmarker_full.task (auto-downloaded)
      groundingdino/   GNDINO .pth weights (auto-downloaded)
      ostrack/         OSTrack ONNX (user-placed .onnx)
      vittrack/        sot-vit TrackerVit ONNX (auto-downloaded or user .onnx)

There is NO legacy flat-root fallback: files MUST live in their own
subdirectory.  sot-mil has no weight file (cv2 5.x MIL trains its own
classifier online) - there is no entry for it in MODEL_DIRS on purpose.

This module exposes:
  * per-family downloaders:        yolo_model_path, mediapipe_model_path,
                                   gndino_weights_path, ensure_sot_onnx
  * per-family listings:           list_ostrack_models, list_vittrack_models
  * per-family resolvers:          ostrack_model_path, vittrack_model_path
  * generic resolver:              get_onnx_model_path(model, hint="onnx")
"""
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

from core import models_directory, model_subdir
from core.config import MODEL_DIRS

YOLO_POSE_URLS = {
    "yolov8n-pose": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
    "yolov8s-pose": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt",
    "yolov8m-pose": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt",
    "yolov8l-pose": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-pose.pt",
    "yolov8x-pose": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose.pt",
    "yolo11n-pose": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    "yolo11s-pose": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt",
    "yolo11m-pose": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
    "yolo11l-pose": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt",
    "yolo11x-pose": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
    "yolo26n-pose": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
    "yolo26s-pose": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt",
    "yolo26m-pose": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt",
    "yolo26l-pose": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt",
    "yolo26x-pose": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt",
}


def _download(url: str, model_path: Path) -> str:
    print(f"Downloading model: {url}")
    tmp = model_path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(model_path)
    print(f"Model saved: {model_path}")
    return str(model_path)


def yolo_model_path(name: str) -> str:
    """Resolve (or auto-download) a YOLO-pose .pt into models/yolo/."""
    model_path: Path = model_subdir(MODEL_DIRS["yolo"]) / f"{name}.pt"
    if model_path.exists():
        return str(model_path)
    url = YOLO_POSE_URLS.get(name)
    if not url:
        return name
    return _download(url, model_path)


def mediapipe_model_path() -> str:
    """Resolve (or auto-download) the MediaPipe pose landmarker into
    models/mediapipe/.  'latest' is a stable, permanent Google URL."""
    model_path: Path = model_subdir(MODEL_DIRS["mediapipe"]) / "pose_landmarker_full.task"
    if model_path.exists():
        return str(model_path)
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    return _download(url, model_path)


# ── GroundingDINO variants ─────────────────────────────────────────────
# Two official checkpoints (IDEA-Research/GroundingDINO, v0.1.0-alpha /
# v0.1.0-alpha2): the Swin-T base (48.4 zero-shot AP, O365/Obj365-style
# data) and the flagship Swin-B (56.7 AP, trained additionally on RefCOCO /
# OpenImages / ODinW-35).  The matching config files ship INSIDE the
# `groundingdino` package (site-packages/groundingdino/config/), so we only
# download the weights into models/.
GNDINO_VARIANTS = {
    "swinb": {
        "weights": "groundingdino_swinb_cogcoor.pth",
        "config": "GroundingDINO_SwinB_cfg.py",
        "urls": [
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
            "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
            "groundingdino_swinb_cogcoor.pth",
        ],
    },
    "swint": {
        "weights": "groundingdino_swint_ogc.pth",
        "config": "GroundingDINO_SwinT_OGC.py",
        "urls": [
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
            "groundingdino_swint_ogc.pth",
        ],
    },
}
DEFAULT_GNDINO_VARIANT = "swinb"


def _find_package_config(name: str) -> Path:
    """Locate a GroundingDINO config file shipped inside the installed
    `groundingdino` package (site-packages/groundingdino/config/...), or
    the user-placed copy in models/groundingdino/."""
    d = model_subdir(MODEL_DIRS["groundingdino"])
    local = d / name
    if local.exists():
        return local
    import groundingdino
    pkg_dir = Path(groundingdino.__file__).resolve().parent
    cand = pkg_dir / "config" / name
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"GroundingDINO config {name!r} not found. Place it into the "
        f"extension models/ directory or the installed package's config/ "
        f"folder (it ships with the `groundingdino` PyPI package).")


def gndino_config_path(variant: str = DEFAULT_GNDINO_VARIANT) -> str:
    """Architecture config for a GroundingDINO variant (swinb / swint)."""
    v = GNDINO_VARIANTS[variant]
    return str(_find_package_config(v["config"]))


def gndino_weights_path(variant: str = DEFAULT_GNDINO_VARIANT) -> str:
    """Resolve (or auto-download) the official weights for a variant.

    Resolution order: $GROUNDINGDINO_MODEL -> models/<name>.pth ->
    auto-download from the first URL that responds (GitHub release or
    the HuggingFace mirror, ~895 MB for Swin-B)."""
    v = GNDINO_VARIANTS[variant]
    target = model_subdir(MODEL_DIRS["groundingdino"]) / v["weights"]
    if "GROUNDINGDINO_MODEL" in os.environ:
        env_p = Path(os.environ["GROUNDINGDINO_MODEL"])
        if env_p.exists():
            return str(env_p)
    if target.exists():
        return str(target)
    last_err = None
    for url in v["urls"]:
        try:
            return _download(url, target)
        except Exception as e:  # try the mirror next
            last_err = e
    raise FileNotFoundError(
        f"GroundingDINO ('{variant}') weights not found and auto-download "
        f"failed ({last_err}). Place {v['weights']!r} into {target.parent} "
        f"(or set GROUNDINGDINO_MODEL to its full path). "
        f"URLs: {v['urls']}")


# ── SAM 2 video-predictor checkpoints (gsam2 tracker) ──────────────────
# Official repo facebookresearch/sam2 (Apache-2.0).  Variant name ->
# (weights filename, config yaml path, [mirror URLs]).
#   * names mirror sam2.build_sam's HF_MODEL_ID_TO_FILENAMES table.
#   * URL = Meta's CDN (dl.fbaipublicfiles.com/segment_anything_2/...);
#     the HuggingFace repo (facebook/sam2.1-hiera-*) is the documented
#     fallback when the primary 404s.
#   * NOTE: SAM 2.1 checkpoints REQUIRE sam2 >= 1.1.0 (see
#     facebookresearch/sam2 RELEASE_NOTES.md).
#   * The config files (configs/sam2*/*.yaml) ship INSIDE the installed
#     `sam2` PyPI package - resolved via `sam2.__file__` (same pattern as
#     `_find_package_config` for GNDINO); only the weights auto-download.
#   * Default = sam2.1-hiera-base-plus: best accuracy/size balance;
#     sam2.1-hiera-large = max accuracy (fits 32 GB VRAM on RTX 5090).
SAM2_VARIANTS = {
    "sam2-hiera-tiny": {
        "weights": "sam2_hiera_tiny.pt",
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_tiny.pt",
            "https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/"
            "sam2_hiera_tiny.pt",
        ],
    },
    "sam2-hiera-small": {
        "weights": "sam2_hiera_small.pt",
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_small.pt",
            "https://huggingface.co/facebook/sam2-hiera-small/resolve/main/"
            "sam2_hiera_small.pt",
        ],
    },
    "sam2-hiera-base-plus": {
        "weights": "sam2_hiera_base_plus.pt",
        "config": "configs/sam2/sam2_hiera_b+.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_base_plus.pt",
            "https://huggingface.co/facebook/sam2-hiera-base-plus/resolve/"
            "main/sam2_hiera_base_plus.pt",
        ],
    },
    "sam2-hiera-large": {
        "weights": "sam2_hiera_large.pt",
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_large.pt",
            "https://huggingface.co/facebook/sam2-hiera-large/resolve/main/"
            "sam2_hiera_large.pt",
        ],
    },
    "sam2.1-hiera-tiny": {
        "weights": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/093024/"
            "sam2.1_hiera_tiny.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/"
            "sam2.1_hiera_tiny.pt",
        ],
    },
    "sam2.1-hiera-small": {
        "weights": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/093024/"
            "sam2.1_hiera_small.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/"
            "sam2.1_hiera_small.pt",
        ],
    },
    "sam2.1-hiera-base-plus": {
        "weights": "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/093024/"
            "sam2.1_hiera_base_plus.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/"
            "main/sam2.1_hiera_base_plus.pt",
        ],
    },
    "sam2.1-hiera-large": {
        "weights": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/093024/"
            "sam2.1_hiera_large.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/"
            "sam2.1_hiera_large.pt",
        ],
    },
}

#: default checkpoint (best balance); sam2.1-hiera-large = max accuracy on
#: RTX 5090 (see the gsam2 tracker docstring + requirements.txt).
DEFAULT_SAM2_VARIANT = "sam2.1-hiera-base-plus"


def _list_pt_in(key: str) -> List[str]:
    """All *.pt / *.pth files in models/<key>/ (bare names, sorted)."""
    d = model_subdir(MODEL_DIRS[key])
    if not d.exists():
        return []
    return sorted((p.name for p in d.iterdir()
                   if p.is_file() and p.suffix in (".pt", ".pth")),
                  key=str.lower)


def list_sam2_models() -> List[str]:
    """All *.pt / *.pth in models/sam2/ (bare filenames, sorted) —
    the gsam2 checkpoint picker in the OFS GUI / CLI."""
    return _list_pt_in("sam2")


def _find_sam2_config(config_rel: str) -> Path:
    """Locate a SAM 2 hydra config (e.g. configs/sam2.1/sam2.1_hiera_b+.yaml)
    shipped INSIDE the installed `sam2` package, or a user-placed copy in
    models/sam2/."""
    name = Path(config_rel).name
    local = model_subdir(MODEL_DIRS["sam2"]) / name
    if local.exists():
        return local
    import sam2
    pkg = Path(sam2.__file__).resolve().parent
    cand = pkg / config_rel.replace("\\", "/")
    if cand.exists():
        return cand
    for c in pkg.glob(f"configs/**/{name}"):
        return c
    raise FileNotFoundError(
        f"SAM 2 config {config_rel!r} not found. The `sam2` PyPI package "
        f"ships its configs under site-packages/sam2/configs/... - check "
        f"`pip show sam2` (need sam2 >= 1.1.0 for SAM 2.1).")


def sam2_config_path(variant: str = DEFAULT_SAM2_VARIANT) -> str:
    """Architecture config path for a SAM 2 variant (hydra compose name)."""
    v = SAM2_VARIANTS[variant]
    return str(_find_sam2_config(v["config"]))


def sam2_weights_path(variant: str = DEFAULT_SAM2_VARIANT) -> str:
    """Resolve (or auto-download) the SAM 2 checkpoint for a variant into
    models/sam2/.

    Resolution order: $SAM2_MODEL -> models/sam2/<filename>
    -> auto-download from the first URL that responds (Meta CDN, then the
    HuggingFace mirror)."""
    v = SAM2_VARIANTS[variant]
    target = model_subdir(MODEL_DIRS["sam2"]) / v["weights"]
    if "SAM2_MODEL" in os.environ:
        env_p = Path(os.environ["SAM2_MODEL"])
        if env_p.exists():
            return str(env_p)
    if target.exists():
        return str(target)
    last_err = None
    for url in v["urls"]:
        try:
            print(f"[sam2] auto-downloading '{variant}' from {url}")
            return _download(url, target)
        except Exception as e:  # try the mirror next
            last_err = e
    raise FileNotFoundError(
        f"SAM 2 ('{variant}') weights not found and auto-download failed "
        f"({last_err}). Place {v['weights']!r} into {target.parent} "
        f"(or set SAM2_MODEL to its full path). URLs: {v['urls']}")


# ── OpenCV SOT ONNX models (ViT only - MIL has no weights file) ─────────
# TrackerCSRT and TrackerKCF are self-contained (verified: construct +
# update() with no external file).  TrackerVit needs an ONNX weights file
# which cv2-python does NOT bundle; OpenCV looks for it in the CURRENT
# WORKING DIRECTORY by bare file name at construction time.  TrackerMIL
# trains its own classifier online (multiple instance learning) - no
# external weight file exists for it in cv2 5.x, so there is no "mil"
# entry here and no models/vittrack file for it.
#
# Storage: the official weights live in models/vittrack/
# (core.config.MODEL_DIRS["vittrack"]).
#
# Strategy in ensure_sot_onnx(algo):
#   1. already in CWD            -> use it (nothing to do);
#   2. already in models/vittrack/ -> copy it into the CWD;
#   3. download into models/vittrack/ -> copy it into the CWD;
#   4. none worked               -> clear error (csrt/kcf/mil always work).
_SOT_ONNX = {
    "vit": "vitTracker.onnx",
}
# candidate download URLs (best-known OpenCV source paths), tried in order
_SOT_ONNX_URLS = {
    "vit": [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/modules/dnn/vitTracker.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/dnn/../../vitTracker.onnx",
        "https://github.com/opencv/opencv_contrib/raw/4.x/modules/dnn/vitTracker.onnx",
    ],
}


def ensure_sot_onnx(algo: str) -> None:
    """Make the ONNX weights for `algo` available in the CWD (vit only).
    csrt / kcf / mil need no external file (mil trains online) and return
    immediately."""
    import shutil
    fname = _SOT_ONNX.get(algo)
    if not fname:
        return  # csrt / kcf / mil: no external weights file
    cwd = Path.cwd()
    if (cwd / fname).exists():
        return
    cached_dir = model_subdir(MODEL_DIRS["vittrack"])
    cached = cached_dir / fname
    if cached.exists():
        shutil.copy2(str(cached), str(cwd / fname))
        return
    for url in _SOT_ONNX_URLS.get(algo, []):
        try:
            print(f"[sot] Downloading {fname} from {url} ...")
            tmp = cached_dir / f".{fname}.tmp"
            urllib.request.urlretrieve(url, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 50 * 1024:
                tmp.rename(cached)
                shutil.copy2(str(cached), str(cwd / fname))
                print(f"[sot] {fname} ready in {cwd}")
                return
            # downloaded too small to be a real model - keep trying the rest
            tmp.unlink(missing_ok=True)
        except Exception:
            continue
    raise FileNotFoundError(
        f"SOT {algo!r} needs '{fname}' but it could not be fetched "
        f"automatically. Either place the file into {cached_dir} "
        f"(or the working directory), or use a self-contained algorithm: "
        f"--tracker sot-csrt or sot-kcf (sot-mil trains online).")


# ── OSTrack ONNX models ────────────────────────────────────────────────────
# No official ONNX release - the botaoye/OSTrack project ships .pth.tar
# (Google Drive) + community ONNX exports exist (e.g. l-sf/Track-trt).
# The user drops one or more *.onnx files into the extension models/
# directory; the OFS GUI lists them (funscript_gen.py --list-onnx prints
# them one per line) and passes --ostrack-model <name> to the CLI.
# A bare filename is resolved against models/ first, then the CWD.


def _list_onnx_in(subdir_key: str) -> List[str]:
    """All non-zero-byte *.onnx files in models/<subdir_key>/ (sorted,
    bare filenames).  `subdir_key` is a key of core.config.MODEL_DIRS
    (e.g. "ostrack" or "vittrack").  Skips zero-byte / temp files - the
    directory is auto-created via model_subdir so listing is safe even
    on a fresh install."""
    d = model_subdir(MODEL_DIRS[subdir_key])
    out = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".onnx":
            continue
        if p.stat().st_size == 0:
            continue
        out.append(p.name)
    out.sort(key=str.lower)
    return out


def list_ostrack_models() -> List[str]:
    """All *.onnx files in models/ostrack/ (bare filenames, sorted).

    The OFS GUI (main.lua) and the CLI `--list ostrack` consume this to
    fill the OSTrack model combo.  sot-mil is deliberately not listed
    anywhere - cv2 5.x MIL trains its own classifier online and has no
    weight file."""
    return _list_onnx_in("ostrack")


def list_vittrack_models() -> List[str]:
    """All *.onnx files in models/vittrack/ (bare filenames, sorted) —
    the sot-vit TrackerVit weight picker in the OFS GUI / CLI."""
    return _list_onnx_in("vittrack")


def _resolve_onnx_in(subdir_key: str, model: Optional[str], label: str,
                     how_to_obtain: str) -> str:
    """Shared resolver: a bare name is looked up in
    models/<subdir_key>/<name> first, then CWD; a full/absolute path is
    returned as-is.  model=None resolves to the first (alphabetical)
    *.onnx in models/<subdir_key>/.  A clear, family-specific
    FileNotFoundError is raised when nothing matches - the message names
    the subfolder and how to obtain the weights."""
    d = model_subdir(MODEL_DIRS[subdir_key])
    cands = []
    if model:
        p = Path(model)
        if p.is_absolute() and p.exists():
            return str(p)
        # try this family's models/ subfolder first, then CWD
        cands = [d / model, Path.cwd() / model]
    else:
        cands = [d / n for n in _list_onnx_in(subdir_key)]
    for c in cands:
        if c.exists():
            return str(c)

    found = _list_onnx_in(subdir_key)
    if found:
        raise FileNotFoundError(
            f"{label} {model!r} not found. Available in {d}:\n  "
            + "\n  ".join(found)
            + f"\nPick one via the GUI or pass the bare filename.\n"
              f"({how_to_obtain})")
    raise FileNotFoundError(
        f"No {label} file in {d}. {how_to_obtain}")


def ostrack_model_path(model: Optional[str] = None) -> str:
    """Resolve an OSTrack ONNX file from models/ostrack/.

        model = full path  -> return as-is
                            -> models/ostrack/<model>
                            -> CWD/<model>
        model = None       -> first *.onnx in models/ostrack/
        nothing found      -> clear FileNotFoundError explaining the layout.
    """
    return _resolve_onnx_in(
        "ostrack", model, "OSTrack ONNX",
        "OSTrack has no official ONNX release. Get one from the official "
        "botaoye/OSTrack Google Drive "
        "(https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy) "
        "+ the l-sf/Track-trt ONNX export recipe "
        "(https://github.com/l-sf/Track-trt). The file must expose inputs "
        "'z' (template) and 'x' (search) plus outputs 'score_map', "
        "'size_map', 'offset_map'.")


def vittrack_model_path(model: Optional[str] = None) -> str:
    """Resolve a sot-vit TrackerVit ONNX file from models/vittrack/.

        model = full path  -> return as-is
                            -> models/vittrack/<model>
                            -> CWD/<model>
        model = None       -> first *.onnx in models/vittrack/
        nothing found      -> clear FileNotFoundError explaining the layout.
    """
    return _resolve_onnx_in(
        "vittrack", model, "sot-vit ONNX",
        "The official vitTracker.onnx can be auto-downloaded via "
        "`--tracker sot-vit` (ensure_sot_onnx) or obtained from "
        "opencv_contrib dnn samples. Any TrackerVit-compatible .onnx "
        "also works.")


def list_onnx_models() -> List[str]:
    """Union of models/ostrack/*.onnx and models/vittrack/*.onnx (sorted,
    bare filenames) - kept only for the CLI `--list` diagnostic.  The OFS
    GUI uses the per-family functions instead."""
    return sorted(list_ostrack_models() + list_vittrack_models(),
                  key=str.lower)


def get_onnx_model_path(model: Optional[str], hint: str = "onnx") -> str:
    """Generic ONNX resolver across BOTH families (ostrack, then vittrack).

    Used by code that does not know which tracker will consume the file.
    Prefer the per-family resolvers (ostrack_model_path / vittrack_model_path)
    in tracker code so error messages point at the correct folder.

        `model` = full path   -> return as-is
        `model` = bare name   -> models/ostrack/<name> -> models/vittrack/<name>
                                 -> CWD/<name>
        `model` = None        -> first *.onnx in models/ostrack/, then
                                 models/vittrack/ (alphabetical)

    `hint` - a human-facing hint used in the error message (e.g. "sot-vit"
    or "OSTrack") so the traceback tells the user WHICH consumer failed."""
    if model:
        p = Path(model)
        if p.is_absolute() and p.exists():
            return str(p)
        for key in ("ostrack", "vittrack"):
            c = model_subdir(MODEL_DIRS[key]) / model
            if c.exists():
                return str(c)
        cwd_cand = Path.cwd() / model
        if cwd_cand.exists():
            return str(cwd_cand)
    else:
        for key in ("ostrack", "vittrack"):
            lst = _list_onnx_in(key)
            if lst:
                return str(model_subdir(MODEL_DIRS[key]) / lst[0])
    found = list_ostrack_models() + list_vittrack_models()
    ostrack_dir = model_subdir(MODEL_DIRS["ostrack"])
    vit_dir = model_subdir(MODEL_DIRS["vittrack"])
    if found:
        raise FileNotFoundError(
            f"{hint} {model!r} not found. Available in the ONNX folders:\n  "
            + "".join(f"{f}\n  " for f in found)
            + f"OSTrack: {ostrack_dir}\nsot-vit:      {vit_dir}\n"
              "Pick one via the GUI combo or pass the bare filename.")
    raise FileNotFoundError(
        f"{hint} not found. Put the .onnx file into {ostrack_dir} (OSTrack) "
        f"or {vit_dir} (sot-vit). Once dropped, pick it in the GUI combo or "
        "pass the bare filename on the CLI.")
