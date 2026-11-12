"""
core - Hismith Funscript Generator package.

EXT_ROOT is the single canonical path to the extension root (the folder
containing main.lua, funscript_gen.py, the stderr log file and the
downloaded models).  Every module that needs one of those paths must go
through EXT_ROOT - the core code lives one level BELOW the extension
root, so Path(__file__) inside a core module points to core/ itself.
"""
from pathlib import Path

# parent of this package = the extension root
EXT_ROOT = Path(__file__).resolve().parent.parent


def models_directory() -> Path:
    """Create and return the models/ root directory (auto-download cache)."""
    d = EXT_ROOT / "models"
    d.mkdir(exist_ok=True)
    return d


def model_subdir(name: str) -> Path:
    """Create and return a models/<family> subdirectory.

    `name` is a key of core.config.MODEL_DIRS - every weight family (yolo,
    mediapipe, groundingdino, ostrack, vittrack, ...) lives in its own
    folder so the models/ tree stays organized as new backends are added.
    """
    d = models_directory() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_file_path() -> Path:
    """The canonical stderr log file (the OFS GUI tail / opens it)."""
    return EXT_ROOT / "funscript_gen.stderr.log"
