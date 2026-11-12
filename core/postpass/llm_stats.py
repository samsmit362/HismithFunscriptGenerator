"""Persistent stats for LLM (Ollama) generation calls - ETA estimation.

Each successful ``generate_track`` call is recorded: (images, elapsed_s).
On subsequent calls the thinking window displays a running-average ETA:

    eta_s = (avg_seconds_per_image) * (target_images) - already_elapsed

The stats persist to ``llm_session_stats.json`` next to main.lua so that
multiple generate calls within one ``funscript_gen.py`` process accumulate
(relevant when the user generates several segments of the same video),
and survive across separate OFS sessions (the next session starts from
the previous session's data, giving an even more stable estimate).

Single-process write (no concurrent access): ``funscript_gen.py`` only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

STATS_FILENAME = "llm_session_stats.json"
MAX_ENTRIES = 500


def stats_path() -> Path:
    """Extension root / STATS_FILENAME (absolute)."""
    from core import EXT_ROOT
    return Path(EXT_ROOT) / STATS_FILENAME


@dataclass
class LLMCallRecord:
    t: str            # ISO-8601 local time of call completion
    model: str
    images: int
    ctx: int
    thumb: int
    elapsed_s: float
    success: bool


def _rec_dict(r: LLMCallRecord) -> Dict[str, Any]:
    return {
        "t": r.t, "model": r.model,
        "images": int(r.images), "ctx": int(r.ctx),
        "thumb": int(r.thumb),
        "elapsed_s": round(float(r.elapsed_s), 3),
        "success": bool(r.success),
    }


def load_all() -> List[LLMCallRecord]:
    p = stats_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    arr = raw.get("calls", [])
    if not isinstance(arr, list):
        return []
    out: List[LLMCallRecord] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        try:
            out.append(LLMCallRecord(
                t=str(it.get("t", "")),
                model=str(it.get("model", "")),
                images=int(it.get("images", 0) or 0),
                ctx=int(it.get("ctx", 0) or 0),
                thumb=int(it.get("thumb", 0) or 0),
                elapsed_s=float(it.get("elapsed_s", 0.0) or 0.0),
                success=bool(it.get("success", False)),
            ))
        except Exception:
            continue
    return out


def record_call(model: str, images: int, ctx: int, thumb: int,
                elapsed_s: float, success: bool) -> None:
    """Append one call record. Never raises."""
    rec = LLMCallRecord(
        t=time.strftime("%Y-%m-%d %H:%M:%S"),
        model=str(model or ""),
        images=int(images or 0),
        ctx=int(ctx or 0),
        thumb=int(thumb or 0),
        elapsed_s=float(elapsed_s or 0.0),
        success=bool(success),
    )
    try:
        all_recs = load_all()
        all_recs.append(rec)
        if len(all_recs) > MAX_ENTRIES:
            all_recs = all_recs[-MAX_ENTRIES:]
        payload = {"calls": [_rec_dict(r) for r in all_recs]}
        tmp = stats_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(stats_path())
    except Exception as e:
        from core.logging import log
        log(f"[llm_stats] WARNING: could not persist stats: {e!r}")


def summary() -> Dict[str, Any]:
    """Aggregate SUCCESSFUL calls.

    Returns:
        n             number of successful calls
        avg_call_s    mean wall-time per successful call
        avg_image_s   mean wall-time per single image (overall)
        last_model    model name of most recent successful call
        last_s        wall-time of most recent successful call
    """
    recs = [r for r in load_all()
            if r.success and r.elapsed_s > 0.0 and r.images > 0]
    if not recs:
        return {"n": 0, "avg_call_s": 0.0, "avg_image_s": 0.0,
                "last_model": "", "last_s": 0.0}
    total_img = sum(r.images for r in recs)
    total_s = sum(r.elapsed_s for r in recs)
    n = len(recs)
    return {
        "n": n,
        "avg_call_s": total_s / n,
        "avg_image_s": total_s / max(1, total_img),
        "last_model": recs[-1].model,
        "last_s": recs[-1].elapsed_s,
    }


def estimate_eta(target_images: int,
                 already_elapsed_s: float = 0.0) -> Optional[float]:
    """Estimate remaining seconds for a call with ``target_images`` images.

    Uses the running average from ``summary()``.  Returns None when no
    prior data exists.
    """
    s = summary()
    if s["n"] <= 0 or s["avg_image_s"] <= 0.0:
        return None
    total_estimate = s["avg_image_s"] * float(target_images)
    eta = total_estimate - max(0.0, already_elapsed_s)
    return max(0.0, eta)


def format_eta(eta_s: Optional[float]) -> str:
    if eta_s is None:
        return "n/a (no prior runs)"
    m, sc = divmod(int(eta_s), 60)
    return f"~{m}m{sc:02d}s remaining" if m else f"~{sc}s remaining"


__all__ = ["LLMCallRecord", "stats_path", "load_all", "record_call",
           "summary", "estimate_eta", "format_eta"]
