"""FunScript JSON assembly (consumed by main.lua importResult)."""
from typing import List, Tuple

from core.config import Config
from core.utilities import format_time
from core.logging import log

def build_funscript(points: List[Tuple[float, float]], fps: float,
                    cfg: Config, title: str) -> dict:
    """Assemble the FunScript JSON.

    points: iterable of ``(t_sec, pos)`` - the ABSOLUTE video time in seconds
        (already carrying any ``start_s`` offset) plus the 0..100 position.
        RDP / the caller may keep either the whole track or a sub-range.
    fps:    video fps (kept for API symmetry; the times are already absolute).
    cfg:    Config (latency_ms, inverted, output_range come from here).
    title:  script title.

    NOTE: times are NOT re-derived from a frame index here - the caller owns
    any reduction (RDP / segment selection). ``build_funscript`` only converts
    absolute seconds -> milliseconds and adds the user-configured latency.
    """
    pts = [(float(t), float(p)) for t, p in points]
    pts = sorted(pts, key=lambda x: x[0])
    if len(pts) < 1:
        raise RuntimeError(f"[funscript] need >=1 point, got {len(pts)}")
    if len(pts) == 1:
        # A single action is a valid minimum for OFS: the boundary
        # anchor(s) are already on the timeline (see main.lua's
        # hf_existing_points.txt + _apply_boundary_anchors), so the new
        # segment is "the next keyframe at the known anchor value".
        log("[build_funscript] NOTE: single-point result (1 action) - "
            f"t={pts[0][0]:.3f}s pos={pts[0][1]:.1f}; the stroke segment "
            f"was shorter than one tracked frame (or stopped immediately "
            f"at the next timeline keyframe), 1 keyframe is the expected "
            f"minimum - OFS cross-dedupes with the existing action at "
            f"the same ms.")

    lat_ms = int(cfg.latency_ms or 0)
    log(f"[build_funscript] lat_ms:{lat_ms}")
    t0_ms = int(round(pts[0][0] * 1000.0))
    t1_ms = int(round(pts[-1][0] * 1000.0))
    actions = [
        {"at": int(round(t * 1000.0)) + lat_ms,
         "pos": int(round(p))}
        for t, p in pts
    ]
    # OFS invariant: one action per millisecond-timestamp (the import keeps the
    # FIRST point of a given timestamp).  Drop any later point whose rounded
    # timestamp repeats -- the batch is already time-sorted -- otherwise the
    # import's Commit() aborts on the duplicate and drops the WHOLE batch.
    _unique, _seen = [], set()
    for _a in actions:
        if _a["at"] not in _seen:
            _seen.add(_a["at"])
            _unique.append(_a)
    if len(_unique) != len(actions):
        log(f"[build_funscript] dropped {len(actions) - len(_unique)} "
            f"duplicate-timestamp action(s)")
    actions = _unique
    duration_ms = max(0, t1_ms - t0_ms)

    return {
        "actions": actions,
        "inverted": cfg.inverted,
        "metadata": {
            "bookmarks": [], "chapters": [], "creator": "",
            "description": (f"Hismith funscript, {format_time(pts[0][0])}"
                            f"-{format_time(pts[-1][0])}"),
            "duration": duration_ms // 1000, "license": "",
            "notes": "0%=forward(end of fwd stroke), 100%=backward(end of back stroke)",
            "performers": [], "script_url": "",
            "tags": ["hismith", "funscript"],
            "title": title, "type": "basic", "video_url": ""
        },
        "range": cfg.output_range,
        "version": "1.0"
    }
