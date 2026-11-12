"""Small shared helpers (no project dependencies)."""
import sys


def parse_time(s: str) -> float:
    """'5:27' → 327.0, '1:02:03' → 3723.0, '327.5' → 327.5"""
    s = s.strip()
    parts = s.split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            return float(s)
    except ValueError:
        sys.exit(f"Invalid time: '{s}'. Format: MM:SS or HH:MM:SS")


def format_time(sec: float) -> str:
    """327.0 → '5:27'"""
    sec = max(0.0, sec)
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def read_existing_points(file_path, start_s: float):
    """Read the existing-points dump written by main.lua and derive anchors.

    File format (one line per existing action, whitespace separated,
    absolute video seconds):
        '<time_seconds> <pos>'

    Returns a 3-tuple:
        times          : sorted list of (absolute) existing-action times,
                         used by the streaming loop to stop at the first one
                         STRICTLY after start_s (requirement #2);
        anchor_begin   : pos of the LAST action with time <  start_s
                         (the value the new segment must LEAVE from), or None
                         if no action exists before start_s;
        anchor_end     : pos of the FIRST action with time >= start_s
                         (the value the new segment must LAND on), or None
                         if no action exists at/after start_s.

    Plain-text on purpose - no JSON dependency needed on the Python side.
    Missing / unreadable / empty file -> ([], None, None) and a clear no-op
    (the caller falls back to legacy behaviour).
    """
    import os
    if not file_path or not os.path.isfile(file_path):
        return [], None, None
    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    entries.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    except OSError:
        return [], None, None
    if not entries:
        return [], None, None
    entries.sort(key=lambda e: e[0])
    anchor_begin = None
    anchor_end = None
    # Semantic: a keyframe AT start_s is the LEFT anchor (i.e., the START of
    # the new segment), NOT the right/stop boundary.  So the split is:
    #   anchor_begin = pos of LAST entry with t <= start_s;
    #   anchor_end   = pos of FIRST entry with t >  start_s.
    # If no entry has t > start_s, anchor_end stays None (the new segment
    # extends to END OF VIDEO).  This is the "user placed the playhead ON
    # the last keyframe" case: generation goes FORWARD from it, not STOPs
    # immediately after.  (The old "t < start_s / t >= start_s" split would
    # treat the at-start keyframe as the stop, collapsing the track to a
    # single frame and discarding the rest of the video.)
    for t, p in entries:
        if t <= start_s:
            anchor_begin = p
        else:
            anchor_end = p
            break
    return [e[0] for e in entries], anchor_begin, anchor_end
