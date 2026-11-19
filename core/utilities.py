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


