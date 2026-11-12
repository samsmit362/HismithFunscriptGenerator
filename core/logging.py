"""
Logging: every line goes to the log file first (the reliable channel),
then best-effort to stdout and stderr (OFS's pipes).

OFS launches funscript_gen.py directly (Process.new, no shell redirect),
so stdout/stderr are OFS's pipes and the parent ends of them are closed
(see OFS source OFS_LuaProcessAPI.cpp).  This file handle is the
reliable log channel: every log message AND the final traceback go
here, in addition to stdout/stderr - so a crash is debuggable even
when both pipes are broken.

The log file is `funscript_gen.stderr.log` NEXT TO main.lua (the OFS
GUI tails it and "Open stderr log" opens it) - see core.log_file_path,
NOT relative to this file (core/ is one level deeper than the root).
"""
import os
import sys

from core import log_file_path

_file_log = None
# prefer an explicit path (OFS can pass one), otherwise use the SAME file
# as the GUI shows / "Open stderr log" opens - with a direct launch
# (no cmd redirect) this IS the primary log
if "PYTHON_LOG_FILE" in os.environ:
    try:
        _file_log = open(os.environ["PYTHON_LOG_FILE"], "a", buffering=1)
    except Exception:
        _file_log = None
if _file_log is None:
    try:
        _file_log = open(log_file_path(), "w", buffering=1)
    except Exception:
        _file_log = None


def log_file(msg: str) -> None:
    """Write a line to the fallback log file (no-op if unavailable)."""
    if _file_log is None:
        return
    try:
        _file_log.write(msg + "\n")
        _file_log.flush()
    except Exception:
        pass


def log(msg: str) -> None:
    """Log to stdout, stderr AND the log file (whichever is usable).

    The log file is the reliable channel (see module-level _file_log);
    stdout/stderr go to OFS's pipes - best effort.
    """
    log_file(msg)
    for _target in (None, sys.stderr):
        try:
            print(msg, file=_target, flush=True)
        except Exception:
            pass
