"""
Identity hold-over (HoldoverPoint) + stream telemetry container.
"""
from typing import List, Optional, Tuple

import numpy as np

from core.tracking.flow_refiner import FlowRefiner
from core.tracking.scaling import odd_clamp, person_scale_estimate
from core.types import Person


class StreamTracks(list):
    """tracks list + per-session telemetry attributes (hold frames, rebinds)
    so harnesses can read them without changing the return signature.

    New fields (2026-09, real-time + filtering invariants):
      * times: List[float] - REAL presentation time (seconds, from
        CAP_PROP_POS_MSEC) aligned 1:1 with the tracks slots.  NEVER
        computed from a frame index, per project requirement #2.
      * dup_skipped: int - duplicate frames that were skipped (identical
        timestamps or bit-identical pixels).
      * ghost_count: int - ghost (hold-over) frames that are stored as None
        slots in this list; the post-pass drops these slots so they never
        reach RDP / OpenFunScripter.
      * ps: List[Optional[Tuple[float, float]]] - optional per-frame RAW
        position of the second "static" ROI (ps) the user draws with Ctrl+LMB
        in the SOT identify window.  1:1 with this list's slots; None when no
        ps-ROI was selected or the second tracker lost the object.  The
        COMPENSATED point (p - (ps - ps_baseline)) is what tracking_stream
        appends to `self`; the raw ps sequence is kept here as telemetry for
        the --motion complex step invariance: step = |(p-p_prev)-(ps-ps_prev)|.
    """
    __slots__ = ("hold_frames", "hold_max_run", "rebinds",
                 "times", "dup_skipped", "ghost_count", "ps")


class HoldoverPoint:
    """
    Identity hold-over: the ACTIVE partner's point p is a per-frame anchor.
    In some frames the active person is not in `detect_all()` at all, or the
    only nearby person in the frame is the passive partner, and the matcher
    (distance-dominant) will "win" on the passive just because it's the only
    candidate - the tracked point teleports to the passive.

    This helper keeps a LK flow prediction from the last accepted point
    (the "ghost") and, whenever the incoming anchor is FAR from the ghost
    (>= accept_tol) OR no anchor at all is available, returns the ghost
    instead of the anchor for that frame.  When the active re-appears close
    to the ghost, the anchor is accepted and the hold-over ends.

    `HoldoverPoint` is independent from `FlowRefiner` - it tracks the last
    ACCEPTED point (not the anchor), so it can be applied in any flow mode
    (off / pure / fusion), and it runs an LK prediction only when a hold-over
    frame is needed (usually one extra LK call per missing frame).

    Parameters
    ----------
    accept_tol   : max distance (px) from the ghost at which an incoming
                   anchor is accepted as "the active re-appeared".  0 = auto
                   (max(80, 3 * median of last accepted steps)).
    max_lost     : max consecutive ghost frames before the state is reset.
    win_size     : LK search window (odd, clamped 15..91); 0 = auto from
                   the last accepted Person.
    """

    def __init__(self,
                 accept_tol: float = 0.0,
                 max_lost: int = 60,
                 win_size: int = 0):
        self.accept_tol = accept_tol
        self.max_lost = max_lost
        self.enabled = True   # False = baseline behavior: missing anchor = None
        self._win_override = win_size if win_size > 0 else None
        self.win_size = 31
        self._gray: Optional[np.ndarray] = None
        self._p: Optional[np.ndarray] = None               # last accepted
        self.step_history: List[float] = []                # rolling accepted steps
        self.lost_run = 0
        self.total_hold = 0
        self.max_run = 0

    def auto_tol(self) -> float:
        if self.accept_tol > 0:
            return self.accept_tol
        if self.step_history:
            med = float(np.median(self.step_history))
            return max(80.0, 3.0 * med)
        return 100.0

    def push_accepted(self, prev_accepted: Optional[Tuple[float, float]],
                      curr: Optional[np.ndarray]) -> None:
        """Called by tracking_stream for every non-ghost accepted point;
        feeds the rolling step history used by auto_tol() so the threshold
        tracks real per-frame motion."""
        if prev_accepted is None or curr is None:
            return
        prev = np.asarray(prev_accepted, dtype=np.float64)
        step = float(np.hypot(*((np.asarray(curr, dtype=np.float64) - prev))))
        if step > 0.5:
            self.step_history.append(step)
            if len(self.step_history) > 40:
                self.step_history = self.step_history[-40:]

    def _set_win(self, person: Optional[Person]) -> None:
        if self._win_override is not None:
            self.win_size = odd_clamp(int(self._win_override), lo=15, hi=91)
            return
        if person is None:
            return
        est = person_scale_estimate(person)
        if est:
            self.win_size = odd_clamp(int(min(91, max(15, est * 1.2))),
                                      lo=15, hi=91)

    def decide(self, gray: np.ndarray,
               anchor: Optional[Tuple[float, float]],
               person: Optional[Person]) -> Tuple[Optional[Tuple[float, float]], bool]:
        """
        Per-frame decision: return (accepted_point, used_ghost).

        `anchor`  - the candidate point from the tracker (None when the
                    active partner was not detected this frame).
        `person`  - the matched Person (for adaptive LK window), may be None.

        Accepts the anchor when it is within `auto_tol()` of the last accepted
        point (the flow-predicted ghost).  Otherwise this is a hold-over frame
        (active lost, or matcher grabbed the passive): we keep the ghost via
        a cheap LK prediction from the last accepted point.  After `max_lost`
        consecutive hold-over frames the state resets to re-seed on the next
        good anchor.
        """
        a = None if anchor is None else np.asarray(anchor, dtype=np.float32)
        if self._gray is None or self._p is None:
            # seed (first usable frame)
            self._gray = gray
            if a is not None:
                self._p = a.copy()
            self._set_win(person)
            return (tuple(float(v) for v in a) if a is not None else None), False

        tol = self.auto_tol()
        if a is not None:
            d = float(np.hypot(*((a - self._p))))
            if d <= tol:
                # active partner present and consistent -> accept anchor
                if self.lost_run:
                    self.max_run = max(self.max_run, self.lost_run)
                    self.lost_run = 0
                self.push_accepted(tuple(float(v) for v in self._p), a)
                self._p = a.copy()
                self._set_win(person)
                self._gray = gray
                return (tuple(float(v) for v in a)), False

        # hold-over (only when enabled: hold_tol <= inf AND max_lost > 0)
        if not self.enabled:
            # baseline: no anchor this frame -> empty slot, like the old code
            self._gray = gray
            if self.lost_run:
                self.max_run = max(self.max_run, self.lost_run)
            self.lost_run = 0
            return None, False

        pred = self._predict(gray, self._p)
        if pred is not None:
            # LK tracked the active partner's texture - the ghost wins over
            # a far anchor (the passive): this frame stays on the active.
            started = (self.lost_run == 0 and a is not None
                       and float(np.hypot(*((a - self._p)))) > tol)
            self._p = pred
            self.lost_run += 1
            self.total_hold += 1
            self._gray = gray
            if started:
                prev_pt = (self._p[0] - (pred[0] - self._p[0]),
                           self._p[1] - (pred[1] - self._p[1]))
                if a is not None:
                    print(f"[holdover] hold_start: anchor "
                          f"({a[0]:.0f},{a[1]:.0f}) is far from the tracked "
                          f"point ({prev_pt[0]:.0f},{prev_pt[1]:.0f}) - "
                          f"likely the passive partner; ghost continues")
                else:
                    print(f"[holdover] hold_start: active not detected "
                          f"this frame; ghost continues")
            return (float(pred[0]), float(pred[1])), True

        # LK itself is lost - only now accept a far anchor (true re-attach
        # when the active partner reappears somewhere else) or hold the
        # last accepted point unchanged.
        if a is not None:
            reattach = (self.lost_run > 0)
            if self.lost_run:
                self.max_run = max(self.max_run, self.lost_run)
                self.lost_run = 0
            self.push_accepted(tuple(float(v) for v in self._p), a)
            self._p = a.copy()
            self._gray = gray
            if reattach:
                print(f"[holdover] reattach: anchor "
                      f"({a[0]:.0f},{a[1]:.0f}) adopted after LK loss")
            return (tuple(float(v) for v in a)), False
        if self.lost_run < self.max_lost:
            # no anchor + LK lost: hold last accepted (static ghost)
            self.lost_run += 1
            self.total_hold += 1
            self._gray = gray
            return (float(self._p[0]), float(self._p[1])), True
        # gave up: reset so the next good anchor re-seeds cleanly
        if self.lost_run:
            self.max_run = max(self.max_run, self.lost_run)
        self.lost_run = 0
        self._p = None
        self._gray = gray
        return None, True

    def _predict(self, cur: np.ndarray,
                 p: np.ndarray) -> Optional[np.ndarray]:
        try:
            new, status, _ = FlowRefiner._lk_call(self._gray, cur,
                                                   np.asarray(p), self.win_size)
        except Exception:
            return None
        new = np.asarray(new, dtype=np.float32).reshape(-1)
        if new.size != 2:
            return None
        ok = bool(status[0][0]) if status.size else False
        h, w = cur.shape[:2]
        if not ok or not (0 <= new[0] < w and 0 <= new[1] < h):
            return None
        drift = float(np.hypot(*((new.astype(np.float64)
                                 - np.asarray(p, np.float64)))))
        # allow the ghost to move a reasonable distance (up to 2x the accept
        # tolerance) so fast but real motion is still followed
        if drift > 2.0 * self.auto_tol():
            return None
        return new
