"""
Optional anti-jitter refinement of the tracked point p via LK optical
flow (calcOpticalFlowPyrLK). The pose model (YOLO / MediaPipe / SOT) still
runs every frame as the ANCHOR; the refiner only decides which value to keep.

Modes:
  pure   - LK flow is the primary state (1 tracked point); when the flow
           result is inconsistent the point snaps back to the pose anchor
           (rebind) and re-seeds.
  fusion - a small particle field around the anchor is flow-tracked;
           the flow position wins when it moves consistently and stays
           near the anchor, otherwise the anchor wins (rebind + the field
           is re-centered on the accepted point).

Adaptation:
  winSize  - auto from the person scale on screen (hip span / 2 and
             thigh bone / 10, confidence-filtered, median, clamped to
             15..71 odd); a fixed size can be forced (win_size > 0).
  max_jump - max plausible per-frame motion = 0.06 * min(box w, h);
             can be overridden (max_jump_px > 0).
"""
from typing import Optional

import cv2
import numpy as np

from core.tracking.scaling import odd_clamp, person_scale_estimate
from core.types import Person


class FlowRefiner:
    def __init__(self, mode: str, win_size: int = 0, max_jump_px: float = 0.0):
        self.mode = mode
        self._win_override = win_size if win_size > 0 else None
        self._jump_override = max_jump_px if max_jump_px > 0 else None
        self.prev_gray: Optional[np.ndarray] = None
        self.win_size = 15
        self.max_jump = 30.0
        self.rebinds = 0
        # fusion: particle field (N, 2)
        self._parts: Optional[np.ndarray] = None
        # pure: single point + suspicion streak
        self._p: Optional[np.ndarray] = None
        self._dupe = 0

    def step(self, gray: np.ndarray,
             anchor: np.ndarray,
             person: Optional[Person]) -> Optional[np.ndarray]:
        """
        gray   - current grayscale frame
        anchor - the point from the pose tracker (x, y)
        person - the matched Person (for scale estimation), may be None.
        Returns the accepted point (x, y); None on the first frame only.
        """
        h, w = gray.shape[:2]
        # adaptive winSize (slow change, <= 1 step per frame)
        if self._win_override is not None:
            target = self._win_override
        elif person is not None:
            est = person_scale_estimate(person)
            target = odd_clamp(est) if est else self.win_size
        else:
            target = self.win_size
        if target != self.win_size:
            s = max(-15, min(15, target - self.win_size))
            self.win_size = odd_clamp(self.win_size + s)
        # adaptive max jump (per-frame motion budget)
        if self._jump_override is not None:
            self.max_jump = float(self._jump_override)
        elif person is not None:
            self.max_jump = max(6.0, 0.06 * min(person.box[2], person.box[3]))

        if self.prev_gray is None:
            self.prev_gray = gray
            self._seed(anchor)
            return np.asarray(anchor, dtype=np.float32)

        if self.mode == "pure":
            accepted = self._step_pure(gray, anchor, h, w)
        else:  # fusion
            accepted = self._step_fusion(gray, anchor, h, w)
        self.prev_gray = gray
        return np.asarray(accepted, dtype=np.float32)

    # ── internals ──────────────────────────────────────────────
    def _seed(self, anchor: np.ndarray) -> None:
        if self.mode == "pure":
            self._p = np.asarray(anchor, dtype=np.float32).copy()
            self._dupe = 0
        else:
            self._parts = self._grid(anchor)

    def _grid(self, center: np.ndarray) -> np.ndarray:
        """9-particle cross grid around center, scaled to the window size."""
        r = max(4.0, self.win_size * 0.5)
        offs = np.asarray([(0, 0), (r, 0), (-r, 0), (0, r), (0, -r),
                           (2 * r, 0), (-2 * r, 0), (0, 2 * r), (0, -2 * r)],
                          dtype=np.float32)
        return np.asarray(center, dtype=np.float32)[None, :] + offs

    @staticmethod
    def _lk_call(prev: np.ndarray, cur: np.ndarray,
                 pts: np.ndarray, win_size: int):
        return cv2.calcOpticalFlowPyrLK(
            prev, cur, pts.reshape(-1, 1, 2).astype(np.float32), None,
            winSize=(win_size, win_size), maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-4))

    def _step_pure(self, gray: np.ndarray, anchor: np.ndarray,
                   h: int, w: int) -> np.ndarray:
        new, status, _ = self._lk_call(self.prev_gray, gray, self._p,
                                       win_size=self.win_size)
        new = new.reshape(-1)                      # flat (2,): [x, y]
        ok = (bool(status[0][0]) if status.size else False) \
            and (0 <= new[0] < w) and (0 <= new[1] < h)
        drift = float(np.hypot(*((new.astype(np.float64) - self._p))))
        disagree = float(np.hypot(*(np.asarray(anchor, np.float64) - new)))
        if ok and drift <= self.max_jump and disagree <= 3.0 * self.max_jump:
            if disagree > self.max_jump:
                self._dupe += 1            # sliding away from the anchor
            else:
                self._dupe = 0
            self._p = new.astype(np.float32).copy()
            return self._p
        # flow lost, or inconsistent (drift too large, far from anchor, or
        # suspicious streak) -> rebind to the anchor
        self.rebinds += 1
        self._dupe = 0
        self._p = np.asarray(anchor, dtype=np.float32).copy()
        return self._p

    def _step_fusion(self, gray: np.ndarray, anchor: np.ndarray,
                     h: int, w: int) -> np.ndarray:
        new, status, _ = self._lk_call(self.prev_gray, gray, self._parts,
                                       win_size=self.win_size)
        st = status.reshape(-1)
        new = new.reshape(-1, 2)
        anchor = np.asarray(anchor, dtype=np.float32)
        best_k, best_score = None, 1e18
        for k in range(len(new)):
            if not st[k]:
                continue
            if not (0 <= new[k, 0] < w and 0 <= new[k, 1] < h):
                continue
            drift = float(np.hypot(*(new[k].astype(np.float64) - self._parts[k])))
            if drift > self.max_jump:
                continue
            disagree = float(np.hypot(*(anchor.astype(np.float64) - new[k])))
            if disagree > 3.0 * self.max_jump:
                continue
            deadband = self.max_jump        # flow may legitimately lead the anchor
            score = drift / max(1.0, 0.20 * self.max_jump) \
                + max(0.0, disagree - deadband) / max(1.0, 0.60 * self.max_jump)
            if score < best_score:
                best_score, best_k = score, k
        if best_k is not None:
            accepted = new[best_k].astype(np.float32).copy()
        else:
            accepted = anchor.copy()
            self.rebinds += 1
        # keep the particle field centered on the accepted point
        if self._parts is not None:
            center_old = self._parts.mean(axis=0)
            self._parts = self._parts - center_old \
                + np.asarray(accepted, dtype=np.float32)
        return accepted

