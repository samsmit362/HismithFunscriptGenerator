from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Optional, Sequence, Tuple
from core.logging import log

import numpy as np


@dataclass
class StrokeInfo:
    frame_start: int          # video frame index where this stroke begins
    frame_end: int            # video frame index where this stroke ends (inclusive)
    direction: str            # 'advance' | 'retract'
    axis: np.ndarray          # unit thrust axis for this stroke
    total_len: float          # raw polyline length (pixels)
    u_span: float             # signed (u_end - u_start) on the local axis
    max_pos_raw: float        # max position value before normalization
    u_rel: Optional[np.ndarray] = None
    conic_coef: Optional[Tuple[float, float, float]] = None


def stroke_axis(prev_axis: Optional[np.ndarray],
                stroke_pts: np.ndarray) -> np.ndarray:
    """Length-weighted PCA of the stroke's displacements (axis = the dominant
    motion direction, sign-free - used only by the straight / arc / parabola
    position models; 'complex' ignores it).

    Falls back to prev_axis (or a default) when the stroke is too short or
    degenerate to estimate a stable axis.  NO angle clamp is applied - the
    axis may drift across strokes to follow the camera; the direction label
    (advance/retract) is owned by the StrokeStateMachine, not by this axis.
    """
    if len(stroke_pts) < 3:
        return prev_axis if prev_axis is not None else np.array([0.0, 1.0])
    segs = np.diff(stroke_pts, axis=0)
    norms = np.linalg.norm(segs, axis=1)
    wsum = float(np.sum(norms))
    if wsum < 1e-6:
        return prev_axis if prev_axis is not None else np.array([0.0, 1.0])
    C = np.zeros((2, 2), dtype=np.float64)
    for s, wi in zip(segs, norms):
        C += wi * np.outer(s, s)
    try:
        evals, evecs = np.linalg.eigh(C)
        cand = evecs[:, int(np.argmax(evals))]
        nrm = float(np.linalg.norm(cand))
        if nrm < 1e-9:
            return prev_axis if prev_axis is not None else np.array([0.0, 1.0])
        cand = cand / nrm
    except np.linalg.LinAlgError:
        return prev_axis if prev_axis is not None else np.array([0.0, 1.0])
    return cand


def arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


class StrokeStateMachine:
    """
    Incremental stroke-direction state machine + reversal-commit boundaries.

    Stroke model - a stroke = one continuous pump motion measured from its
    ANCHOR (the stroke origin; for the first stroke = the first accepted
    point, for each later stroke = the extreme of the previous stroke).

    Rule 1 (once): the first stroke's direction is set by the calibration
    axis bf (unit b->f) against the FIRST significant displacement s0 of
    p from the anchor (|s0| >= INIT_EPS to count):
       <bf, s0> > 0  ->  'advance' (toward f, pos 100 -> 0)
      <bf, s0> < 0  ->  'retract' (toward b, pos   0 -> 100)
      bf = None -> 'advance' by convention.

    Commit (reversal boundary): progress D = ||p - anchor|| (Euclidean
    distance from the stroke origin, always >= 0, direction-free);
    M = max(D) so far = the stroke amplitude; the argmax-D frame is the
    stroke extreme (star_idx / star_pt / star_t).  The moment p starts to
    move AWAY from the extreme (D < M) the reversal commits immediately —
    boundary = the EXTREME frame, so pos 0/100 lands exactly on the
    physical f/b instant with zero lag.  No hysteresis / 30% gate: every
    retreat past the peak opens the next stroke.

    Direction defender (SOFTENED): before each commit the machine checks that
    the stroke it just finished actually moved in the direction it was
    labelled as, measured along bf.  When these disagree (e.g. rule 1 says
    'advance' but p moved AWAY toward b) the finished stroke is RE-LABELLED
    to its actual direction (dir = the sign of <star_pt - anchor, bf>) and the
    commit is recorded normally - predicting the exact end of motion ahead is
    hard, so we fix the label instead of discarding the rest of the track.
    The original "raise Exception('stop')" (a GRACEFUL STOP that halts the
    streaming preview and truncates the post-pass) is still in feed() with
    the raise commented out; re-enable it if re-labelling mis-fires.  The
    live preview and the post-pass both run the (softened) defender, so the
    dynamic f/b marker and the generated positions stay consistent.

    feed(idx, pt, cur_t) is called once per ACCEPTED point (ghost / missing
    frames are skipped by the caller), in order, with the caller's own index
    basis.  `cur_t` is the REAL presentation time (seconds) of the frame that
    `pt` came from - it is REQUIRED and always goes together with `pt` (the
    two describe the SAME physical frame).  Returns the commit event dict
    {star_idx, star_pt, star_t, dir, amp} when a reversal committed, else
    None.  `dir` is the ACTUAL direction of the stroke that ends at this
    commit (defender-corrected against bf); `star_t` is the real time of the
    stroke extremum (star_pt) so downstream (clean_commits) can place the
    merged boundary at a true TIME midpoint, not just an index midpoint.
    """

    INIT_EPS = 3.0         # px, minimum first displacement for rule 1

    def __init__(self, bf_axis: Optional[np.ndarray] = None,
                 markers: Optional[Tuple] = None):
        """
        bf_axis : unit axis b->f ('advance' direction), or None (derived)
        markers : (f, b) user calibration points - kept for reference /
                  future use; the anchor of the first stroke is the first
                  accepted tracked point, not a user marker
        """
        bf = None
        if bf_axis is not None:
            b = np.asarray(bf_axis, dtype=np.float64).ravel()
            nrm = float(np.linalg.norm(b))
            if nrm > 1e-9:
                bf = b / nrm
        self.bf = bf
        fm = bm = None
        if markers is not None:
            fm, bm = markers
        self.marker_f = np.asarray(fm, dtype=np.float64) if fm is not None else None
        self.marker_b = np.asarray(bm, dtype=np.float64) if bm is not None else None

        self.initial_dir = "advance"
        self.dir = "advance"
        self.anchor = None       # A0 = current stroke origin (the dynamic f/b)
        self.D = 0.0            # progress = distance from the anchor (>= 0)
        self.M = 0.0            # amplitude (max D in the current stroke)
        self.star_idx = -1      # index of the running extreme (argmax D)
        self.star_pt = None     # the extreme point itself
        self.star_t = None      # real time of the running extreme (of star_idx)
        self.prev_pt = None
        self.started = False
        self.min_move = 10
        self.commits: List[dict] = []

    @property
    def n_strokes(self) -> int:
        return len(self.commits) + 1

    def feed(self, idx, pt, cur_t):
        """Feed one accepted point (index basis = caller's choice).
        `cur_t` is the REAL time (seconds) of the frame that `pt` came from -
        REQUIRED, always passed together with `pt` (same physical frame).
        Returns the commit event dict when a reversal committed, else None.
        May raise Exception("stop") when the direction defender fires."""
        p = np.asarray(pt, dtype=np.float64).ravel()
        if not self.started:
            if self.anchor is None:
                # first accepted point = anchor of the first stroke
                self.anchor = p.copy()
                self.prev_pt = p.copy()
                return None
            s0 = p - self.anchor
            if float(np.linalg.norm(s0)) < self.INIT_EPS:
                # still too close to the anchor: no direction yet
                self.prev_pt = p.copy()
                return None
            # rule 1: first significant displacement vs the calibration axis
            if self.bf is not None:
                advance = float(s0 @ self.bf) > 0.0
            else:
                advance = True
            self.initial_dir = "advance" if advance else "retract"
            log(f"[StrokeStateMachine.feed]: self.initial_dir: {self.initial_dir} self.bf: {self.bf}")
            self.dir = self.initial_dir
            self.started = True
        D = float(np.linalg.norm(p - self.anchor))
        self.D = D
        if self.star_pt is None:
            self.star_idx = int(idx)
            self.star_pt = p.copy()
            self.star_t = float(cur_t)
            self.M = D
        elif D >= self.M:
            self.star_idx = int(idx)
            self.star_pt = p.copy()
            self.star_t = float(cur_t)
            self.M = D
        move_from_stroke_end = float(np.linalg.norm(p - self.star_pt))
        if D < self.M and move_from_stroke_end >= self.min_move:
            # direction defender: the stroke we are finishing must actually
            # have moved the way it was labelled; a conflict is a stop signal
            if self.bf is not None:
                toward_f = float((self.star_pt - self.anchor) @ self.bf) > 0.0
                implied_dir = "advance" if toward_f else "retract"
                if implied_dir != self.dir:
                    log(f"[StrokeStateMachine.feed]: fixed dir:{self.dir} => {implied_dir} [p_end]prev_pt==star_pt:{self.star_pt} ==?{self.star_pt == self.prev_pt} {self.prev_pt} [p_start]anchor:{self.anchor} [id_end]star_idx:{self.star_idx} pt:{pt} cur_t:{float(cur_t)}")
                    self.dir = implied_dir
                    #raise Exception("stop")
                else:
                    log(f"[StrokeStateMachine.feed]: dir:{self.dir} [p_end]prev_pt==star_pt:{self.star_pt} ==?{self.star_pt == self.prev_pt} {self.prev_pt} [p_start]anchor:{self.anchor} [id_end]star_idx:{self.star_idx} pt:{pt} cur_t:{float(cur_t)}")
            else:
                b = self.star_pt - self.anchor
                nrm = float(np.linalg.norm(b))
                if nrm > 1e-9:
                    self.bf = b / nrm
            # COMMIT: the point has retreated from the extreme - the stroke
            # boundary is the extreme itself (star_idx / star_pt / star_t),
            # the next stroke starts from it.
            evt = {
                "star_idx": self.star_idx,
                "star_pt": self.star_pt.copy(),
                "star_t": self.star_t,   # real time of the stroke extreme
                "dir": self.dir,      # direction of the stroke that ends
                "amp": self.M,
            }
            self.commits.append(evt)
            self.dir = "retract" if self.dir == "advance" else "advance"
            self.anchor = self.star_pt.copy()
            D = float(np.linalg.norm(p - self.anchor))
            self.D = D
            self.M = D
            self.star_idx = int(idx)
            self.star_pt = p.copy()
            self.star_t = float(cur_t)
            self.prev_pt = p.copy()
            return evt
        self.prev_pt = p.copy()
        return None

def clean_commits(pts: np.ndarray, sm: StrokeStateMachine,
                   times) -> int:
    """Merge consecutive same-direction commits into a single commit.

    When the direction of a real motion is hard to predict ahead, the state
    machine may commit several strokes in a row whose labels do NOT
    alternate (e.g. advance, advance, retract).  Physically those back-to-
    back same-direction strokes are ONE motion whose end the machine could
    not place precisely - they are stitched together into a SINGLE commit
    and the stroke boundary is placed at the TIME MIDPOINT between the prev
    to last and the last original extreme.

    The midpoint is computed over the REAL presentation times carried on each
    commit (`star_t`, now REQUIRED - see StrokeStateMachine.feed):
       t_mid  = (prev_to_last.star_t + last.star_t) / 2
       star_idx = the valid-point index whose time is closest to t_mid
       star_pt  = pts[star_idx]   (a real tracked frame - the boundary stays
                                  on an existing frame)
       star_t   = pt_times[mid]   (the REAL time of the closest
                                    tracked frame, not the
                                    synthetic t_mid)
       dir      = the shared direction
       amp      = the LAST commit's amp  (metadata, used only in logs; the
                   merged strokes below derive their own amplitudes)

    Single-commit commits (runs of length 1) are kept unchanged.  Returns the
    number of commits removed (0 = no same-direction runs, no change).

    Invariant that keeps downstream correct: after merging, each stroke still
    gets exactly ONE consistent direction (commit `c_i` labels stroke `i`,
    the final stroke gets the flip of `c_last`) - see the `dirs` build in
    `segment_strokes` for the pairing logic.

    `pts` MUST be the valid-point array in the SAME index basis the commits'
    `star_idx` was built with (in `segment_strokes` that is P = A[valid_idx];
    do NOT pass the dense array A - the indices would mismatch when frames
    were skipped / ghosted).
    """
    if not sm.commits:
        return 0
    log(f"[StrokeStateMachine.clean_commits]: original commits {len(sm.commits)}\n-------------")
    for c in sm.commits:
        log(f"[StrokeStateMachine.clean_commits]: dir: {c["dir"]} [p_end]star_pt:{c["star_pt"]} [id_end]star_idx:{c["star_idx"]} [t_end]star_t:{c["star_t"]}")
    log("-------------")
    cleaned: List[dict] = []
    removed = 0
    for direction, group in groupby(sm.commits, key=lambda c: c["dir"]):
        items = list(group)
        if len(items) == 1:
            cleaned.append(dict(items[0]))
            continue
        prev_to_last = items[-2]
        last = items[-1]
        # TIME midpoint over the real presentation times carried on the commits
        # (star_t is REQUIRED now - see StrokeStateMachine.feed).  The boundary
        # index is then the valid-point whose star_t is closest to t_mid, so the
        # merged boundary stays on a REAL tracked frame.
        t_mid = (float(prev_to_last["star_t"]) + float(last["star_t"])) / 2.0
        pt_times = np.asarray(times, dtype=np.float64)
        if len(pt_times) != len(pts):
            raise ValueError(
                "clean_commits: times has {} entries but pts has {} points "
                "(they must be aligned 1:1).".format(len(pt_times), len(pts)))
        mid = int(np.argmin(np.abs(pt_times - t_mid)))
        mid = max(0, min(len(pts) - 1, mid))
        cleaned.append({
            "star_idx": mid,
            "star_pt": np.asarray(pts[mid], dtype=np.float64).copy()
                       if len(pts) > 0 else last["star_pt"].copy(),
            "star_t": float(pt_times[mid]),
            "dir": direction,
            "amp": last["amp"],
        })
        removed += len(items) - 1
    sm.commits = cleaned
    if removed:
        log(f"[strokes] merged {removed} same-direction commit(s) "
            f"to time-midpoint boundary -> {len(cleaned)} stroke(s)")
    log(f"[StrokeStateMachine.clean_commits]: result commits {len(sm.commits)}\n-------------")
    for c in sm.commits:
        log(f"[StrokeStateMachine.clean_commits]: dir: {c["dir"]} [p_end]star_pt:{c["star_pt"]} [id_end]star_idx:{c["star_idx"]} [t_end]star_t:{c["star_t"]}")
    log("-------------")
    return removed


def segment_strokes(A: np.ndarray,
                    valid_idx: List[int],
                    global_axis: Optional[np.ndarray],
                    times: Sequence[float],
                    cal_f: Optional[np.ndarray] = None,
                    cal_b: Optional[np.ndarray] = None) -> List[StrokeInfo]:
    """
    Segment frames into forward/backward strokes.

    Direction + boundaries: StrokeStateMachine (see its docstring):
      rule 1 - the first stroke's direction is the sign of
                <calibration axis bf, first displacement from the anchor>;
      commit - the moment the point retreats from the stroke extreme
                (D < M) the direction flips; the boundary is the EXTREME
                frame (argmax D), so pos 0/100 lands exactly on the real
                f/b instant of the video (no lag); the next stroke starts
                at that same extreme frame (no gap, no duplicate);
      defender - SOFTENED: a mislabelled finishing stroke is re-labelled to
                  its actual direction (the sign of the dot product of the
                  stroke displacement with bf) and the commit is recorded
                  normally; the original `stop` path is still in feed() with
                  the raise commented out.
      merge  - clean_commits(P, sm) runs AFTER the SM is fed, stitching
                  consecutive same-direction commits (noise / uncertain
                  end-of-motion) into ONE commit whose boundary is the TIME
                  MIDPOINT between the original extremes; clean (alternating)
                  commit lists pass through unmodified.
      labels - stroke i takes commit i's REAL `dir` field (defender-
                  corrected).  The final in-progress stroke gets the FLIP of
                  the last committed direction.  NOT the old "alternate from
                  initial_dir" logic (which broke on same-direction runs).

    A LOCAL PCA axis per stroke is still estimated - needed by the
    straight / arc / parabola modes ('complex' uses only the direction
    label + the raw path length).

    A           : dense (N,2) track array (missing frames carried forward)
    valid_idx   : frame indices where a real detection happened
    global_axis : calibration axis (b->f), unit or None (derived from P if None)
    times       : REAL presentation time (seconds) for EACH valid point,
                  aligned 1:1 with `valid_idx` (times[k] is the real time of
                  P[k], i.e. A[valid_idx[k]]).  REQUIRED for correct time-
                  midpoint merging: it is forwarded as `cur_t` to the state
                  machine's feed() so each commit carries its real star_t.
                  When None it is synthesized as start_s-free index times
                  (times[k] = valid_idx[k]) so the contract still holds.
    cal_f/cal_b : user f/b marks - forwarded to the state machine
                  (reference for the direction defender)
    """
    min_len = 1        # minimum number of valid points in a stroke

    if len(valid_idx) < 2:
        f0 = valid_idx[0] if valid_idx else 0
        ax = global_axis if global_axis is not None else np.array([0.0, 1.0])
        return [StrokeInfo(frame_start=f0, frame_end=f0, direction="advance",
                           axis=ax, total_len=0.0, u_span=0.0, max_pos_raw=0.0)]

    P = A[valid_idx]

    if global_axis is None:
        c = P.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(P - c)
            global_axis = Vt[0]
        except np.linalg.LinAlgError:
            global_axis = np.array([0.0, 1.0])

    # Per-valid-point REAL times (1:1 with P).  Required by feed() as cur_t.
    # When the caller supplies them (compute_positions / tests) use them as-is;
    # otherwise fall back to the dense frame index as a degenerate time basis.
    if times is None:
        raise ValueError(
            "segment_strokes: `times` is REQUIRED (REAL presentation time "
            "per valid point, aligned 1:1 with valid_idx).")
    if len(times) != len(valid_idx):
        raise ValueError(
            "segment_strokes: times has {} entries but valid_idx has "
            "{} indices (they must be aligned 1:1).".format(len(times), len(valid_idx)))
    valid_times = [float(times[k]) for k in range(len(valid_idx))]

    # Feed the track to the state machine in valid-point index space.
    # The direction defender may raise Exception("stop") part-way through
    # (see the class docstring) - that is a GRACEFUL STOP: the strokes
    # collected up to that moment (sm.commits) are still valid and get
    # segmented below; the remainder of the track is inconsistent with the
    # calibration axis and is left out of the result.
    sm = StrokeStateMachine(global_axis, markers=(cal_f, cal_b))
    last_valid_k = 0
    stopped = False
    k = 0
    while k < len(P):
        try:
            sm.feed(k, P[k], valid_times[k])
        except Exception as e:
            if str(e) != "stop":
                raise
            # the stroke that ends here committed up to the LAST commit;
            # the in-progress stroke after the last commit boundary has no
            # committed extreme - cut the track there
            stopped = True
            if sm.commits:
                last_valid_k = sm.commits[-1]["star_idx"]
            break
        last_valid_k = k
        k += 1
    if stopped:
        P = P[:last_valid_k + 1]

    # Stitch together consecutive same-direction commits (noise / uncertain
    # end-of-motion): clean_commits collapses each run into one commit whose
    # boundary is the TIME MIDPOINT between the first and last original
    # extreme, so the stroke endpoints stay at real tracked frames.
    # Runs of length 1 are left as-is (no-op) - in clean cases the commit
    # list is unchanged and downstream behavior matches the pre-merge run.
    clean_commits(P, sm, times=valid_times)

    # Stroke boundaries in valid-point index space.  Every commit's star_idx
    # is the END point of the stroke that ends there (its last frame; after
    # clean_commits: the real-time midpoint of the merged run).
    # Ranges = [0, star_1], [star_1, star_2], ..., [star_N, N-1] - the
    # NEXT stroke starts at the boundary itself (pos continues from
    # 0/100, so no duplicate frames, no gap).
    bounds = [c["star_idx"] for c in sm.commits]
    ranges: List[List[int]] = []
    prev = 0
    for b_ in bounds:
        ranges.append([prev, b_])
        prev = b_
    ranges.append([prev, len(P) - 1])
    # drop ranges shorter than min_len by absorbing them into the previous one
    merged_ranges: List[List[int]] = []
    for a, b_ in ranges:
        if merged_ranges and (b_ - a + 1) < min_len:
            merged_ranges[-1][1] = b_
        else:
            merged_ranges.append([a, b_])
    if (merged_ranges and (merged_ranges[0][1] - merged_ranges[0][0] + 1) < min_len
            and len(merged_ranges) > 1):
        merged_ranges[1][0] = merged_ranges[0][0]
        merged_ranges.pop(0)
    # Stroke direction comes from the REAL commit label (the state machine
    # already resolved it against the calibration axis during feed) - NOT
    # from alternating, which assumes every commit flips direction and breaks
    # when the run of back-to-back same-direction commits was just merged.
    # Stroke i ends at commit i: its direction IS commit i's dir; the final
    # in-progress stroke (no commit yet) ends the run, so it gets the flip
    # of the last committed direction.  With zero commits the single stroke
    # runs in the rule-1 initial_dir (as before).
    dir_at: Dict[int, str] = {}
    dirs = [c["dir"] for c in sm.commits]
    if sm.commits:
        dirs.append("retract" if dirs[-1] == "advance" else "advance")
    else:
        dirs = [sm.initial_dir]
    for d, (a, b_) in zip(dirs, ranges):
        for idx in range(a, b_ + 1):
            dir_at[idx] = d

    result: List[StrokeInfo] = []
    prev_axis: Optional[np.ndarray] = global_axis
    for a, b_ in merged_ranges:
        frame_start = int(valid_idx[a])
        frame_end = int(valid_idx[b_])
        P_stroke = A[frame_start:frame_end + 1]
        local_axis = stroke_axis(prev_axis, P_stroke)
        u_rel = (P_stroke - P_stroke[0]) @ local_axis
        u_span = float(u_rel[-1] - u_rel[0])
        result.append(StrokeInfo(
            frame_start=frame_start,
            frame_end=frame_end,
            direction=dir_at.get(a, "advance"),
            axis=local_axis,
            total_len=arc_length(P_stroke),
            u_span=u_span,
            max_pos_raw=0.0,
            u_rel=u_rel,
        ))
        prev_axis = local_axis
    return result
