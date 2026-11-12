"""
Per-stroke position computation (straight / arc / parabola / complex) and
the full post-pass assembly (compute_positions): dense track -> strokes ->
per-frame pos -> normalization (raw 0..100 scaling kept as-is, no pre-denoise).
"""
from typing import List, Optional, Tuple

import numpy as np

from core.postpass.strokes import (StrokeInfo, segment_strokes)


def fit_conic(pts: np.ndarray, axis: np.ndarray):
    """
    Fit a quadratic v(u) = a0 + a1*u + a2*u^2 to the stroke points, where
        u = coordinate along `axis` (rebased so that u[0] == 0)
        v = coordinate perpendicular to `axis` (also rebased, signed)
    Returns (coef, u) tuple or None on failure.
    """
    P = np.asarray(pts, dtype=np.float64)
    if len(P) < 3:
        return None
    origin = P[0]
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    u = (P - origin) @ axis
    v = (P - origin) @ perp
    M = np.column_stack([np.ones_like(u), u, u * u])
    try:
        coef, *_ = np.linalg.lstsq(M, v, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return (tuple(coef), u)


def conic_arc_fractions(coef, u: np.ndarray, n_sample: int = 1024) -> np.ndarray:
    """Fraction of conic arc length from u[0] to each u[i] (0..1 by construction)."""
    a0, a1, a2 = coef
    u_lo = float(min(u.min(), u[0]))
    u_hi = float(max(u.max(), u[0]))
    if u_hi - u_lo < 1e-9:
        return np.zeros(len(u), dtype=np.float64)
    grid = np.linspace(u_lo, u_hi, n_sample)
    dv_du = a1 + 2.0 * a2 * grid
    integrand = np.sqrt(1.0 + dv_du * dv_du)
    du = grid[1] - grid[0]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * du)])
    arc_at = np.interp(np.clip(u, u_lo, u_hi), grid, cum)
    out = np.abs(arc_at - arc_at[0])
    total = out[-1]
    if total < 1e-9:
        return np.zeros(len(u), dtype=np.float64)
    return out / total


def _conic_total_arc(coef, u: np.ndarray, n_sample: int = 1024) -> float:
    """Total arc length of the conic between the min and max u values present.
    (Same integration as conic_arc_fractions, but returns the absolute
    total length instead of the fractions.)"""
    a0, a1, a2 = coef
    u_lo = float(min(u.min(), u[0]))
    u_hi = float(max(u.max(), u[0]))
    if u_hi - u_lo < 1e-9:
        return 0.0
    grid = np.linspace(u_lo, u_hi, n_sample)
    dv_du = a1 + 2.0 * a2 * grid
    integrand = np.sqrt(1.0 + dv_du * dv_du)
    du = grid[1] - grid[0]
    return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * du))


def stroke_progress(stroke: StrokeInfo, stroke_pts: np.ndarray,
                    motion: str) -> Tuple[np.ndarray, float]:
    """
    Per-frame progress of ONE stroke, direction-free, plus its physical amplitude.

    Returns (frac, amp):
      frac : (N,) float, monotone from 0 (stroke origin) to 1 (stroke extreme).
             Direction-agnostic - the caller applies the advance/retract sign.
      amp  : the stroke's total path length (pixels), in the motion mode's
             natural unit:
              complex    -> raw polyline path length;
              straight   -> linear displacement on the local axis (axial span);
              arc/parab  -> arc length along the fitted conic
                            (total length = "length of path" in the user's algorithm).

    Motion modes (see AGENTS.md):
      straight -> linear projection on the stroke axis
      arc      -> arc length along a fitted conic
      parabola -> same conic fit (reported as a parabolic curve)
      complex  -> raw polyline length (the real, unidealized path of p)
    """
    pts = np.asarray(stroke_pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return np.zeros(n), 0.0

    if motion == "complex":
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        frac = cum / total if total > 1e-9 else np.zeros(n)
        amp = float(total)
    elif motion in ("arc", "parabola"):
        fit = fit_conic(pts, stroke.axis)
        if fit is None:
            u_rel = (pts - pts[0]) @ stroke.axis
            span = u_rel[-1]
            frac = (u_rel / span) if abs(span) > 1e-9 else np.zeros(n)
            amp = abs(span)
        else:
            coef, u = fit
            stroke.conic_coef = coef
            frac = conic_arc_fractions(coef, u)
            amp = _conic_total_arc(coef, u)
    else:  # straight
        u_rel = (pts - pts[0]) @ stroke.axis
        span = u_rel[-1]
        frac = (u_rel / span) if abs(span) > 1e-9 else np.zeros(n)
        amp = abs(span)

    frac = np.clip(frac, 0.0, 1.0)
    return frac, amp


def positions_for_motion(stroke: StrokeInfo, stroke_pts: np.ndarray,
                         motion: str) -> List[float]:
    """
    Per-frame position (0..100) inside ONE stroke, direction-aware.
    Motion towards f (advance) runs pos 100 -> 0.
    Motion towards b  (retract) runs pos  0 -> 100.
    """
    frac, _ = stroke_progress(stroke, stroke_pts, motion)
    if stroke.direction == "advance":
        frac = 1.0 - frac
    return [float(v * 100.0) for v in frac]


def _rescale_to_range(arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Linearly rescale a contiguous stroke so its FIRST frame lands on t0
    and its LAST frame lands on t1, preserving the stroke's internal shape
    (and direction) exactly.

    A stroke is always monotonic (frac 0..1 within the stroke), so
        out = t0 + (a - a[0]) * (t1 - t0) / (a[-1] - a[0])
    maps first->t0, last->t1 and all interior frames proportionally, for BOTH
    increasing (retract) and decreasing (advance) strokes.
    A flat stroke (a[0] == a[-1]) maps every frame to t0 (degenerate no-range).
    """
    a = np.asarray(arr, dtype=np.float64)
    n = a.size
    if n == 0:
        return a
    if n == 1 or abs(float(a[-1]) - float(a[0])) < 1e-9:
        return np.full_like(a, float(t0))
    scale = (t1 - t0) / (float(a[-1]) - float(a[0]))
    return t0 + scale * (a - a[0])


def _apply_boundary_anchors(pos: np.ndarray, strokes: list,
                            normalize: str,
                            anchor_begin_pos: Optional[float],
                            anchor_end_pos: Optional[float]) -> np.ndarray:
    """Rescale the boundary stroke(s) so the new track lands EXACTLY on the
    anchor points already on the timeline (requirement #3, user spec).

    Per-stroke: 1 boundary stroke at each edge, rescaled to
        BEGIN advance -> [P => 0],  BEGIN retract -> [P => 100]
        END   advance -> [100 => P], END   retract -> [0 => P]

    Global-max: 2 strokes at each edge (boundary + the stroke NEXT to it).
        Boundary: same as per-stroke (above).
        Neighbor (BEGIN):
             neighbor retract -> [0 => OLD_MAX of that stroke]
             neighbor advance -> [100 => OLD_MIN of that stroke]
        Neighbor (END):
             neighbor retract -> [OLD_MIN of that stroke => 100]
             neighbor advance -> [OLD_MAX of that stroke => 0]
        where OLD_MIN / OLD_MAX are the pre-rescale min/max of the neighbor's
        pos array (the "current_stroke_X_Y_value" from the user spec).

    Either anchor may be None (no existing point on that side of the new
    segment) - the affected stroke keeps its already-normalized value.
    """
    if not strokes:
        # Degenerate: the segment had so few tracked frames that the
        # stroke detector did not produce any usable stroke (the stroke
        # extreme must be at least 2 frames long, otherwise it is
        # discarded by the frame_end > frame_start filter).
        # We still have the boundary info: rescale the WHOLE track to
        # span [anchor_begin, anchor_end] so the single / very few frame(s)
        # land within the known physical range instead of an arbitrary
        # 0.0 / 50.0.  This is the "1 stroke but anchor_begin/end present"
        # case from the user spec: the stroke is shorter than one frame,
        # both endpoints are already on the OFS timeline, and the new
        # segment should simply hold one of the two known anchor values.
        if anchor_begin_pos is not None and anchor_end_pos is not None:
            pos = _rescale_to_range(pos, float(anchor_begin_pos),
                                    float(anchor_end_pos))
            print(f"[boundary] no strokes detected; pinned track to "
                  f"[{anchor_begin_pos}..{anchor_end_pos}] "
                  f"(sub-frame stroke, endpoints already on the timeline)")
        elif anchor_begin_pos is not None:
            pos = np.full_like(pos, float(anchor_begin_pos))
            print(f"[boundary] no strokes detected; pinned track to "
                  f"{anchor_begin_pos} (begin anchor only)")
        elif anchor_end_pos is not None:
            pos = np.full_like(pos, float(anchor_end_pos))
            print(f"[boundary] no strokes detected; pinned track to "
                  f"{anchor_end_pos} (end anchor only)")
        return np.clip(pos, 0.0, 100.0)
    n = len(strokes)

    # Capture OLD min/max of every stroke BEFORE any rescale (needed by the
    # global-max neighbor rule: it references the neighbor's OLD extreme).
    old = {}
    for i, s in enumerate(strokes):
        arr = pos[s.frame_start : s.frame_end + 1]
        if arr.size:
            old[i] = (float(arr.min()), float(arr.max()))
        else:
            old[i] = (0.0, 0.0)

    def apply(idx: int, t0: float, t1: float) -> None:
        s = strokes[idx]
        sl = slice(s.frame_start, s.frame_end + 1)
        pos[sl] = _rescale_to_range(pos[sl], t0, t1)
        print(f"[anchors] stroke {idx} ({s.direction}): "
              f"[{t0:.1f} => {t1:.1f}]")

    # ── BEGIN anchor: boundary = stroke 0 (+ neighbor = stroke 1 for global-max) ──
    if anchor_begin_pos is not None:
        P = float(anchor_begin_pos)
        if strokes[0].direction == "advance":
            apply(0, P, 0.0)
        else:
            apply(0, P, 100.0)
        if normalize == "global-max" and n >= 2:
            s1 = strokes[1]
            if s1.direction == "retract":
                apply(1, 0.0, old[1][1])       # [0 => OLD max of retract]
            else:
                apply(1, 100.0, old[1][0])     # [100 => OLD min of advance]

    # ── END anchor: boundary = stroke n-1 (+ neighbor = stroke n-2 for global-max) ──
    if anchor_end_pos is not None:
        P = float(anchor_end_pos)
        if strokes[n - 1].direction == "advance":
            apply(n - 1, 100.0, P)
        else:
            apply(n - 1, 0.0, P)
        if normalize == "global-max" and n >= 2:
            s2 = strokes[n - 2]
            if s2.direction == "retract":
                apply(n - 2, old[n - 2][0], 100.0)   # [OLD min of retract => 100]
            else:
                apply(n - 2, old[n - 2][1], 0.0)     # [OLD max of advance => 0]

    return np.clip(pos, 0.0, 100.0)


def compute_positions(raw_tracks: List[Optional[Tuple[float, float]]],
                      start_s: float, fps: float,
                      anchor_axis: Optional[np.ndarray],
                      motion: str, normalize: str,
                      cal_f=None, cal_b=None,
                      anchor_begin_pos: Optional[float] = None,
                      anchor_end_pos: Optional[float] = None,
                      times: Optional[List[float]] = None
                      ) -> Tuple[List[Tuple[float, float]], List[StrokeInfo]]:
    """
    Post-pass:
      1. segment the ACTIVE partner's per-frame tracks into strokes
         (StrokeStateMachine: bf-signed rule 1, commit at the real
          extreme, defender RE-LABELS a mislabelled stroke; then
          clean_commits stitches consecutive same-direction commits
          together at the time midpoint);
      2. compute per-frame pos (0..100) inside each stroke per `motion`;
      3. optionally rescale so the LARGEST stroke spans 0..100
         (normalize == 'global-max').
    cal_f / cal_b: user f/b marks - forwarded to the state machine
    (reference for the direction defender; not required).
    times: REAL presentation time (seconds) aligned 1:1 with raw_tracks,
    as produced by tracking_stream (see core.tracking.seek).  When passed,
    the returned (t, pos) pairs use THESE times (POS_MSEC based, NOT
    frame_index/fps) and the None slots (ghost / hold-over frames that
    tracking_stream stores as placeholders) are DROPPED from the result,
    so they reach neither RDP nor OpenFunScripter.  When `times` is None
    (legacy callers / tests), the old behavior is preserved: every slot
    gets a point, and `t = start_s + i / fps`.
    Returns (points [(time_s, pos)], strokes).
    """
    n = len(raw_tracks)
    last_pt: Optional[Tuple[float, float]] = None
    dense: List[Tuple[float, float]] = []
    valid_idx: List[int] = []
    for i, t in enumerate(raw_tracks):
        if t is not None:
            last_pt = t
            valid_idx.append(i)
        dense.append(last_pt if last_pt is not None else (0.0, 0.0))
    if not valid_idx:
        return [], []

    A = np.asarray(dense, dtype=np.float64)
    # Per-valid-point REAL times (1:1 with P = A[valid_idx]) fed to the
    # state machine as cur_t so each commit carries its real star_t (used by
    # clean_commits for a true TIME midpoint).  When times is None (legacy
    # callers / tests) synthesize start_s + frame_index/fps for every slot.
    if times is None:
        raise ValueError(
            "compute_positions: `times` is REQUIRED (REAL presentation "
            "time per raw_track slot, aligned 1:1 with raw_tracks; see "
            "core.tracking.seek / core.tracking.streaming).")
    if len(times) != n:
        raise ValueError(
            f"compute_positions: times has {len(times)} entries but "
            f"raw_tracks has {n} slots (they must be aligned 1:1).")
    valid_times = [float(times[i]) for i in valid_idx]
    strokes = [s for s in segment_strokes(A, valid_idx, anchor_axis,
                                          times=valid_times,
                                          cal_f=cal_f, cal_b=cal_b)
               if s.frame_end > s.frame_start]
    pos = np.zeros(n, dtype=np.float64)
    amplitudes = []    # per-stroke total path length (for logging)

    if normalize == "per-stroke":
        # Legacy behaviour: each stroke independently spans 0..100:
        #   advance -> pos 100..0  (towards f)
        #   retract -> pos  0..100 (towards b)
        for s in strokes:
            P_stroke = A[s.frame_start:s.frame_end + 1]
            frac, amp = stroke_progress(s, P_stroke, motion)
            amplitudes.append(amp)
            if s.direction == "advance":
                frac = 1.0 - frac
            for j in range(len(P_stroke)):
                pos[s.frame_start + j] = frac[j] * 100.0
    else:
        # global-max - user's algorithm: build the physical displacement
        # graph G (advance = negative, retract = positive), then normalize
        # by its GLOBAL span: Gmin -> 0, Gmax -> 100.
        # The biggest stroke defines the 0..100 range; smaller strokes are
        # scaled proportionally to their physical amplitude (complex: raw
        # path length; straight/arc/parabola: axial/conic arc length).
        #
        # User's example (advance 120, retract 80, advance 50, retract 100):
        #   G: 0 -> -120 -> -40 -> -90 -> +10
        #   Gmin = -120, Gmax = +10, span = 130
        #   normalized: -120 -> 0, -90 -> (90/130)*100 = 69.2,
        #               -40 -> (40/130)*100 = 30.8,  +10 -> 100
        running = 0.0
        for s in strokes:
            P_stroke = A[s.frame_start:s.frame_end + 1]
            frac, amp = stroke_progress(s, P_stroke, motion)
            amplitudes.append(amp)
            sign = 1.0 if s.direction == "retract" else -1.0
            for j in range(len(P_stroke)):
                pos[s.frame_start + j] = running + sign * float(frac[j]) * amp
            running = running + sign * amp
        gmin = float(pos.min())
        gmax = float(pos.max())
        span_n = gmax - gmin
        if span_n > 1e-6:
            pos = 100.0 * (pos - gmin) / span_n
        else:
            # degenerate (single stroke with amp=0): map to 50 for downstream
            # clip / RDP / keyframe conversion to stay well-defined.
            pos = np.full_like(pos, 50.0)

    # fill frames outside any stroke (gaps / leads) with the last in-stroke value
    if strokes:
        first_start = strokes[0].frame_start
        last_end = strokes[-1].frame_end
        last_val = 0.0
        for i in range(n):
            if i < first_start or i > last_end:
                pos[i] = last_val
            else:
                last_val = pos[i]

    # ── boundary anchors (requirement #3) ──
    # If the OFS timeline already has points before / after this segment
    # (main.lua dumped them and forwarded the anchor pos values), rescale the
    # boundary stroke(s) so the new track lands EXACTLY on those anchor values.
    # Runs AFTER the normal per-stroke / global-max normalization (which
    # produced the raw 0..100 scaling above) and BEFORE the final clip.
    if anchor_begin_pos is not None or anchor_end_pos is not None:
        pos = _apply_boundary_anchors(pos, strokes, normalize,
                                      anchor_begin_pos, anchor_end_pos)
        # the boundary rescale can leave gap/lead frames at stale values -
        # re-fill outside-stroke frames from the (rescaled) boundary strokes.
        if strokes:
            first_start = strokes[0].frame_start
            last_end = strokes[-1].frame_end
            last_val = pos[0]
            for i in range(n):
                if i < first_start or i > last_end:
                    pos[i] = last_val
                else:
                    last_val = pos[i]
    else:
        # No pre-denoise: the raw per-frame 0..100 scaling is kept as-is, so the
        # sharp stroke reversals stay visible on the RDP graph for manual
        # verification.  Stroke extreme endpoints are ALREADY exactly 0 or 100 by
        # construction of stroke_progress (frac[-1] == 1.0 in every motion mode:
        # complex = cum/total, straight = u_rel/span, arc/parabola = out/total),
        # so no endpoint re-pinning is needed.  All smoothing happens later, in the
        # interactive RDP window (user-tuned epsilon).
        pos = np.clip(pos, 0.0, 100.0)

    # human-readable summary (both modes)
    if amplitudes:
        amp_s = " ".join(f"{a:g}" for a in amplitudes)
        dir_s = " ".join(s.direction[0].upper() for s in strokes)
        rng = f"range=[{float(pos.min()):.1f}..{float(pos.max()):.1f}]"
        print(f"[pos] normalize={normalize} motion={motion} "
              f"strokes={len(strokes)} dirs=[{dir_s}] amps=[{amp_s}] {rng}")

    # ── final (t, pos) list ──────────────────────────────────────────────
    # Use the REAL presentation times (POS_MSEC based, from
    # core.tracking.seek) and DROP the None slots (ghost / hold-over
    # frames) so they reach neither RDP nor OpenFunScripter.
    # `times` is REQUIRED (validated above: not None, aligned 1:1
    # with raw_tracks) - there is no legacy i/fps fallback.
    kept = 0
    points = []
    for i in range(n):
        if raw_tracks[i] is None:
            continue  # ghost / hold-over slot -> excluded from the result
        points.append((round(float(times[i]), 4), float(pos[i])))
        kept += 1
    skipped = n - kept
    if skipped:
        print(f"[pos] dropped {skipped} ghost/hold-over slot(s) "
              f"from the result (kept {kept})")
    return points, strokes