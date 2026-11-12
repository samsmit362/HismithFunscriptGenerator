"""
Interactive f/b calibration window for the ACTIVE partner's point p.

The tracked point comes from the selected tracker (pose / dino / SOT) —
the window only consumes whatever `tracker.detect_all(frame)` returns.
"""
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detectors.person_tracker import PersonTracker
from core.interface.identify import _drag_for, _drag_reset
from core.interface.preview import PREVIEW_H, PREVIEW_W, draw_skeleton, fit, setup_window
from core.interface.win32 import KEY_LEFT, KEY_RIGHT, key_held
from core.matching import box_iou, match_active
from core.tracking.seek import seek_to_time_exact
from core.types import Person, ThrustCalibration


def calibrate_thrust(video_path: str, start_s: float, fps: float,
                     tracker: PersonTracker,
                     active: Optional[Person],
                     auto: bool = False,
                     passive: Optional[Person] = None) -> ThrustCalibration:
    """
    Interactive f/b calibration window for the ACTIVE partner's point p.

    Keys:
        <- / ->     - step frames (shift = larger step)
        f / F       - mark FULL INSERTION (pos 0)
        b / B       - mark FULL RETRACTION (pos 100)
        m / M       - optional 50% arc mark (arc/parabola fit reference)
        Enter/Space - confirm (requires f and b marked)
        Q / Esc     - abort
        hold S      - overlay skeletons of ACTIVE (yellow) and PASSIVE (orange)

    Mouse (LMB drag):
        press       - candidate b (anchor)
        hold+move   - live line drawn from anchor to cursor (b -> f)
        release     - candidate f is fixed at the release position
        a click without movement is a NO-OP - it does NOT create or
        overwrite existing f/b marks.

    Collision rule (keys AND mouse): if the mark being assigned lands on
    exactly the same pixel as the OTHER mark (within a few px), the OTHER
    mark is cleared - "f == b" is not a calibration, so only one endpoint
    survives. Enter/Space then still requires both marks to be present,
    confirming the invariant that the two endpoints are distinct.

    `passive`, when provided, is drawn in the skeleton overlay next to the
    active one so you can visually see which person each skeleton belongs to.
    """
    cap = cv2.VideoCapture(video_path)
    # Reliable start seek (POS_MSEC based, see core.tracking.seek):
    # CAP_PROP_POS_FRAMES alone is unreliable on variable-frame-rate videos.
    # seek_to_time_exact ALSO reads the first frame and returns it together
    # with its REAL presentation time.
    frame0, _t0, _seek_ok = seek_to_time_exact(cap, start_s, fps=fps)
    if not _seek_ok:
        # fall back to the legacy seek; cap.read() below then produces the
        # first frame for us.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))

    # shared state for per-frame matching (last known point + box from the
    # identify stage) - so match_active picks "the person closest to where
    # we tracked last" instead of whichever partner the tracker currently
    # scores higher.
    state = {"last_point": (tuple(active.point) if active else None,
                            active.box if active else None)}
    # per-frame result cache: the interactive loop calls detect_all once
    # per frame and may need the full detection list again for the S-hold
    # passive overlay (avoids running the detector twice for the same frame).
    _cache: List[Person] = []
    # frame index the tracker was last fed (None = not yet detected)
    detected_frame: Optional[int] = None

    def detect_persons(frame: np.ndarray) -> List[Person]:
        # The interactive loop RE-RENDERS the same frame every ~30 ms while
        # waiting for a key, but the tracker is STATEFUL (e.g. SOT CSRT/KCF:
        # every call = a real tracker.update() step that updates the
        # correlation filter and the EMA size state).  Feeding one frame
        # 20-30 times per second makes the tracked point jitter on a static
        # frame and pollutes the tracker state, while tracking_stream feeds
        # it exactly once per NEW frame (cap.read() -> one update()).
        # Call the detector only when the frame index actually changes —
        # same contract as the video processing path.
        try:
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        except cv2.error:
            cur = None  # seek race - treat as new frame (safe: one extra update)
        nonlocal detected_frame
        if detected_frame is not None and cur is not None and cur == detected_frame:
            return list(_cache)
        detected_frame = cur
        _cache.clear()
        _cache.extend(tracker.detect_all(frame))
        return list(_cache)

    # Scene transform (SOT only): every cap.read() below is passed through
    # _scene_apply, so the interactive loop and the SOT tracker work in the
    # SAME coordinate system (scene.apply(raw)).  identity scene -> same
    # ndarray, zero extra work.
    _sc = getattr(tracker, "scene", None)

    def _scene_apply(fr: np.ndarray) -> np.ndarray:
        if _sc is not None and not _sc.is_identity(shape=fr.shape[:2]):
            return _sc.apply(fr)
        return fr

    if auto:
        # auto fallback: scan forward up to 6 s, use min/max projection onto the
        # dominant motion direction as the f / b pair
        cal = ThrustCalibration()
        pts: List[Tuple[float, float]] = []
        n = int(6.0 * fps)
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frame = _scene_apply(frame)
            persons = detect_persons(frame)
            if not persons:
                continue
            idx, _cost = match_active(persons,
                                      state["last_point"][0],
                                      state["last_point"][1])
            if idx < 0:
                idx = int(np.argmax([p.score for p in persons]))
            state["last_point"] = (persons[idx].point, persons[idx].box)
            pts.append(persons[idx].point)
        cap.release()
        if len(pts) < 8:
            # RuntimeError, not sys.exit(str) (= os._exit): lets main() log
            # the reason to the stderr log before exiting with code 1.
            raise RuntimeError("Not enough frames for auto-calibration. "
                               "Use the interactive mode.")
        P = np.asarray(pts, dtype=np.float64)
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c)
        d = vt[0]
        proj = (P - c) @ d
        cal.f = tuple(float(v) for v in P[np.argmin(proj)])
        cal.b = tuple(float(v) for v in P[np.argmax(proj)])
        print(f"[cal] AUTO axis={d}  f={cal.f}  b={cal.b}")
        return cal

    # interactive mode only: the window is created HERE (not at function
    # entry) so the auto path never leaves a dangling empty "Calibrate"
    # window on screen.
    win = ("Calibrate f / b  (<->=frame, f=0%, b=100%, m=50%, "
           "LMB drag: press=b release=f, Enter=confirm, Q=abort)")
    setup_window(win)

    # mouse state for the "draw a line = b -> f" interaction.
    # press = candidate b (anchor), hold+move = live line, release = candidate f.
    # a click without movement is a NO-OP (does not create or overwrite marks).
    drag = _drag_for(win).p
    _drag_reset(win)
    # "release within this many PX (frame coords) of the anchor -> click, no-op".
    # A real drag covers the thrust stroke so any real f-b distance is >> this;
    # 3 px is well below any intentional stroke and above any hand jitter.
    MIN_DRAG_FRAME_PX = 3.0

    cal = ThrustCalibration()
    # hold-over for interactive frame-stepping: when the only nearby
    # candidate is the passive partner (far from the last tracked point) or
    # the active partner is missing, keep the last accepted point instead of
    # snapping to the passive.  User steps the frame forward and the point
    # re-attaches automatically the moment the active partner reappears.
    calib_hold = {
        "last": (tuple(active.point) if active else None),
        "budget": 120.0,
        "running": 0,
    }

    def _calib_budget() -> float:
        return max(80.0, 1.4 * calib_hold["budget"])

    def active_person(persons: List[Person]) -> Tuple[int, float]:
        if not persons:
            return -1, 0.0
        idx, cost = match_active(persons, state["last_point"][0],
                                 state["last_point"][1])
        if idx < 0:
            idx = int(np.argmax([p.score for p in persons]))
            return idx, 0.0
        return idx, cost

    def tracked_point(frame: np.ndarray) -> Optional[Tuple[float, float]]:
        persons = detect_persons(frame)
        ai = active_person(persons)[0]
        if ai < 0:
            # active missing - hold the last point
            if calib_hold["last"] is not None:
                calib_hold["running"] += 1
            return calib_hold["last"]
        cand = persons[ai]
        if calib_hold["last"] is not None:
            d = float(np.hypot(cand.point[0] - calib_hold["last"][0],
                               cand.point[1] - calib_hold["last"][1]))
            if d > _calib_budget():
                # likely the passive partner - keep the last accepted point
                calib_hold["running"] += 1
                return calib_hold["last"]
        # accepted - clear any running hold
        calib_hold["last"] = cand.point
        calib_hold["running"] = 0
        state["last_point"] = (cand.point, cand.box)
        return cand.point

    if _seek_ok:
        # seek already decoded the first frame for us; use it.
        frame = frame0
        ret = True
    else:
        ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read a calibration frame.")
    frame = _scene_apply(frame)

    # "LMB drag = press b, release f" is STATEFUL - a release is only
    # meaningful as the transition down==True -> down==False.  The flag
    # below snapshots the PREVIOUS tick's state, so the release is seen
    # exactly once on the tick after UP.
    _drag_was_down = False

    def _assign_endpoint(me_key: str, other_key: str,
                         val: Tuple[float, float]) -> None:
        """Assign one of f/b and CLEAR the other if it lands within
        MIN_DRAG_FRAME_PX of the new value.

        The rule: f == b at the same pixel is not a calibration; only one
        endpoint survives. This is the semantic "last write wins + collision
        cleanup" the user asked for - pressing F where B already is makes B
        go away (and vice versa), so a stale mark can never hide the new one.
        """
        setattr(cal, me_key, val)
        other = getattr(cal, other_key, None)
        if other is not None:
            d = float(np.hypot(val[0] - other[0], val[1] - other[1]))
            if d < MIN_DRAG_FRAME_PX:
                setattr(cal, other_key, None)
                print(f"[cal] {me_key} == {other_key} (within "
                      f"{MIN_DRAG_FRAME_PX:.0f} px) -> cleared {other_key}")

    while True:
        view, scale, x0, y0 = fit(frame)

        # ── mouse: LMB drag = press b, release f, click (no movement) = no-op ──
        # Release detection: the down->not-down transition happens ONCE, the
        # tick AFTER the LBUTTONUP callback set st.down=False. We read
        # _drag_was_down (previous tick's state) to see it exactly once.
        if _drag_was_down and not drag.down and drag.anchor is not None:
            ax, ay = drag.anchor
            b_pt = ((ax - x0) / scale, (ay - y0) / scale)
            r_pt = ((drag.px - x0) / scale, (drag.py - y0) / scale)
            # clamp to the frame bounds (the event can leave the letterboxed
            # region on the edge of the window; frame coords must stay inside)
            fh, fw = frame.shape[:2]
            b_pt = (float(np.clip(b_pt[0], 0.0, float(fw))),
                    float(np.clip(b_pt[1], 0.0, float(fh))))
            r_pt = (float(np.clip(r_pt[0], 0.0, float(fw))),
                    float(np.clip(r_pt[1], 0.0, float(fh))))
            drag.anchor = None
            d_px = float(np.hypot(r_pt[0] - b_pt[0], r_pt[1] - b_pt[1]))
            if d_px < MIN_DRAG_FRAME_PX:
                # click without movement: NO-OP - do NOT create or overwrite
                # existing f/b marks (user's requirement #2).
                print(f"[cal] mouse click at "
                      f"({r_pt[0]:.0f},{r_pt[1]:.0f}) - no-op, f/b unchanged")
            else:
                # real drag - atomic: assign b and f (collision cleanup is a
                # no-op here because d_px >= threshold, but the helper keeps
                # the code symmetric with the keyboard path).
                _assign_endpoint("b", "f", b_pt)
                _assign_endpoint("f", "b", r_pt)
                print(f"[cal] line drag: b=100% @ ({b_pt[0]:.0f},{b_pt[1]:.0f})  "
                      f"f=0% @ ({r_pt[0]:.0f},{r_pt[1]:.0f})  len={d_px:.1f}px")
        _drag_was_down = drag.down

        # live line (button held): anchor = candidate b, cursor = candidate f
        if drag.down and drag.anchor is not None:
            cv2.line(view, (int(drag.anchor[0]), int(drag.anchor[1])),
                     (int(drag.px), int(drag.py)), (80, 250, 250), 2)
            cv2.circle(view, (int(drag.anchor[0]), int(drag.anchor[1])), 10,
                       (80, 250, 250), 1)
            cv2.circle(view, (int(drag.px), int(drag.py)), 10,
                       (80, 250, 250), 1)
            cv2.putText(view, "b (anchor)",
                        (int(drag.anchor[0]) - 64, int(drag.anchor[1]) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 250, 250), 2)
            cv2.putText(view, "f (release)",
                        (int(drag.px) + 14, int(drag.py) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 250, 250), 2)

        t = tracked_point(frame)
        if t is not None:
            cv2.circle(view, (int(t[0] * scale) + x0, int(t[1] * scale) + y0),
                       9, (0, 255, 0), -1)
        if cal.f:
            fx, fy = int(cal.f[0] * scale) + x0, int(cal.f[1] * scale) + y0
            cv2.circle(view, (fx, fy), 14, (0, 160, 255), 2)
            cv2.putText(view, "f = 0%", (fx - 36, fy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 2)
        if cal.b:
            bx, by = int(cal.b[0] * scale) + x0, int(cal.b[1] * scale) + y0
            cv2.circle(view, (bx, by), 14, (0, 255, 255), 2)
            cv2.putText(view, "b = 100%", (bx - 44, by - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if cal.f and cal.b:
            ax = np.asarray(cal.b) - np.asarray(cal.f)
            nrm = float(np.linalg.norm(ax))
            if nrm > 1e-6:
                ax = ax / nrm
                c = (np.asarray(cal.f) + np.asarray(cal.b)) / 2.0
                L = max(80.0, 0.5 * nrm)
                cv2.line(view,
                         (int((c[0] - ax[0] * L) * scale) + x0,
                          int((c[1] - ax[1] * L) * scale) + y0),
                         (int((c[0] + ax[0] * L) * scale) + x0,
                          int((c[1] + ax[1] * L) * scale) + y0),
                         (80, 80, 255), 2)
        status = f"f={'set' if cal.f else '-'}  b={'set' if cal.b else '-'}"
        if drag.down and drag.anchor is not None:
            status = "[LMB drag: b -> f]  " + status
        elif calib_hold["running"] > 0:
            status = f"[hold={calib_hold['running']}]  {status}"
        info = (f"< / > step  f=0%  b=100%  m=50%  "
                f"LMB drag: press=b, release=f  "
                f"Enter=confirm  Q=abort  "
                f"[hold S]=active+passive skeletons  {status}")
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(view, info, ((PREVIEW_W - tw) // 2, PREVIEW_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if key_held(ord('s')) or key_held(ord('S')):
            persons = list(_cache) if _cache else detect_persons(frame)
            if persons:
                # active: closest to where we last tracked (distance-dominant)
                ai, _cost = active_person(persons)
                draw_skeleton(view, scale, x0, y0, persons[ai],
                              color=(0, 220, 255), label="active p")
                # passive: best remaining candidate by box IoU with the chosen
                # passive person (excluding the one already drawn as active)
                if passive is not None and len(persons) > 1:
                    pi = -1
                    best_p = -1.0
                    for j, p in enumerate(persons):
                        if j == ai:
                            continue
                        v = box_iou(p.box, passive.box)
                        if v > best_p:
                            best_p, pi = v, j
                    if pi == -1:
                        pi = [j for j in range(len(persons)) if j != ai][0]
                    draw_skeleton(view, scale, x0, y0, persons[pi],
                                  color=(0, 140, 255), label="passive p")
        cv2.imshow(win, view)

        k = cv2.waitKeyEx(30)
        # Enter / Space = confirm (f + b required), Q / Esc = abort - same
        # semantics as the SOT ROI selection window (identify.py).
        if (k & 0xFF) in (13, 32):
            if not (cal.f and cal.b):
                print("[cal] f and b must both be marked before confirming (press Enter / Space)")
                continue
            break
        if k in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            print("[cal] aborted by user")
            sys.exit(2)
        if k in (ord('f'), ord('F')) and t:
            # last-write-wins: assigning f where b already is CLEARS b
            # (a f == b pair is not a calibration) - the new mark only.
            _assign_endpoint("f", "b", (float(t[0]), float(t[1])))
            print(f"[cal] f=0%   @ ({t[0]:.0f},{t[1]:.0f})")
        elif k in (ord('b'), ord('B')) and t:
            _assign_endpoint("b", "f", (float(t[0]), float(t[1])))
            print(f"[cal] b=100% @ ({t[0]:.0f},{t[1]:.0f})")
        elif k in (ord('m'), ord('M')) and t:
            cal.m = (float(t[0]), float(t[1]))
            print(f"[cal] m=50%  @ ({t[0]:.0f},{t[1]:.0f})")
        elif k in (KEY_LEFT, KEY_RIGHT):
            mods = (cv2.waitKeyEx(1) >> 16) & 0xFF
            step = 10 if mods in (0x1, 0x2, 0x4) else 1
            delta = step if k == KEY_RIGHT else -15
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + delta))
            ret, frame = cap.read()
            if not ret:
                break
            frame = _scene_apply(frame)

    cap.release()
    cv2.destroyAllWindows()
    if not (cal.f and cal.b):
        print("[cal] aborted by user (f or b missing)")
        sys.exit(2)
    axis = cal.initial_axis
    print(f"[cal] f={cal.f}  b={cal.b}  axis={axis}")
    return cal