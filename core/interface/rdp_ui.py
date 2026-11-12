"""
Interactive range-select + RDP window (one window, three sliders at the bottom):

  Plotted:
    gray line       -- selected frame range, pos rescaled locally to 0..100
    red line + dots -- RDP approximation (visible only when eps > 0)

  Sliders (bottom, full X-axis width, in order top-to-bottom):
    Start at Frame   0..stop-1       (trim from the left, step 1)
    Stop at Frame    start+1..n-1    (trim from the right, step 1)
    Epsilon          0.0..0.2 float  (default 0.0 = off; no red line when 0;
                                      step 0.001)

  Mouse:
    wheel over PLOT   -> zoom in/out (x1.25 steps, 1x..64x), cursor-anchored
                         with 10% edge protection
    wheel over SLIDER -> +/- 1 for that slider
    LMB press+drag PLOT   -> horizontal pan (delta px -> track-idx units)
    LMB press+drag SLIDER -> adjust that slider's value
    plain MOUSEMOVE never changes anything

  Keyboard:
    A / a  -> start -= step     (step = max(1, n//100))
    Z / z  -> start += step
    S / s  -> stop  += step
    X / x  -> stop  -= step
    + / =  -> eps  += 0.001
    - / _  -> eps  -= 0.001
    Left / Right arrows -> pan view
    R      -> reset zoom to full [start..stop]
    Home   -> reset ALL three sliders to defaults (start=0, stop=n-1, eps=0)
    Enter / Space / q    -> OK
    Esc  -> keep all original points (full track, unscaled)

  Return values:
    ("ok",         [(t_sec, pos_scaled), ...])   Enter -- selected range,
                                                  pos 0..100 rescaled, RDP if eps>0
    ("all",        [(t_sec, pos_original), ...])  Esc   -- FULL original track, unscaled
    ("closed_ext", [(t_sec, pos_original), ...])  user closed the window (X button)
    ("all",        None)                         n < 10 -- window not shown
    _run (funscript_gen.py) treats "all" AND "closed_ext" identically
    (the full unscaled track), but logs each mode separately so the
    stderr log distinguishes a real "Esc/short-track" from a user closing
    the RDP window with X.

  RDP core: `simplification.cutil.simplify_coords_idx` (Cython, returns
  INDICES of kept points).

  Local rescale:
    p_lo = pts[start..stop, 1].min();  p_hi = pts[start..stop, 1].max()
    pos_scaled = 100.0 * (pos - p_lo) / (p_hi - p_lo)   (50.0 if flat)
    This is the canonical value written to the funscript (0..100).

  Drag semantics (all 3 sliders): LBUTTONDOWN inside the band starts a drag;
  MOUSEMOVE updates only while LMB stays down AND cursor is still in the band;
  LBUTTONUP finalises.  A plain MOUSEMOVE with no button never changes anything.
"""
import cv2
import numpy as np

from simplification.cutil import simplify_coords_idx

from core.interface.preview import center_window
from core.utilities import format_time

RDP_W, RDP_H = 1280, 720
ML, MR = 80, 40                    # left/right margins (plot area)
MT = 60                            # top margin (status text)
MB = 210                           # bottom: time labels + 3 sliders + hint

PLOT_X0, PLOT_X1 = ML, RDP_W - MR
PLOT_Y0, PLOT_Y1 = MT, RDP_H - MB
PLOT_W, PLOT_H = PLOT_X1 - PLOT_X0, PLOT_Y1 - PLOT_Y0

# slider y-centres (top to bottom: start, stop, epsilon)
SLIDER_START_Y = RDP_H - 165
SLIDER_STOP_Y  = RDP_H - 125
SLIDER_EPS_Y   = RDP_H - 85
SLIDER_HIT_PAD = 18                # hit-detection padding above/below line
SLIDER_END_PAD = 20                # horizontal hit-zone extension

EPS_MIN, EPS_MAX = 0.0, 0.2
EPS_DEFAULT      = 0.0             # default: off (no red line)
EPS_STEP         = 0.001           # wheel / +/- keyboard nudge step
START_DEFAULT    = 0

ZOOM_STEP  = 1.25
EDGE_FRAC  = 0.10                  # 10% edge protection for zoom cursors
MAX_PLOT_POINTS = 9000             # plotting cap (per visible range)


def interactive_rdp(points):
    """Combined range-select + RDP window (3 sliders).

    Returns (mode, approx):
      ("ok",  list[(t_sec, pos_scaled)])   Enter -- selected, 0..100 rescaled
      ("all", list[(t_sec, pos_original)]) Esc   -- full track, unscaled
      ("all", None)                        n < 10
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 10:
        print(f"[rdp] track too short ({n} < 10 frames), passing through")
        return ("all", None)

    step = max(1, n // 100)
    state = {
        "start": START_DEFAULT, "stop": n - 1, "eps": EPS_DEFAULT,
        "v_lo": 0.0, "v_hi": float(n - 1),
        "drag": None, "pan_px": None,
        "_key": None, "_p_lo": 0.0, "_p_hi": 1.0, "_flat": False,
    }

    def clamp_start(v):
        return int(max(0, min(state["stop"] - 1, int(round(float(v))))))

    def clamp_stop(v):
        return int(max(state["start"] + 1, min(n - 1, int(round(float(v))))))

    def clamp_eps(v):
        # Float epsilon in [EPS_MIN, EPS_MAX]; stepped to EPS_STEP (3 decimals).
        return float(max(EPS_MIN, min(EPS_MAX, round(float(v), 3))))

    def view_span():
        return max(1.0, state["v_hi"] - state["v_lo"])

    def full_span():
        return max(1, state["stop"] - state["start"])

    def zoom_ratio():
        return full_span() / max(1e-9, view_span())

    def reset_view():
        state["v_lo"] = float(state["start"])
        state["v_hi"] = float(state["stop"])

    def do_pan(delta_idx):
        span = view_span()
        lo, hi = float(state["start"]), float(state["stop"])
        v_lo = state["v_lo"] + delta_idx
        v_hi = v_lo + span
        if v_lo < lo:
            v_lo, v_hi = lo, lo + span
        if v_hi > hi:
            v_hi, v_lo = hi, hi - span
        state["v_lo"] = max(lo, v_lo)
        state["v_hi"] = min(hi, max(state["v_lo"] + span, state["v_lo"]))

    def do_zoom(direction, cursor_px):
        span = view_span()
        full = float(full_span())
        new_span = (span / ZOOM_STEP) if direction > 0 else (span * ZOOM_STEP)
        new_span = max(1.0, min(full, new_span))
        if abs(new_span - span) < 1e-9:
            return
        cf = (cursor_px - PLOT_X0) / max(1, PLOT_W)
        cf = max(EDGE_FRAC, min(1.0 - EDGE_FRAC, cf))
        anchor = state["v_lo"] + cf * span
        v_lo = anchor - cf * new_span
        v_hi = v_lo + new_span
        lo, hi = float(state["start"]), float(state["stop"])
        if v_lo < lo:
            v_lo, v_hi = lo, lo + new_span
        if v_hi > hi:
            v_hi, v_lo = hi, hi - new_span
        state["v_lo"] = max(lo, v_lo)
        state["v_hi"] = min(hi, max(state["v_lo"] + new_span, state["v_lo"]))

    def to_px(track_idx, pos_scaled):
        vl, vh = state["v_lo"], state["v_hi"]
        x = PLOT_X0 + ((track_idx - vl) / max(1e-6, vh - vl)) * PLOT_W
        y = PLOT_Y0 + (1.0 - pos_scaled / 100.0) * PLOT_H
        return int(x), int(y)

    def recompute_range():
        key = (state["start"], state["stop"])
        if key != state["_key"]:
            col = pts[state["start"]:state["stop"] + 1, 1]
            state["_p_lo"]  = float(col.min()) if len(col) else 0.0
            state["_p_hi"]  = float(col.max()) if len(col) else 1.0
            state["_flat"]  = (state["_p_hi"] - state["_p_lo"]) < 1e-6
            state["_key"]   = key
            reset_view()

    def slider_zone(x, y):
        if ((SLIDER_START_Y - SLIDER_HIT_PAD) <= y <= (SLIDER_START_Y + SLIDER_HIT_PAD)
                and (PLOT_X0 - SLIDER_END_PAD) <= x <= (PLOT_X1 + SLIDER_END_PAD)):
            return "start"
        if ((SLIDER_STOP_Y - SLIDER_HIT_PAD) <= y <= (SLIDER_STOP_Y + SLIDER_HIT_PAD)
                and (PLOT_X0 - SLIDER_END_PAD) <= x <= (PLOT_X1 + SLIDER_END_PAD)):
            return "stop"
        if ((SLIDER_EPS_Y - SLIDER_HIT_PAD) <= y <= (SLIDER_EPS_Y + SLIDER_HIT_PAD)
                and (PLOT_X0 - SLIDER_END_PAD) <= x <= (PLOT_X1 + SLIDER_END_PAD)):
            return "eps"
        if PLOT_X0 <= x <= PLOT_X1 and PLOT_Y0 <= y <= PLOT_Y1:
            return "plot"
        return None

    def frac_to_val(px_x, vmin, vmax):
        f = max(0.0, min(1.0, (px_x - PLOT_X0) / max(1, PLOT_W)))
        return int(round(f * (vmax - vmin) + vmin))

    def frac_to_eps(px_x):
        # Float epsilon from cursor x (0..1 -> EPS_MIN..EPS_MAX); the value is
        # clamped/stepped to EPS_STEP by the caller's clamp_eps.
        f = max(0.0, min(1.0, (px_x - PLOT_X0) / max(1, PLOT_W)))
        return float(f * (EPS_MAX - EPS_MIN) + EPS_MIN)

    def on_mouse(event, x, y, flags, _param):
        lmb  = bool(flags & cv2.EVENT_LBUTTONDOWN)
        zone = slider_zone(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if zone == "start":
                state["drag"] = "start"
                state["start"] = clamp_start(frac_to_val(x, 0, state["stop"] - 1))
            elif zone == "stop":
                state["drag"] = "stop"
                state["stop"] = clamp_stop(frac_to_val(x, state["start"] + 1, n - 1))
            elif zone == "eps":
                state["drag"] = "eps"
                state["eps"] = clamp_eps(frac_to_eps(x))
            elif zone == "plot":
                state["drag"] = "pan"
                state["pan_px"] = x

        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None
            state["pan_px"] = None

        elif event == cv2.EVENT_MOUSEMOVE:
            z = state["drag"]
            if z is not None and not lmb:
                state["drag"] = None
            elif z == "start" and zone == "start" and lmb:
                state["start"] = clamp_start(frac_to_val(x, 0, state["stop"] - 1))
            elif z == "stop" and zone == "stop" and lmb:
                state["stop"] = clamp_stop(frac_to_val(x, state["start"] + 1, n - 1))
            elif z == "eps" and zone == "eps" and lmb:
                state["eps"] = clamp_eps(frac_to_eps(x))
            elif z == "pan" and zone == "plot" and lmb and state["pan_px"] is not None:
                dx = x - state["pan_px"]
                state["pan_px"] = x
                do_pan(-dx * (view_span() / max(1, PLOT_W)))

        elif event == cv2.EVENT_MOUSEWHEEL:
            d = 1 if flags > 0 else -1
            if zone == "plot":
                do_zoom(d, x)
            elif zone == "start":
                state["start"] = clamp_start(state["start"] + d)
            elif zone == "stop":
                state["stop"] = clamp_stop(state["stop"] + d)
            elif zone == "eps":
                state["eps"] = clamp_eps(state["eps"] + EPS_STEP * d)

    # -- window setup ----------------------------------------------------------
    win = "Range + RDP smoothing"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if hasattr(cv2, "WND_KEEP_ASPECT_RATIO"):
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WND_KEEP_ASPECT_RATIO)
    cv2.resizeWindow(win, RDP_W, RDP_H)
    center_window(win, RDP_W, RDP_H)
    cv2.setMouseCallback(win, on_mouse)

    print(f"[rdp] {n} track frames.  Sliders: start/stop/eps  "
          f"Wheel(plot)=zoom  Drag(plot)=pan  A/Z=S/X=nudge  R=reset zoom  "
          f"Home=reset all  Enter=OK  Esc=keep originals")

    # Drain stale Win32 messages: the streaming preview window was destroyed
    # in core/tracking/streaming.py (cv2.destroyAllWindows) right before we
    # opened this window.  On Win32 / cv2>=5 the first few cv2.waitKey() calls
    # in a FRESH window still return -1 ("no key pressed during the 30 ms")
    # as the OS message queue catches up - empirically 3-4-6 consecutive -1s
    # (verified under cv2 5.0.0: waitKey sequence [-1]*6 over 200 ms while
    # getWindowProperty(VISIBLE)=1.0 throughout).  A lone `-1` therefore does
    # NOT mean "window closed".  The real signal is
    # WND_PROP_VISIBLE: 1.0 live (even minimized), 0.0 after destroyWindow -
    # the branch below uses that.
    # This drain also keeps the very-first loop iteration from racing the
    # previous window's destroy in ways that would confuse the visible-check
    # under some backends.
    for _ in range(10):
        cv2.waitKey(1)

    last_rdp_key = None
    rdp_indices  = None

    def _draw_slider(canvas, sy, label, val, vmin, vmax):
        cv2.line(canvas, (PLOT_X0, sy), (PLOT_X1, sy), (200, 200, 200), 4)
        f = max(0.0, min(1.0, (val - vmin) / max(1e-9, vmax - vmin)))
        thx = int(PLOT_X0 + f * PLOT_W)
        cv2.circle(canvas, (thx, sy), 11, (255, 140, 0), -1)
        cv2.circle(canvas, (thx, sy), 4,  (0, 0, 0),     -1)
        cv2.putText(canvas, f"{label}: {val}", (PLOT_X0, sy - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (60, 60, 60), 2)

    while True:
        recompute_range()
        st, sp, ep = state["start"], state["stop"], state["eps"]

        sl       = pts[st:sp + 1]
        t_data   = sl[:, 0]
        p_lo, p_hi = state["_p_lo"], state["_p_hi"]
        if state["_flat"]:
            scaled_pos = np.full(len(sl), 50.0, dtype=np.float64)
        else:
            scaled_pos = 100.0 * (sl[:, 1] - p_lo) / (p_hi - p_lo)

        # -- RDP (only when eps > 0) ------------------------------------------
        if ep > 0:
            rdp_key = (st, sp, ep)
            if rdp_key != last_rdp_key:
                arr = np.column_stack([t_data, scaled_pos])
                try:
                    rdp_indices = np.asarray(
                        simplify_coords_idx(
                            np.ascontiguousarray(arr), float(ep)),
                        dtype=np.intp)
                except Exception:
                    print("[rdp] simplify_coords_idx failed, keeping all in range")
                    rdp_indices = np.arange(len(sl), dtype=np.intp)
                last_rdp_key = rdp_key
        else:
            last_rdp_key = None
            rdp_indices  = None

        # -- canvas ------------------------------------------------------------
        canvas = np.full((RDP_H, RDP_W, 3), 248, np.uint8)

        # Y grid (scaled 0..100)
        for g in range(0, 101, 20):
            _, gy = to_px(0, float(g))
            gy = max(PLOT_Y0, min(PLOT_Y1, int(gy)))
            cv2.line(canvas, (PLOT_X0, gy), (PLOT_X1, gy), (220, 220, 220), 1)
            cv2.putText(canvas, str(g), (ML - 52, gy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 2)

        # X ticks: absolute video times within the current view
        vi_lo = max(0, int(state["v_lo"]))
        vi_hi = min(n - 1, int(state["v_hi"]))
        if vi_hi > vi_lo:
            t_vlo  = float(pts[vi_lo, 0])
            t_vhi  = float(pts[vi_hi, 0])
            t_span = max(1e-6, t_vhi - t_vlo)
            tstep  = max(1, int(t_span // 8))
            t0t    = int(t_vlo // tstep) * tstep
            g = 0
            while t0t + g * tstep <= t_vhi + tstep:
                t_mark = t0t + g * tstep
                if t_mark >= t_vlo - tstep and t_mark <= t_vhi + tstep:
                    t_frac = (t_mark - t_vlo) / t_span
                    idx_at = vi_lo + (vi_hi - vi_lo) * t_frac
                    gx, _ = to_px(float(idx_at), 50.0)
                    gx = max(PLOT_X0, min(PLOT_X1, int(gx)))
                    cv2.line(canvas, (gx, PLOT_Y0), (gx, PLOT_Y1), (238, 238, 238), 1)
                    cv2.putText(canvas, format_time(t_mark), (gx - 22, PLOT_Y1 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
                g += 1

        # Axes
        cv2.line(canvas, (PLOT_X0, PLOT_Y1), (PLOT_X1, PLOT_Y1), (150, 150, 150), 2)
        cv2.line(canvas, (PLOT_X0, PLOT_Y0), (PLOT_X0, PLOT_Y1), (150, 150, 150), 2)

        # Gray original line (visible portion, subsampled)
        vis_lo = max(0, int(state["v_lo"]) - st)
        vis_hi = min(len(sl) - 1, int(state["v_hi"]) - st)
        if vis_hi - vis_lo >= 2:
            count = vis_hi - vis_lo + 1
            if count > MAX_PLOT_POINTS:
                idx_plot = np.linspace(0, count - 1, MAX_PLOT_POINTS).astype(int)
            else:
                idx_plot = np.arange(count)
            px_list = [to_px(float(st + vis_lo + i), float(scaled_pos[vis_lo + i]))
                       for i in idx_plot]
            cv2.polylines(canvas, [np.array(px_list, np.int32).reshape(-1, 1, 2)],
                          False, (120, 120, 120), 2)

        # Red RDP line + green dots (only when eps > 0)
        if ep > 0 and rdp_indices is not None and len(rdp_indices) >= 2:
            rdp_px = [to_px(float(st + int(i)), float(scaled_pos[int(i)]))
                      for i in rdp_indices]
            rdp_arr = np.array(rdp_px, np.int32)
            cv2.polylines(canvas, [rdp_arr.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
            for (px, py) in rdp_arr:
                cv2.circle(canvas, (int(px), int(py)), 5, (0, 160, 0), -1)
                cv2.circle(canvas, (int(px), int(py)), 2, (0, 0, 0), -1)

        # -- three sliders -----------------------------------------------------
        _draw_slider(canvas, SLIDER_START_Y, "Start at Frame", st, 0, max(1, sp - 1))
        _draw_slider(canvas, SLIDER_STOP_Y,  "Stop at Frame",  sp, st + 1, n - 1)
        _draw_slider(canvas, SLIDER_EPS_Y,   "Epsilon",         ep, EPS_MIN, EPS_MAX)

        # -- status bar --------------------------------------------------------
        n_sel  = sp - st + 1
        zoom_x = zoom_ratio()
        if ep == 0:
            rdp_lbl = "off"
        elif rdp_indices is not None:
            rdp_lbl = f"{len(rdp_indices)} pts"
        else:
            rdp_lbl = "-"
        if state["_flat"]:
            stat = (f"frames {st}..{sp} ({n_sel}/{n})  pos: flat [{p_lo:.1f}]  "
                    f"eps={ep:.3f}  RDP={rdp_lbl}  zoom={zoom_x:.1f}x")
        else:
            stat = (f"frames {st}..{sp} ({n_sel}/{n})  "
                    f"pos [{p_lo:.1f},{p_hi:.1f}]->[0,100]  "
                    f"eps={ep:.3f}  RDP={rdp_lbl}  zoom={zoom_x:.1f}x")
        cv2.putText(canvas, stat, (ML, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2)
        cv2.putText(canvas, "pos (scaled 0..100)", (ML - 70, PLOT_Y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)

        # -- hint (bottom) -----------------------------------------------------
        hint = (f"wheel=zoom/step  drag=pan/adjust  "
                f"A/Z=start-/+{step}  S/X=stop-/+{step}  +/-=eps  "
                f"R=reset zoom  Home=reset all  Enter=OK  Esc=keep originals")
        cv2.putText(canvas, hint, (ML, RDP_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)

        cv2.imshow(win, canvas)
        raw_key = cv2.waitKey(30)

        # window closed externally (X / OS / Alt-F4):
        # THE AUTHORITATIVE signal is WND_PROP_VISIBLE == 0 (destroyed).
        # Do NOT rely on cv2.waitKey() == -1: on Win32 + cv2>=5 a freshly
        # created window still reports -1 for several dozen ms of "no key"
        # right after the previous OpenCV window was destroyed (see the
        # drain + probe comment above; the original sot-csrt RDP-skip bug
        # was exactly this mis-read).  VISIBLE stays 1.0 for a live window,
        # even minimized, and flips to 0.0 on DestroyWindow - a clean
        # backend-independent distinction.
        try:
            _vis = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
            _closed = (_vis < 0.5)
        except Exception:
            _closed = (raw_key == -1)   # conservative fallback (rare)
        if _closed:
            cv2.destroyAllWindows()
            return ("closed_ext", list(map(tuple, pts)))

        key = raw_key & 0xFFFF
        if key in (13, 108, ord('q')):
            # Enter / Space / q = OK  -> selected range, scaled, RDP if eps > 0
            if ep > 0 and rdp_indices is not None and len(rdp_indices) > 0:
                result = [(float(t_data[i]), float(scaled_pos[i])) for i in rdp_indices]
            else:
                result = [(float(t_data[i]), float(scaled_pos[i]))
                          for i in range(len(sl))]
            cv2.destroyAllWindows()
            print(f"[rdp] OK: {len(result)} frames [{st}..{sp}], "
                  f"pos [{p_lo:.1f},{p_hi:.1f}]->[0,100], eps={ep:.3f}")
            return ("ok", result)
        elif key == 27:
            # Esc = keep ALL original points (full track, unscaled, no RDP)
            cv2.destroyAllWindows()
            print(f"[rdp] Esc: keeping all {n} original points (unscaled)")
            return ("all", list(map(tuple, pts)))
        elif key == ord('r'):
            reset_view()
        elif key == 36:          # Home
            state["start"] = START_DEFAULT
            state["stop"]  = n - 1
            state["eps"]   = EPS_DEFAULT
            reset_view()
        elif key == 37:          # Left arrow (VK_LEFT = 0x25 = 37)
            do_pan(-view_span() * 0.10)
        elif key == 39:          # Right arrow (VK_RIGHT = 0x27 = 39)
            do_pan(view_span() * 0.10)
        elif key in (ord('A'), ord('a')):
            state["start"] = clamp_start(state["start"] - step)
        elif key in (ord('Z'), ord('z')):
            state["start"] = clamp_start(state["start"] + step)
        elif key in (ord('S'), ord('s')):
            state["stop"] = clamp_stop(state["stop"] + step)
        elif key in (ord('X'), ord('x')):
            state["stop"] = clamp_stop(state["stop"] - step)
        elif key in (43, 107, ord('+'), ord('=')):
            state["eps"] = clamp_eps(state["eps"] + EPS_STEP)
        elif key in (45, 109, ord('-'), ord('_')):
            state["eps"] = clamp_eps(state["eps"] - EPS_STEP)
