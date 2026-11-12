"""Ramer–Douglas–Peucker approximation (pure numpy, testable without GUI)."""
import numpy as np


def rdp(points, epsilon: float):
    """
    Vectorized (numpy).
    points: (N, 2) - (t_sec, pos). epsilon - in pos units (%).
    Returns a subarray of points (kept points only).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts
    mask = np.zeros(n, dtype=bool)
    mask[0] = mask[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, b = pts[s], pts[e]
        ab = b - a
        L = np.hypot(ab[0], ab[1])
        if L < 1e-12:
            d = np.abs(pts[s:e + 1, 1] - a[1])
        else:
            d = np.abs(ab[0] * (pts[s:e + 1, 1] - a[1])
                       - ab[1] * (pts[s:e + 1, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] > epsilon:
            mask[s + i] = True
            stack.append((s, s + i))
            stack.append((s + i, e))
    return pts[mask]
