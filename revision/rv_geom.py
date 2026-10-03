import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(os.path.dirname(HERE), "code")
if ORIG not in sys.path:
    sys.path.insert(0, ORIG)

import planners as P0

FREE = 0
OBST = 1
TOL = 1e-9
TURN_THRESHOLDS = (13.0, 18.0, 23.0, 28.0)


def check_grid(grid):
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty 2D ndarray")
    if not np.isin(grid, (FREE, OBST)).all():
        raise ValueError("grid values must be 0 or 1")


def check_weights(grid, w):
    if not isinstance(w, np.ndarray) or w.shape != grid.shape:
        raise ValueError("weight map shape mismatch")
    if not np.isfinite(w).all() or (w < 1.0 - 1e-12).any():
        raise ValueError("weights must be finite and >= 1")


def check_point(grid, p, name):
    if len(p) != 2:
        raise ValueError(name + " must be a 2-tuple")
    r, c = int(p[0]), int(p[1])
    if r != p[0] or c != p[1]:
        raise ValueError(name + " must have integer coordinates")
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        raise ValueError(name + " out of bounds")
    if grid[r, c] != FREE:
        raise ValueError(name + " lies on an obstacle")


def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def seg_box_dist(ax, ay, bx, by, r, c):
    lo_r, hi_r, lo_c, hi_c = r - 0.5, r + 0.5, c - 0.5, c + 0.5
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    hit = True
    for p, q in ((-dx, ax - lo_r), (dx, hi_r - ax), (-dy, ay - lo_c), (dy, hi_c - ay)):
        if p == 0.0:
            if q < 0.0:
                hit = False
                break
        else:
            t = q / p
            if p < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                hit = False
                break
    if hit:
        return 0.0
    d = min(math.hypot(max(abs(ax - r) - 0.5, 0.0), max(abs(ay - c) - 0.5, 0.0)),
            math.hypot(max(abs(bx - r) - 0.5, 0.0), max(abs(by - c) - 0.5, 0.0)))
    for vx in (lo_r, hi_r):
        for vy in (lo_c, hi_c):
            d = min(d, _pt_seg(vx, vy, ax, ay, bx, by))
    return d


def seg_clear(grid, a, b, rho):
    R, C = grid.shape
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    if min(ax, bx) + 0.5 < rho - 1e-12 or R - 0.5 - max(ax, bx) < rho - 1e-12:
        return False
    if min(ay, by) + 0.5 < rho - 1e-12 or C - 0.5 - max(ay, by) < rho - 1e-12:
        return False
    k = int(math.ceil(rho + 0.5))
    r_lo = max(0, int(math.floor(min(ax, bx))) - k)
    r_hi = min(R - 1, int(math.ceil(max(ax, bx))) + k)
    c_lo = max(0, int(math.floor(min(ay, by))) - k)
    c_hi = min(C - 1, int(math.ceil(max(ay, by))) + k)
    sub = grid[r_lo:r_hi + 1, c_lo:c_hi + 1]
    for r, c in np.argwhere(sub != FREE):
        if seg_box_dist(ax, ay, bx, by, r + r_lo, c + c_lo) < rho - 1e-12:
            return False
    return True


def point_clearance(grid, x, y, cap=3.0):
    R, C = grid.shape
    best = min(cap, x + 0.5, R - 0.5 - x, y + 0.5, C - 0.5 - y)
    r0, c0 = int(round(x)), int(round(y))
    k = int(math.ceil(cap)) + 1
    for r in range(max(0, r0 - k), min(R, r0 + k + 1)):
        for c in range(max(0, c0 - k), min(C, c0 + k + 1)):
            if grid[r, c] != FREE:
                d = math.hypot(max(abs(x - r) - 0.5, 0.0), max(abs(y - c) - 0.5, 0.0))
                if d < best:
                    best = d
    return best


def walk(grid, a, b):
    a0, a1 = a
    b0, b1 = b
    d0 = b0 - a0
    d1 = b1 - a1
    L = math.hypot(d0, d1)
    if L == 0.0:
        return []
    ev = []
    if d0:
        lo, hi = (a0, b0) if a0 < b0 else (b0, a0)
        ev.extend(((k + 0.5 - a0) / d0, 0) for k in range(lo, hi))
    if d1:
        lo, hi = (a1, b1) if a1 < b1 else (b1, a1)
        ev.extend(((k + 0.5 - a1) / d1, 1) for k in range(lo, hi))
    ev.sort()
    ev.append((1.0, 2))
    s0 = 1 if d0 > 0 else -1
    s1 = 1 if d1 > 0 else -1
    out = []
    prev = 0.0
    i = 0
    n = len(ev)
    while i < n:
        t = ev[i][0]
        k0 = k1 = False
        j = i
        while j < n and ev[j][0] - t < 1e-9:
            if ev[j][1] == 0:
                k0 = True
            elif ev[j][1] == 1:
                k1 = True
            j += 1
        if t - prev > 1e-12:
            tm = 0.5 * (prev + t)
            r = int(math.floor(a0 + tm * d0 + 0.5))
            c = int(math.floor(a1 + tm * d1 + 0.5))
            if grid[r, c] != FREE:
                return None
            out.append((r, c, (t - prev) * L))
        if k0 and k1:
            x = a0 + t * d0
            y = a1 + t * d1
            ra, ca = int(round(x - 0.5 * s0)), int(round(y + 0.5 * s1))
            rb, cb = int(round(x + 0.5 * s0)), int(round(y - 0.5 * s1))
            if grid[ra, ca] != FREE and grid[rb, cb] != FREE:
                return None
        prev = t
        i = j
    return out


def step_cost(grid, w, a, b, rho=0.0):
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == 0 and dc == 0:
        return 0.0
    if grid[a[0], a[1]] != FREE or grid[b[0], b[1]] != FREE:
        return math.inf
    if dr and dc and grid[a[0] + dr, a[1]] != FREE and grid[a[0], a[1] + dc] != FREE:
        return math.inf
    if rho > 0.0 and not seg_clear(grid, a, b, rho):
        return math.inf
    return 0.5 * math.hypot(dr, dc) * (w[a[0], a[1]] + w[b[0], b[1]])


def seg_cost(grid, w, a, b, rho=0.0):
    if abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
        return step_cost(grid, w, a, b, rho)
    cells = walk(grid, a, b)
    if cells is None:
        return math.inf
    if rho > 0.0 and not seg_clear(grid, a, b, rho):
        return math.inf
    return math.fsum(l * w[r, c] for r, c, l in cells)


def path_cost(grid, w, path, rho=0.0):
    return math.fsum(seg_cost(grid, w, path[k], path[k + 1], rho) for k in range(len(path) - 1))


def path_length(path):
    return math.fsum(math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                     for k in range(len(path) - 1))


def smooth(grid, w, path, iterations=3, rho=0.0):
    if not path or len(path) <= 2:
        return list(path) if path else path
    pts = list(path)
    for _ in range(iterations):
        pre = [0.0]
        for k in range(len(pts) - 1):
            pre.append(pre[-1] + seg_cost(grid, w, pts[k], pts[k + 1], rho))
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if seg_cost(grid, w, pts[i], pts[j], rho) <= pre[j] - pre[i] + 1e-12:
                    break
                j -= 1
            out.append(pts[j])
            i = j
        if len(out) == len(pts):
            break
        pts = out
    return pts


def weight_maps(grid, hotspots, warehouse, alpha=P0.ALPHA, lam=P0.LAMBDA, sigma=P0.SIGMA):
    if not warehouse:
        one = np.ones(grid.shape, dtype=np.float64)
        return one, one
    we = P0.env_weight_map(P0.density_map(grid), alpha=alpha)
    wh = P0.hotspot_weight_map(grid, hotspots, lam=lam, sigma=sigma)
    return we * wh, wh


def hotspot_reference(grid, hotspots, sigma_ref=2.0):
    R, C = grid.shape
    if not hotspots:
        z = np.zeros((R, C))
        return z, z.astype(bool)
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    d2 = np.full((R, C), np.inf)
    for hr, hc in hotspots:
        np.minimum(d2, (rr - hr) ** 2 + (cc - hc) ** 2, out=d2)
    return np.exp(-d2 / (2.0 * sigma_ref ** 2)), d2 <= sigma_ref ** 2


def exposure(grid, path, field, mask):
    total = 0.0
    cells = set()
    for k in range(len(path) - 1):
        seg = walk(grid, path[k], path[k + 1])
        if seg is None:
            return math.nan, -1
        for r, c, l in seg:
            total += l * field[r, c]
            if mask[r, c] and l > 1e-9:
                cells.add((r, c))
    return total, len(cells)


def turns(path, thresh):
    return P0.count_turns(path, thresh)


def validate_path(grid, path, start, goal, rho=0.0):
    problems = []
    if not path:
        return ["empty path"]
    if tuple(path[0]) != tuple(start):
        problems.append("path does not begin at start")
    if tuple(path[-1]) != tuple(goal):
        problems.append("path does not end at goal")
    one = np.ones(grid.shape)
    for k in range(len(path) - 1):
        if not math.isfinite(seg_cost(grid, one, path[k], path[k + 1], rho)):
            problems.append("segment %d collides (rho=%g)" % (k, rho))
            break
    return problems


def bres_ok(grid, w, a, b):
    if abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
        return math.isfinite(step_cost(grid, np.ones(grid.shape), a, b))
    return bool(P0.line_of_sight(grid, a, b))


def smooth_los(grid, path, iterations=3):
    if not path or len(path) <= 2:
        return list(path) if path else path
    pts = list(path)
    for _ in range(iterations):
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if walk(grid, pts[i], pts[j]) is not None:
                    break
                j -= 1
            out.append(pts[j])
            i = j
        if len(out) == len(pts):
            break
        pts = out
    return pts
