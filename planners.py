import heapq
import math
import numpy as np

FREE = 0
OBSTACLE = 1

R_D = 3
RHO_TH = 0.25
ALPHA = 0.3
BETA = 0.15
GAMMA = 0.2
LAMBDA = 0.5
SIGMA = 2.0
WEIGHTED_EPS = 1.5
TURN_THRESH_DEG = 18.0

NBRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

KNIGHT = [(-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)]


def in_bounds(grid, r, c):
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def passable(grid, r, c):
    return in_bounds(grid, r, c) and grid[r, c] == FREE


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def no_corner_cut(grid, r, c, dr, dc):
    if dr != 0 and dc != 0:
        if grid[r + dr, c] == OBSTACLE and grid[r, c + dc] == OBSTACLE:
            return False
    return True


def line_of_sight(grid, a, b):
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    if dr == 0 and dc == 0:
        return True
    r, c = r0, c0
    if dc >= dr:
        err = dc // 2
        for _ in range(dc):
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
            if not passable(grid, r, c):
                return False
    else:
        err = dr // 2
        for _ in range(dr):
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
            if not passable(grid, r, c):
                return False
    return True


def reconstruct(came_from, current):
    path = [current]
    while current in came_from and came_from[current] != current:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def smooth_path(grid, path, iterations=3):
    if path is None or len(path) <= 2:
        return path
    pts = list(path)
    for _ in range(iterations):
        i = 0
        out = [pts[0]]
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if line_of_sight(grid, pts[i], pts[j]):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        if len(out) == len(pts):
            pts = out
            break
        pts = out
    return pts


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    return sum(euclid(path[i], path[i + 1]) for i in range(len(path) - 1))


def count_turns(path, thresh_deg=TURN_THRESH_DEG):
    if path is None or len(path) < 3:
        return 0
    turns = 0
    for i in range(1, len(path) - 1):
        v1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        v2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))
        if ang > thresh_deg:
            turns += 1
    return turns


def _result(grid, path, expanded, smooth=True):
    sp = smooth_path(grid, path) if smooth else path
    return {
        "path": path,
        "smoothed": sp,
        "length": path_length(sp),
        "nodes": expanded,
        "turns": count_turns(sp),
        "success": path is not None,
    }


def astar(grid, start, goal):
    open_h = [(euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    came = {}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(came, cur), expanded)
        r, c = cur
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            ng = gc + math.hypot(dr, dc)
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(open_h, (ng + euclid(nb, goal), ng, nb))
    return _result(grid, None, expanded)


def weighted_astar(grid, start, goal, eps=WEIGHTED_EPS):
    open_h = [(eps * euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    came = {}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(came, cur), expanded)
        r, c = cur
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            ng = gc + math.hypot(dr, dc)
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(open_h, (ng + eps * euclid(nb, goal), ng, nb))
    return _result(grid, None, expanded)


def _knight_clear(grid, r, c, dr, dc):
    steps = max(abs(dr), abs(dc))
    for k in range(1, steps + 1):
        ir = r + round(dr * k / steps)
        ic = c + round(dc * k / steps)
        if not passable(grid, ir, ic):
            return False
    return True


def astar16(grid, start, goal):
    moves = NBRS8 + KNIGHT
    open_h = [(euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    came = {}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(came, cur), expanded)
        r, c = cur
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc):
                continue
            if abs(dr) <= 1 and abs(dc) <= 1:
                if not no_corner_cut(grid, r, c, dr, dc):
                    continue
            else:
                if not _knight_clear(grid, r, c, dr, dc):
                    continue
            nb = (nr, nc)
            if nb in closed:
                continue
            ng = gc + math.hypot(dr, dc)
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                heapq.heappush(open_h, (ng + euclid(nb, goal), ng, nb))
    return _result(grid, None, expanded)


def theta_star(grid, start, goal):
    open_h = [(euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    parent = {start: start}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(parent, cur), expanded)
        r, c = cur
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            p = parent[cur]
            if line_of_sight(grid, p, nb):
                ng = g[p] + euclid(p, nb)
                if ng < g.get(nb, math.inf):
                    g[nb] = ng
                    parent[nb] = p
                    heapq.heappush(open_h, (ng + euclid(nb, goal), ng, nb))
            else:
                ng = gc + math.hypot(dr, dc)
                if ng < g.get(nb, math.inf):
                    g[nb] = ng
                    parent[nb] = cur
                    heapq.heappush(open_h, (ng + euclid(nb, goal), ng, nb))
    return _result(grid, None, expanded)


def density_map(grid, r_d=R_D):
    R, C = grid.shape
    obs = (grid == OBSTACLE).astype(np.float64)
    integ = np.zeros((R + 1, C + 1), dtype=np.float64)
    integ[1:, 1:] = np.cumsum(np.cumsum(obs, axis=0), axis=1)
    rows = np.arange(R)
    cols = np.arange(C)
    r0 = np.maximum(0, rows - r_d)
    r1 = np.minimum(R - 1, rows + r_d)
    c0 = np.maximum(0, cols - r_d)
    c1 = np.minimum(C - 1, cols + r_d)
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    R1, C1 = np.meshgrid(r1, c1, indexing="ij")
    s = integ[R1 + 1, C1 + 1] - integ[R0, C1 + 1] - integ[R1 + 1, C0] + integ[R0, C0]
    area = (R1 - R0 + 1) * (C1 - C0 + 1)
    return s / area


def env_weight_map(rho, alpha=ALPHA):
    w = np.ones_like(rho)
    mask = rho > RHO_TH
    w[mask] = 1.0 + alpha * rho[mask]
    return w


def hotspot_weight_map(grid, hotspots, lam=LAMBDA, sigma=SIGMA):
    R, C = grid.shape
    if not hotspots:
        return np.ones((R, C), dtype=np.float64)
    hs = np.array(hotspots, dtype=np.float64)
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    d2min = np.full((R, C), np.inf)
    for hr, hc in hs:
        d2 = (rr - hr) ** 2 + (cc - hc) ** 2
        np.minimum(d2min, d2, out=d2min)
    return 1.0 + lam * np.exp(-d2min / (2 * sigma ** 2))


def _unit(v):
    n = math.hypot(v[0], v[1])
    if n == 0:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def proposed(grid, start, goal, hotspots=None, warehouse=False):
    rho = density_map(grid)
    we = env_weight_map(rho)
    wh = hotspot_weight_map(grid, hotspots) if warehouse else np.ones(grid.shape)

    def h_prime(node, move_dir):
        h = euclid(node, goal)
        dg = _unit((goal[0] - node[0], goal[1] - node[1]))
        a = move_dir[0] * dg[0] + move_dir[1] * dg[1]
        delta = BETA * max(0.0, 1.0 - a)
        return h * (1.0 + delta)

    h0 = euclid(start, goal)
    open_h = [(h0, 0.0, start)]
    g = {start: 0.0}
    came = {}
    move_into = {start: (0.0, 0.0)}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(came, cur), expanded)
        r, c = cur
        d_prev = move_into[cur]
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            d_curr = _unit((dr, dc))
            step = math.hypot(dr, dc)
            c_turn = 0.0
            if d_prev != (0.0, 0.0):
                dp = d_prev[0] * d_curr[0] + d_prev[1] * d_curr[1]
                dp = max(-1.0, min(1.0, dp))
                theta = math.acos(dp)
                c_turn = GAMMA * theta / math.pi
            ng = gc + step * we[nr, nc] * wh[nr, nc] + c_turn
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                move_into[nb] = d_curr
                heapq.heappush(open_h, (ng + h_prime(nb, d_curr), ng, nb))
    return _result(grid, None, expanded)


def proposed_v2(grid, start, goal, hotspots=None, warehouse=False,
                eps=1.0, beta=BETA, gamma=GAMMA, use_env=True):
    if warehouse and use_env:
        rho = density_map(grid)
        we = env_weight_map(rho)
    elif use_env:
        rho = density_map(grid)
        we = env_weight_map(rho)
    else:
        we = np.ones(grid.shape)
    wh = hotspot_weight_map(grid, hotspots) if warehouse else np.ones(grid.shape)

    def h_prime(node, move_dir):
        h = euclid(node, goal)
        dg = _unit((goal[0] - node[0], goal[1] - node[1]))
        a = move_dir[0] * dg[0] + move_dir[1] * dg[1]
        delta = beta * max(0.0, 1.0 - a)
        return eps * h * (1.0 + delta)

    open_h = [(eps * euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    came = {}
    move_into = {start: (0.0, 0.0)}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(came, cur), expanded)
        r, c = cur
        d_prev = move_into[cur]
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            d_curr = _unit((dr, dc))
            step = math.hypot(dr, dc)
            c_turn = 0.0
            if d_prev != (0.0, 0.0):
                dp = max(-1.0, min(1.0, d_prev[0] * d_curr[0] + d_prev[1] * d_curr[1]))
                c_turn = gamma * math.acos(dp) / math.pi
            ng = gc + step * we[nr, nc] * wh[nr, nc] + c_turn
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                came[nb] = cur
                move_into[nb] = d_curr
                heapq.heappush(open_h, (ng + h_prime(nb, d_curr), ng, nb))
    return _result(grid, None, expanded)


def proposed_v3(grid, start, goal, hotspots=None, warehouse=False,
                eps=1.0, beta=BETA, alpha=ALPHA, lam=LAMBDA, sigma=SIGMA):
    we = env_weight_map(density_map(grid), alpha=alpha) if warehouse else np.ones(grid.shape)
    wh = hotspot_weight_map(grid, hotspots, lam=lam, sigma=sigma) if warehouse else np.ones(grid.shape)

    def cost(a, b):
        base = euclid(a, b)
        return base * 0.5 * (we[a] * wh[a] + we[b] * wh[b])

    def h_prime(node):
        h = euclid(node, goal)
        dg = _unit((goal[0] - node[0], goal[1] - node[1]))
        dm = _unit((goal[0] - node[0], goal[1] - node[1]))
        a = dm[0] * dg[0] + dm[1] * dg[1]
        delta = beta * max(0.0, 1.0 - a)
        return eps * h * (1.0 + delta)

    open_h = [(eps * euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    parent = {start: start}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(parent, cur), expanded)
        r, c = cur
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            p = parent[cur]
            if line_of_sight(grid, p, nb):
                ng = g[p] + cost(p, nb)
                anchor = p
            else:
                ng = gc + cost(cur, nb)
                anchor = cur
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = anchor
                dm = _unit((nb[0] - anchor[0], nb[1] - anchor[1]))
                dg = _unit((goal[0] - nb[0], goal[1] - nb[1]))
                a = dm[0] * dg[0] + dm[1] * dg[1]
                delta = beta * max(0.0, 1.0 - a)
                hn = eps * euclid(nb, goal) * (1.0 + delta)
                heapq.heappush(open_h, (ng + hn, ng, nb))
    return _result(grid, None, expanded)


def proposed_adaptive(grid, start, goal, hotspots=None, warehouse=False,
                      eps_max=1.5, beta=BETA, adaptive=True):
    rho = density_map(grid)
    we = env_weight_map(rho) if warehouse else np.ones(grid.shape)
    wh = hotspot_weight_map(grid, hotspots) if warehouse else np.ones(grid.shape)

    d0 = euclid(start, goal) + 1e-9

    def eps_at(node):
        if not adaptive:
            return eps_max
        prog = euclid(node, goal) / d0
        return 1.0 + (eps_max - 1.0) * min(1.0, prog)

    def cost(a, b):
        base = euclid(a, b)
        return base * 0.5 * (we[a] * wh[a] + we[b] * wh[b])

    open_h = [(eps_at(start) * euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    parent = {start: start}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(parent, cur), expanded)
        r, c = cur
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            p = parent[cur]
            if line_of_sight(grid, p, nb):
                ng = g[p] + cost(p, nb)
                anchor = p
            else:
                ng = gc + cost(cur, nb)
                anchor = cur
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = anchor
                dm = _unit((nb[0] - anchor[0], nb[1] - anchor[1]))
                dg = _unit((goal[0] - nb[0], goal[1] - nb[1]))
                al = dm[0] * dg[0] + dm[1] * dg[1]
                delta = beta * max(0.0, 1.0 - al)
                hn = eps_at(nb) * euclid(nb, goal) * (1.0 + delta)
                heapq.heappush(open_h, (ng + hn, ng, nb))
    return _result(grid, None, expanded)


def _jps_prune_dirs(grid, node, parent):
    r, c = node
    if parent is None:
        dirs = []
        for dr, dc in NBRS8:
            if passable(grid, r + dr, c + dc) and no_corner_cut(grid, r, c, dr, dc):
                dirs.append((dr, dc))
        return dirs
    pr, pc = parent
    dr = (r - pr)
    dc = (c - pc)
    dr = (dr > 0) - (dr < 0)
    dc = (dc > 0) - (dc < 0)
    dirs = []
    if dr != 0 and dc != 0:
        if passable(grid, r, c + dc):
            dirs.append((0, dc))
        if passable(grid, r + dr, c):
            dirs.append((dr, 0))
        if passable(grid, r + dr, c + dc) and no_corner_cut(grid, r, c, dr, dc):
            dirs.append((dr, dc))
        if (not passable(grid, r - dr, c)) and passable(grid, r - dr, c + dc):
            dirs.append((-dr, dc))
        if (not passable(grid, r, c - dc)) and passable(grid, r + dr, c - dc):
            dirs.append((dr, -dc))
    elif dr != 0:
        if passable(grid, r + dr, c):
            dirs.append((dr, 0))
        if (not passable(grid, r, c + 1)) and passable(grid, r + dr, c + 1):
            dirs.append((dr, 1))
        if (not passable(grid, r, c - 1)) and passable(grid, r + dr, c - 1):
            dirs.append((dr, -1))
    else:
        if passable(grid, r, c + dc):
            dirs.append((0, dc))
        if (not passable(grid, r + 1, c)) and passable(grid, r + 1, c + dc):
            dirs.append((1, dc))
        if (not passable(grid, r - 1, c)) and passable(grid, r - 1, c + dc):
            dirs.append((-1, dc))
    return dirs


def _jps_jump(grid, r, c, dr, dc, goal):
    nr, nc = r + dr, c + dc
    if not passable(grid, nr, nc):
        return None
    if dr != 0 and dc != 0 and not no_corner_cut(grid, r, c, dr, dc):
        return None
    if (nr, nc) == goal:
        return (nr, nc)
    if dr != 0 and dc != 0:
        if (not passable(grid, nr - dr, nc)) and passable(grid, nr - dr, nc + dc):
            return (nr, nc)
        if (not passable(grid, nr, nc - dc)) and passable(grid, nr + dr, nc - dc):
            return (nr, nc)
        if _jps_jump(grid, nr, nc, dr, 0, goal) is not None:
            return (nr, nc)
        if _jps_jump(grid, nr, nc, 0, dc, goal) is not None:
            return (nr, nc)
    elif dr != 0:
        if (not passable(grid, nr, nc + 1)) and passable(grid, nr + dr, nc + 1):
            return (nr, nc)
        if (not passable(grid, nr, nc - 1)) and passable(grid, nr + dr, nc - 1):
            return (nr, nc)
    else:
        if (not passable(grid, nr + 1, nc)) and passable(grid, nr + 1, nc + dc):
            return (nr, nc)
        if (not passable(grid, nr - 1, nc)) and passable(grid, nr - 1, nc + dc):
            return (nr, nc)
    return _jps_jump(grid, nr, nc, dr, dc, goal)


def jps(grid, start, goal):
    open_h = [(euclid(start, goal), 0.0, start)]
    g = {start: 0.0}
    came = {}
    parent_of = {start: None}
    closed = set()
    expanded = 0
    while open_h:
        f, gc, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return _result(grid, reconstruct(came, cur), expanded)
        for dr, dc in _jps_prune_dirs(grid, cur, parent_of[cur]):
            jp = _jps_jump(grid, cur[0], cur[1], dr, dc, goal)
            if jp is None:
                continue
            if jp in closed:
                continue
            ng = gc + euclid(cur, jp)
            if ng < g.get(jp, math.inf):
                g[jp] = ng
                came[jp] = cur
                parent_of[jp] = cur
                heapq.heappush(open_h, (ng + euclid(jp, goal), ng, jp))
    return _result(grid, None, expanded)


def proposed_grid_only(grid, start, goal, hotspots=None, warehouse=False,
                        eps=1.1, beta=0.15):
    """
    Proposed method with any-angle LOS disabled (8-neighbor grid expansion only).
    Used for ablation to isolate the contribution of the any-angle mechanism.
    """
    rho = density_map(grid)
    we = env_weight_map(rho) if warehouse else np.ones(grid.shape)
    wh = hotspot_weight_map(grid, hotspots) if warehouse else np.ones(grid.shape)

    def cost(a, b):
        base = euclid(a, b)
        return base * 0.5 * (we[a] * wh[a] + we[b] * wh[b])

    open_h = []
    g = {start: 0.0}
    parent = {start: start}
    closed = set()
    expanded = 0

    dm_start = _unit((goal[0] - start[0], goal[1] - start[1]))
    dg_start = dm_start
    al_start = dm_start[0] * dg_start[0] + dm_start[1] * dg_start[1]
    delta_start = beta * max(0.0, 1.0 - al_start)
    h_start = eps * euclid(start, goal) * (1.0 + delta_start)
    heapq.heappush(open_h, (h_start, 0.0, start))

    while open_h:
        _, gc, cur = heapq.heappop(open_h)
        if cur == goal:
            return _result(grid, reconstruct(parent, cur), expanded)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1

        r, c = cur
        for dr, dc in NBRS8:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc) or not no_corner_cut(grid, r, c, dr, dc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue

            # Grid-only: always use current node as parent (no LOS shortcut)
            ng = gc + cost(cur, nb)
            anchor = cur

            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = anchor
                dm = _unit((nb[0] - anchor[0], nb[1] - anchor[1]))
                dg = _unit((goal[0] - nb[0], goal[1] - nb[1]))
                al = dm[0] * dg[0] + dm[1] * dg[1]
                delta = beta * max(0.0, 1.0 - al)
                hn = eps * euclid(nb, goal) * (1.0 + delta)
                heapq.heappush(open_h, (ng + hn, ng, nb))

    return _result(grid, None, expanded)
