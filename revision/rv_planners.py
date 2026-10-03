import heapq
import math
import sys

import numpy as np

import rv_geom as G

sys.setrecursionlimit(100000)

NB8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
KNIGHT = ((-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1))
NB16 = NB8 + KNIGHT


class SearchBudgetExceeded(RuntimeError):
    pass


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reconstruct(parent, goal, start, limit):
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
        if len(path) > limit:
            raise RuntimeError("parent cycle detected")
    path.reverse()
    return path


def search(grid, w, start, goal, anyangle=False, eps=1.0, beta=0.0, adaptive=False,
           reopen=True, moves=NB8, max_expansions=None, rho=0.0,
           collision="exact", cost_mode="integrated", parent_rule="min"):
    G.check_grid(grid)
    G.check_weights(grid, w)
    G.check_point(grid, start, "start")
    G.check_point(grid, goal, "goal")
    if not (eps >= 1.0 and math.isfinite(eps)):
        raise ValueError("eps must be finite and >= 1")
    if not (beta >= 0.0 and math.isfinite(beta)):
        raise ValueError("beta must be finite and >= 0")
    if collision not in ("exact", "bresenham") or cost_mode not in ("integrated", "endpoint") \
            or parent_rule not in ("min", "greedy"):
        raise ValueError("unknown collision / cost_mode / parent_rule")
    if collision == "bresenham" and cost_mode == "integrated":
        raise ValueError("bresenham collision is only defined with endpoint cost")
    if (collision != "exact" or cost_mode != "integrated") and (rho > 0.0 or moves is not NB8):
        raise ValueError("original-semantics switches support only 8-neighbour point search")
    alt = collision != "exact" or cost_mode != "integrated"
    greedy = parent_rule == "greedy"
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    R, C = grid.shape
    if max_expansions is None:
        max_expansions = 50 * R * C
    stats = {"expanded": 0, "reopened": 0, "los_calls": 0}
    if start == goal:
        return dict(stats, success=True, path=[start], cost=0.0, reason="trivial")

    d0 = _dist(start, goal)
    g0, g1 = goal

    def key(n, anchor, gn):
        hn = math.hypot(g0 - n[0], g1 - n[1])
        if hn == 0.0:
            return gn
        e = eps
        if adaptive:
            e = 1.0 + (eps - 1.0) * min(1.0, hn / d0)
        delta = 0.0
        if beta > 0.0 and anchor != n:
            mr, mc = n[0] - anchor[0], n[1] - anchor[1]
            ml = math.hypot(mr, mc)
            a = (mr * (g0 - n[0]) + mc * (g1 - n[1])) / (ml * hn)
            delta = beta * max(0.0, 1.0 - a)
        return gn + e * hn * (1.0 + delta)

    g = {start: 0.0}
    parent = {start: start}
    closed = set()
    cnt = 0
    heap = [(key(start, start, 0.0), 0.0, cnt, start)]
    while heap:
        f, neg_g, _, cur = heapq.heappop(heap)
        gc = -neg_g
        if gc > g[cur] + 1e-12 or cur in closed:
            continue
        closed.add(cur)
        stats["expanded"] += 1
        if stats["expanded"] > max_expansions:
            raise SearchBudgetExceeded("expansion budget %d exceeded" % max_expansions)
        if cur == goal:
            path = _reconstruct(parent, goal, start, R * C + 1)
            return dict(stats, success=True, path=path, cost=gc, reason="ok")
        r, c = cur
        p = parent[cur]
        use_p = anyangle and p != cur
        gp = g[p] if use_p else 0.0
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= R or nc < 0 or nc >= C or grid[nr, nc] != 0:
                continue
            nb = (nr, nc)
            if nb in closed and not reopen:
                continue
            if -1 <= dr <= 1 and -1 <= dc <= 1:
                c1 = G.step_cost(grid, w, cur, nb, rho)
            else:
                c1 = G.seg_cost(grid, w, cur, nb, rho)
            if not math.isfinite(c1):
                continue
            cand = gc + c1
            anchor = cur
            if use_p and nb != p:
                stats["los_calls"] += 1
                if alt:
                    c2 = _alt_seg(grid, w, p, nb, collision)
                else:
                    c2 = G.seg_cost(grid, w, p, nb, rho)
                if greedy:
                    if math.isfinite(c2):
                        cand = gp + c2
                        anchor = p
                elif gp + c2 < cand - 1e-12:
                    cand = gp + c2
                    anchor = p
            if cand < g.get(nb, math.inf) - 1e-12:
                if nb in closed:
                    closed.discard(nb)
                    stats["reopened"] += 1
                g[nb] = cand
                parent[nb] = anchor
                cnt += 1
                heapq.heappush(heap, (key(nb, anchor, cand), -cand, cnt, nb))
    return dict(stats, success=False, path=None, cost=math.inf, reason="no path")


def _alt_seg(grid, w, a, b, collision):
    if abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
        return G.step_cost(grid, w, a, b)
    if collision == "bresenham":
        ok = G.P0.line_of_sight(grid, a, b)
    else:
        ok = G.walk(grid, a, b) is not None
    if not ok:
        return math.inf
    return 0.5 * math.hypot(b[0] - a[0], b[1] - a[1]) * (w[a[0], a[1]] + w[b[0], b[1]])


def jps(grid, start, goal, max_expansions=None):
    G.check_grid(grid)
    G.check_point(grid, start, "start")
    G.check_point(grid, goal, "goal")
    P0 = G.P0
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    R, C = grid.shape
    if max_expansions is None:
        max_expansions = 50 * R * C
    stats = {"expanded": 0, "reopened": 0, "los_calls": 0}
    if start == goal:
        return dict(stats, success=True, path=[start], cost=0.0, reason="trivial")
    g = {start: 0.0}
    came = {start: start}
    par = {start: None}
    closed = set()
    cnt = 0
    heap = [(_dist(start, goal), 0.0, cnt, start)]
    while heap:
        f, neg_g, _, cur = heapq.heappop(heap)
        gc = -neg_g
        if gc > g[cur] + 1e-12 or cur in closed:
            continue
        closed.add(cur)
        stats["expanded"] += 1
        if stats["expanded"] > max_expansions:
            raise SearchBudgetExceeded("expansion budget %d exceeded" % max_expansions)
        if cur == goal:
            path = _reconstruct(came, goal, start, R * C + 1)
            return dict(stats, success=True, path=path, cost=gc, reason="ok")
        for dr, dc in P0._jps_prune_dirs(grid, cur, par[cur]):
            jp = P0._jps_jump(grid, cur[0], cur[1], dr, dc, goal)
            if jp is None or jp in closed:
                continue
            ng = gc + _dist(cur, jp)
            if ng < g.get(jp, math.inf) - 1e-12:
                g[jp] = ng
                came[jp] = cur
                par[jp] = cur
                cnt += 1
                heapq.heappush(heap, (ng + _dist(jp, goal), -ng, cnt, jp))
    return dict(stats, success=False, path=None, cost=math.inf, reason="no path")


def run_method(name, grid, w, start, goal, cfg):
    if name == "Traditional A*":
        return search(grid, w, start, goal)
    if name == "Weighted A*":
        return search(grid, w, start, goal, eps=cfg["wastar_eps"])
    if name == "16-neighbor A*":
        return search(grid, w, start, goal, moves=NB16)
    if name == "Theta*":
        return search(grid, w, start, goal, anyangle=True)
    if name == "JPS":
        return jps(grid, start, goal)
    if name.startswith("WTheta*@"):
        e = float(name.split("@", 1)[1])
        return search(grid, w, start, goal, anyangle=True, eps=e)
    if name == "Weighted Theta*":
        return search(grid, w, start, goal, anyangle=True, eps=cfg["eps_max"])
    if name == "Weighted Theta* (matched W)":
        return search(grid, w, start, goal, anyangle=True,
                      eps=(1.0 + 2.0 * cfg["beta"]) * cfg["eps_max"])
    if name == "Proposed":
        return search(grid, w, start, goal, anyangle=True, eps=cfg["eps_max"],
                      beta=cfg["beta"], adaptive=True)
    if name == "Proposed (fixed eps)":
        return search(grid, w, start, goal, anyangle=True, eps=cfg["eps_max"], beta=cfg["beta"])
    if name == "Proposed (no reopen)":
        return search(grid, w, start, goal, anyangle=True, eps=cfg["eps_max"],
                      beta=cfg["beta"], adaptive=True, reopen=False)
    if name == "Proposed (grid only)":
        return search(grid, w, start, goal, eps=cfg["eps_max"], beta=cfg["beta"], adaptive=True)
    if name == "Proposed (beta=0)":
        return search(grid, w, start, goal, anyangle=True, eps=cfg["eps_max"], adaptive=True)
    raise KeyError("unknown method " + name)


def bound_factor(cfg):
    return (1.0 + 2.0 * cfg["beta"]) * cfg["eps_max"]


def np_ones(grid):
    return np.ones(grid.shape, dtype=np.float64)
