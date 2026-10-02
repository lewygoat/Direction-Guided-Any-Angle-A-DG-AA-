import math
import sys
import time

import numpy as np

import rv_geom as G
import rv_planners as RP
import environments as E

TOL = 1e-7
FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def t_walk():
    rng = np.random.default_rng(1)
    empty = np.zeros((30, 30), dtype=np.int8)
    for _ in range(500):
        a = tuple(int(x) for x in rng.integers(0, 30, 2))
        b = tuple(int(x) for x in rng.integers(0, 30, 2))
        cells = G.walk(empty, a, b)
        check(cells is not None, "walk blocked on empty grid %s %s" % (a, b))
        if cells:
            s = math.fsum(l for _, _, l in cells)
            check(abs(s - math.hypot(a[0] - b[0], a[1] - b[1])) < 1e-9, "walk length mismatch")
    for seed in range(20):
        grid, _, _, _ = E.make_general(seed)
        free = np.argwhere(grid == 0)
        for _ in range(200):
            a = tuple(int(x) for x in free[rng.integers(len(free))])
            b = tuple(int(x) for x in free[rng.integers(len(free))])
            cells = G.walk(grid, a, b)
            if cells is not None:
                check(all(grid[r, c] == 0 for r, c, _ in cells), "walk passes obstacle")
                check(G.walk(grid, b, a) is not None, "walk asymmetric")


def t_astar_matches_original():
    for seed in range(15):
        grid, s, g, _ = E.make_general(seed)
        one = RP.np_ones(grid)
        r = RP.search(grid, one, s, g)
        o = G.P0.astar(grid, s, g)
        check(r["success"] and o["success"], "astar failed seed %d" % seed)
        check(abs(r["cost"] - G.path_length(o["path"])) < 1e-6,
              "astar cost differs from original seed %d: %.6f vs %.6f" % (seed, r["cost"], G.path_length(o["path"])))


def _bound_case(grid, w, s, g, cfg, tag):
    ref = RP.search(grid, w, s, g)
    if not ref["success"]:
        return
    c8 = ref["cost"]
    W = RP.bound_factor(cfg)
    for name in ("Proposed", "Proposed (fixed eps)", "Weighted Theta*", "Theta*", "Proposed (grid only)"):
        r = RP.run_method(name, grid, w, s, g, cfg)
        check(r["success"], "%s failed %s" % (name, tag))
        if not r["success"]:
            continue
        probs = G.validate_path(grid, r["path"], s, g)
        check(not probs, "%s invalid path %s %s" % (name, tag, probs))
        pc = G.path_cost(grid, w, r["path"])
        check(pc <= r["cost"] + TOL, "%s path cost %.6f > g %.6f %s" % (name, pc, r["cost"], tag))
        lim = c8 if name == "Theta*" else (cfg["eps_max"] if name == "Weighted Theta*" else W) * c8
        check(pc <= lim + TOL, "%s bound violated %.6f > %.6f %s" % (name, pc, lim, tag))
        sm = G.smooth(grid, w, r["path"])
        check(not G.validate_path(grid, sm, s, g), "%s smoothed invalid %s" % (name, tag))
        check(G.path_cost(grid, w, sm) <= pc + TOL, "%s smoothing raised cost %s" % (name, tag))


def t_bounds():
    for cfg in ({"eps_max": 1.1, "beta": 0.15}, {"eps_max": 1.5, "beta": 0.3}, {"eps_max": 2.0, "beta": 0.5}):
        for seed in range(10):
            grid, s, g, _ = E.make_general(seed)
            _bound_case(grid, RP.np_ones(grid), s, g, cfg, "gen%d" % seed)
            grid, s, g, hs = E.make_warehouse(seed)
            w, _ = G.weight_maps(grid, hs, True)
            _bound_case(grid, w, s, g, cfg, "wh%d" % seed)


def t_adversarial():
    rng = np.random.default_rng(7)
    cfg = {"eps_max": 2.0, "beta": 0.5}
    viol_noreopen = 0
    cases = 0
    for k in range(400):
        n = int(rng.integers(6, 14))
        grid = (rng.random((n, n)) < rng.uniform(0.2, 0.45)).astype(np.int8)
        w = 1.0 + rng.random((n, n)) * rng.choice([0.0, 1.0, 4.0])
        free = np.argwhere(grid == 0)
        if len(free) < 2:
            continue
        s = tuple(int(x) for x in free[0])
        g = tuple(int(x) for x in free[-1])
        if s == g or not E.has_solution(grid, s, g):
            continue
        cases += 1
        _bound_case(grid, w, s, g, cfg, "adv%d" % k)
        c8 = RP.search(grid, w, s, g)["cost"]
        r = RP.run_method("Proposed (no reopen)", grid, w, s, g, cfg)
        if r["success"] and G.path_cost(grid, w, r["path"]) > RP.bound_factor(cfg) * c8 + TOL:
            viol_noreopen += 1
    print("adversarial cases=%d, no-reopen bound violations=%d" % (cases, viol_noreopen))
    check(cases > 100, "too few adversarial cases")


def t_validation():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = 1
    one = RP.np_ones(grid)
    bad = [
        lambda: RP.search(grid, one, (2, 2), (0, 0)),
        lambda: RP.search(grid, one, (0, 0), (9, 9)),
        lambda: RP.search(grid, one, (0, 0), (4, 4), eps=0.5),
        lambda: RP.search(grid, one, (0, 0), (4, 4), beta=-1),
        lambda: RP.search(grid, one * 0.5, (0, 0), (4, 4)),
        lambda: RP.search(grid.astype(float) * 2, one, (0, 0), (4, 4)),
        lambda: RP.search(grid, one, (0.5, 0), (4, 4)),
    ]
    for i, fn in enumerate(bad):
        try:
            fn()
            FAIL.append("validation case %d not rejected" % i)
        except ValueError:
            pass
    walled = np.zeros((5, 5), dtype=np.int8)
    walled[:, 2] = 1
    r = RP.search(walled, RP.np_ones(walled), (0, 0), (0, 4))
    check(not r["success"] and r["reason"] == "no path", "unreachable goal not reported")
    try:
        RP.search(np.zeros((40, 40), dtype=np.int8), RP.np_ones(np.zeros((40, 40))), (0, 0), (39, 39), max_expansions=5)
        FAIL.append("budget not enforced")
    except RP.SearchBudgetExceeded:
        pass


def t_clearance():
    rng = np.random.default_rng(3)
    for seed in range(8):
        grid, s, g, _ = E.make_general(seed)
        free = np.argwhere(grid == 0)
        for _ in range(150):
            a = tuple(int(x) for x in free[rng.integers(len(free))])
            b = tuple(int(x) for x in free[rng.integers(len(free))])
            for rho in (0.3, 0.5):
                fast = G.seg_clear(grid, a, b, rho)
                L = math.hypot(a[0] - b[0], a[1] - b[1])
                n = max(2, int(L / 0.02))
                brute = min(G.point_clearance(grid, a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                            for i in range(n + 1)) >= rho - 1e-3
                if fast and not brute:
                    FAIL.append("seg_clear accepts colliding segment %s %s rho=%g" % (a, b, rho))
    for seed in range(5):
        for env in ("general", "warehouse"):
            grid, s, g, hs = (E.make_general(seed) if env == "general" else E.make_warehouse(seed))
            w = G.weight_maps(grid, hs, env == "warehouse")[0]
            for rho in (0.3, 0.5):
                if min(G.point_clearance(grid, *s), G.point_clearance(grid, *g)) < rho:
                    continue
                r = RP.search(grid, w, s, g, anyangle=True, eps=1.5, beta=0.15, adaptive=True, rho=rho)
                if r["success"]:
                    sm = G.smooth(grid, w, r["path"], rho=rho)
                    check(not G.validate_path(grid, sm, s, g, rho), "footprint path collides %s %d" % (env, seed))


def main():
    t0 = time.time()
    for fn in (t_walk, t_astar_matches_original, t_bounds, t_adversarial, t_validation, t_clearance):
        n0 = len(FAIL)
        fn()
        print("%-28s %s" % (fn.__name__, "PASS" if len(FAIL) == n0 else "FAIL (%d)" % (len(FAIL) - n0)))
    for m in FAIL[:30]:
        print("  -", m)
    print("total failures=%d, %.1fs" % (len(FAIL), time.time() - t0))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
