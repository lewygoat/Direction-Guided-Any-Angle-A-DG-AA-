import json
import math
import os
import signal
import time
import traceback

import numpy as np

import rv_geom as G
import rv_planners as RP
import environments as E

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TOL = 1e-7
SQ2H = math.sqrt(2.0) / 2.0


class InstanceTimeout(Exception):
    pass


_NB_READY = [False]


def _nb():
    import rv_numba as NB
    if not _NB_READY[0]:
        NB.warmup()
        _NB_READY[0] = True
    return NB


def _nb_equal(NB, out, r):
    st, cst, ex, ro, los, pth = out
    if bool(st == NB.ST_OK) != bool(r["success"]):
        return False
    if ex != r["expanded"] or ro != r["reopened"] or los != r["los_calls"]:
        return False
    if r["success"]:
        if abs(cst - r["cost"]) > 1e-9 * max(1.0, r["cost"]):
            return False
        if [tuple(int(v) for v in p) for p in pth] != [tuple(p) for p in r["path"]]:
            return False
    return True


def _alarm(signum, frame):
    raise InstanceTimeout("instance timeout")


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def method_kwargs(label, cfg):
    e, b = cfg["eps_max"], cfg["beta"]
    W = (1.0 + 2.0 * b) * e
    table = {
        "Traditional A*": ({}, 1.0),
        "Weighted A*": ({"eps": e}, e),
        "16-neighbor A*": ({"moves": RP.NB16}, 1.0),
        "Theta*": ({"anyangle": True}, 1.0),
        "JPS": (None, None),
        "Weighted Theta*": ({"anyangle": True, "eps": e}, e),
        "Weighted Theta* (matched W)": ({"anyangle": True, "eps": W}, W),
        "Proposed": ({"anyangle": True, "eps": e, "beta": b, "adaptive": True}, W),
        "Proposed (fixed eps)": ({"anyangle": True, "eps": e, "beta": b}, W),
        "Proposed (beta=0)": ({"anyangle": True, "eps": e, "adaptive": True}, e),
        "Proposed (grid only)": ({"eps": e, "beta": b, "adaptive": True}, W),
        "Proposed (no reopen)": ({"anyangle": True, "eps": e, "beta": b, "adaptive": True, "reopen": False}, None),
    }
    if label in table:
        return table[label]
    parts = label.split("@")
    if parts[0] == "WTheta*":
        x = float(parts[1])
        return {"anyangle": True, "eps": x}, x
    if parts[0] in ("Prop", "PropFix"):
        x, y = float(parts[1]), float(parts[2])
        return {"anyangle": True, "eps": x, "beta": y, "adaptive": parts[0] == "Prop"}, (1.0 + 2.0 * y) * x
    raise KeyError("unknown method label " + label)


def clearance_profile(grid, path, step=0.1, cap=2.0):
    R, C = grid.shape
    samples = []
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(L / step)))
        for i in range(n):
            t = i / n
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            best = cap
            r0, c0 = int(round(x)), int(round(y))
            for rr in range(r0 - 3, r0 + 4):
                for cc in range(c0 - 3, c0 + 4):
                    if 0 <= rr < R and 0 <= cc < C and grid[rr, cc] == G.OBST:
                        d = math.hypot(max(abs(x - rr) - 0.5, 0.0), max(abs(y - cc) - 0.5, 0.0))
                        if d < best:
                            best = d
            samples.append((best, L / n))
    return samples


def build_map(spec):
    env = spec["env"]
    seed = spec["seed"]
    if env == "general":
        if spec.get("gen") == "legacy":
            grid, s, g, hs = E.make_general(seed)
        else:
            S = spec.get("size", 50)
            grid, s, g, hs = E.make_general_density(seed, spec.get("density", 0.2), R=S, C=S)
    elif env == "warehouse":
        S = spec.get("size", 50)
        grid, s, g, hs = E.make_warehouse(seed, R=S, C=S)
    elif env == "movingai":
        import rv_movingai as MA
        grid = np.array(MA.load_map(spec["map"]), dtype=np.int8)
        q = MA.load_queries(spec["map"])[seed]
        return grid, tuple(q["start"]), tuple(q["goal"]), []
    else:
        raise ValueError("unknown env " + str(env))
    hs = hs or []
    q = spec.get("query", "diag")
    if q != "diag":
        s, g = E.random_query(grid, seed * 100 + int(q) + 5000000)
    return grid, (int(s[0]), int(s[1])), (int(g[0]), int(g[1])), hs


def weight_for(grid, hs, env, wkey):
    wh_ = env in ("warehouse", "movingai")
    if wkey == "full":
        return G.weight_maps(grid, hs, wh_)[0]
    if wkey == "unit":
        return RP.np_ones(grid)
    if wkey == "nohot":
        return G.weight_maps(grid, [], wh_)[0]
    if wkey == "noenv":
        return G.weight_maps(grid, hs, wh_, alpha=0.0)[0]
    if wkey.startswith("p:"):
        _, a, l, s = wkey.split(":")
        return G.weight_maps(grid, hs, wh_, alpha=float(a), lam=float(l), sigma=float(s))[0]
    raise ValueError("unknown weight key " + wkey)


def select_hotspot_query(grid, hs, seed, env):
    field, mask = G.hotspot_reference(grid, hs)
    w0 = weight_for(grid, hs, env, "p:0.3:0.0:2.0")
    for q in range(60):
        try:
            s, g = E.random_query(grid, seed * 100 + q + 7000000)
        except RuntimeError:
            continue
        r = RP.search(grid, w0, s, g, anyangle=True)
        if r["success"]:
            _, cells = G.exposure(grid, G.smooth(grid, w0, r["path"]), field, mask)
            if cells > 0:
                return (int(s[0]), int(s[1])), (int(g[0]), int(g[1])), q
    return None, None, None


def _unc_eval(grid, path, hs, start, goal, seed):
    rng = np.random.default_rng(seed + 424242)
    out = {}
    free = np.argwhere(grid == G.FREE)
    for sd in (1.0, 2.0):
        vals = []
        for _ in range(20):
            moved = []
            for hr, hc in hs:
                r = int(np.clip(round(hr + rng.normal(0, sd)), 0, grid.shape[0] - 1))
                c = int(np.clip(round(hc + rng.normal(0, sd)), 0, grid.shape[1] - 1))
                moved.append((r, c))
            f, m = G.hotspot_reference(grid, moved)
            vals.append(G.exposure(grid, path, f, m)[0])
        out["exp_shift%g" % sd] = float(np.mean(vals))
    for frac in (0.01, 0.03):
        ok = 0
        k = max(1, int(frac * len(free)))
        for _ in range(20):
            g2 = grid.copy()
            idx = rng.choice(len(free), size=k, replace=False)
            for i in idx:
                cell = (int(free[i][0]), int(free[i][1]))
                if cell != start and cell != goal:
                    g2[cell] = G.OBST
            ok += int(not G.validate_path(g2, path, start, goal))
        out["valid_add%g" % frac] = ok / 20.0
    return out


def dispatch(spec):
    if spec.get("runner") == "bridge":
        import rv_bridge
        return rv_bridge.run_bridge(spec)
    return run_instance(spec)


def select_clear_query(grid, seed, rho):
    one = RP.np_ones(grid)
    for q in range(200):
        try:
            s, g = E.random_query(grid, seed * 100 + q + 9000000)
        except RuntimeError:
            continue
        if min(G.point_clearance(grid, *s), G.point_clearance(grid, *g)) < rho - 1e-12:
            continue
        if RP.search(grid, one, s, g, rho=rho)["success"]:
            return (int(s[0]), int(s[1])), (int(g[0]), int(g[1])), q
    return None, None, None


def run_instance(spec):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(spec.get("timeout", 1800)))
    base = {k: spec[k] for k in ("stage", "env", "seed") if k in spec}
    base.update({k: spec.get(k) for k in ("density", "size", "query", "inflate", "map", "reps")})
    recs = []
    done_labels = set()
    try:
        grid0, s, g, hs = build_map(spec)
        if spec.get("hotsel"):
            s, g, qsel = select_hotspot_query(grid0, hs, spec["seed"], spec["env"])
            base["query"] = "hot%s" % qsel
            if s is None:
                signal.alarm(0)
                return [dict(base, key=spec["key_prefix"] + "|" + m["label"] + "|" + m.get("w", "full"),
                             method=m["label"], wkey=m.get("w", "full"), success=False,
                             error="no hotspot-crossing query") for m in spec["methods"]]
        if spec.get("fpsel"):
            s, g, qsel = select_clear_query(grid0, spec["seed"], float(spec["inflate"]))
            base["query"] = "clr%s" % qsel
            if s is None:
                signal.alarm(0)
                return [dict(base, key=spec["key_prefix"] + "|" + m["label"] + "|" + m.get("w", "full"),
                             method=m["label"], wkey=m.get("w", "full"), rho_used=float(spec["inflate"]),
                             success=False, reason="no clearance-feasible query") for m in spec["methods"]]
        grid = grid0
        rho = float(spec.get("inflate") or 0.0)
        base.update({"start": list(s), "goal": list(g), "n_hot": len(hs)})
        field, mask = G.hotspot_reference(grid0, hs)
        wcache = {}
        c8cache = {}

        def W(wk):
            if wk not in wcache:
                t0 = time.perf_counter()
                wcache[wk] = (weight_for(grid, hs, spec["env"], wk), (time.perf_counter() - t0) * 1000)
            return wcache[wk]

        def C8(wk):
            if wk not in c8cache:
                c8cache[wk] = RP.search(grid, W(wk)[0], s, g, rho=rho)
            return c8cache[wk]

        blocked = rho > 0 and min(G.point_clearance(grid, *s), G.point_clearance(grid, *g)) < rho - 1e-12
        eval_w = W("full")[0]
        for m in spec["methods"]:
            label, wk = m["label"], m.get("w", "full")
            key = spec["key_prefix"] + "|" + label + "|" + wk
            rec = dict(base, key=key, method=label, wkey=wk, rho_used=rho)
            try:
                if blocked:
                    rec.update(success=False, reason="endpoint clearance below radius")
                    recs.append(rec)
                    done_labels.add(key)
                    continue
                w, tprep = W(wk)
                kw, factor = method_kwargs(label, spec["cfg"])
                if kw is None and rho > 0:
                    raise ValueError("JPS does not support footprint radius")
                reps = int(spec.get("reps") or 1)
                times = []
                if spec.get("engine") == "numba":
                    if rho > 0:
                        raise ValueError("compiled engine does not support footprint radius")
                    NB = _nb()
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        out = NB.run_nb(kw, grid, w, s, g)
                        t1 = time.perf_counter()
                        times.append((t1 - t0) * 1000)
                    r = RP.jps(grid, s, g) if kw is None else RP.search(grid, w, s, g, **kw)
                    rec["engine"] = "numba"
                    rec["nb_equal"] = _nb_equal(NB, out, r)
                else:
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        r = RP.jps(grid, s, g) if kw is None else RP.search(grid, w, s, g, rho=rho, **kw)
                        t1 = time.perf_counter()
                        times.append((t1 - t0) * 1000)
                if reps > 1:
                    rec["t_reps"] = times
                    rec["t_median"] = float(np.median(times))
                    rec["t_iqr"] = float(np.percentile(times, 75) - np.percentile(times, 25))
                rec.update(success=r["success"], reason=r["reason"], nodes=r["expanded"],
                           reopened=r["reopened"], los_calls=r["los_calls"],
                           t_search=times[0], t_prep=tprep)
                if r["success"]:
                    raw = r["path"]
                    t2 = time.perf_counter()
                    sm = G.smooth(grid, w, raw, rho=rho)
                    rec["t_smooth"] = (time.perf_counter() - t2) * 1000
                    probs = G.validate_path(grid, sm, s, g, rho=rho)
                    rec["valid"] = not probs
                    rec["problems"] = probs[:3]
                    raw_cost = G.path_cost(grid, w, raw, rho)
                    cost = G.path_cost(grid, w, sm, rho)
                    c8 = C8(wk)["cost"]
                    if spec["env"] == "movingai":
                        import rv_movingai as MA
                        rec["opt_official"] = MA.load_queries(spec["map"])[spec["seed"]]["opt"]
                    rec.update(raw_cost=raw_cost, cost=cost, c8=c8, ratio=cost / c8 if c8 > 0 else 1.0,
                               eval_cost=G.path_cost(grid, eval_w, sm), length=G.path_length(sm),
                               n_way=len(sm))
                    for th in G.TURN_THRESHOLDS:
                        rec["turns%g" % th] = G.turns(sm, th)
                    if kw is not None:
                        rec["g_consistent"] = raw_cost <= r["cost"] + TOL
                    rec["bound"] = factor
                    rec["bound_ok"] = None if factor is None else bool(cost <= factor * c8 + TOL and raw_cost <= factor * c8 + TOL)
                    if hs:
                        ex, cells = G.exposure(grid0, sm, field, mask)
                        rec.update(exposure=ex, hot_cells=cells)
                    if spec.get("inflate") or spec.get("clearance"):
                        prof = clearance_profile(grid0, sm)
                        tot = sum(l for _, l in prof) or 1.0
                        rec["min_clear"] = min(d for d, _ in prof)
                        for thr in (0.3, 0.5):
                            rec["frac_below%g" % thr] = sum(l for d, l in prof if d < thr) / tot
                    if spec.get("kin"):
                        import rv_kin
                        rec.update(rv_kin.profile(sm))
                        rec["path_sm"] = [[int(p[0]), int(p[1])] for p in sm]
                    if spec.get("uncertainty") and hs:
                        rec.update(_unc_eval(grid0, sm, hs, s, g, spec["seed"]))
            except InstanceTimeout:
                raise
            except Exception as ex:
                rec.update(success=False, error=repr(ex), tb=traceback.format_exc()[-800:])
            recs.append(rec)
            done_labels.add(key)
    except InstanceTimeout:
        pass
    except Exception as ex:
        err = repr(ex)
        tb = traceback.format_exc()[-800:]
        for m in spec["methods"]:
            key = spec["key_prefix"] + "|" + m["label"] + "|" + m.get("w", "full")
            if key not in done_labels:
                recs.append(dict(base, key=key, method=m["label"], wkey=m.get("w", "full"),
                                 success=False, error=err, tb=tb))
                done_labels.add(key)
    finally:
        signal.alarm(0)
    for m in spec["methods"]:
        key = spec["key_prefix"] + "|" + m["label"] + "|" + m.get("w", "full")
        if key not in done_labels:
            recs.append(dict(base, key=key, method=m["label"], wkey=m.get("w", "full"),
                             success=False, error="timeout"))
    return recs
