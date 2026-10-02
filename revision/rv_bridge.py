import math
import signal
import traceback

import rv_core as K
import rv_geom as G
import rv_planners as RP

P0 = G.P0
ENV_EPS = {"general": 1.1, "warehouse": 1.3}
BETA0 = 0.15
TOL = 1e-6

B0 = "B0 Traditional A* (original code)"
B1 = "B1 Traditional A* (revised)"
L0 = "L0 original code"
L1 = "L1 original semantics re-implemented"
L2 = "L2 + exact collision"
L3 = "L3 + integrated cost"
L4 = "L4 + cost-based parent"
L5 = "L5 + reopening"
L6 = "L6 + cost-monotone smoothing"
L7 = "L7 + adaptive eps"
L8 = "L8 + selected config (= Proposed)"
LABELS = [B0, B1, L0, L1, L2, L3, L4, L5, L6, L7, L8]
SMOOTH = {B1: "cost", L1: "orig", L2: "los", L3: "los", L4: "los", L5: "los",
          L6: "cost", L7: "cost", L8: "cost"}


def step_kwargs(label, env, cfg):
    e0 = ENV_EPS[env]
    w0 = (1.0 + 2.0 * BETA0) * e0
    base = {"anyangle": True, "eps": e0, "beta": BETA0}
    if label == B1:
        return {}, 1.0
    if label == L1:
        return dict(base, collision="bresenham", cost_mode="endpoint", parent_rule="greedy", reopen=False), None
    if label == L2:
        return dict(base, cost_mode="endpoint", parent_rule="greedy", reopen=False), None
    if label == L3:
        return dict(base, parent_rule="greedy", reopen=False), None
    if label == L4:
        return dict(base, reopen=False), None
    if label in (L5, L6):
        return dict(base), w0
    if label == L7:
        return dict(base, adaptive=True), w0
    if label == L8:
        kw, wf = K.method_kwargs("Proposed", cfg)
        return dict(kw), wf
    raise KeyError(label)


def _close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def _evaluate(label, grid, w, s, g, hs, env, cfg, c8, field, mask):
    out = {}
    fails = []
    greported = None
    wf = None
    if label == B0:
        r0 = P0.astar(grid, s, g)
        raw, sm = r0["path"], r0["smoothed"]
        out["nodes"] = r0["nodes"]
    elif label == L0:
        r0 = P0.proposed_v3(grid, s, g, hs if env == "warehouse" else None,
                            warehouse=(env == "warehouse"), eps=ENV_EPS[env], beta=BETA0)
        raw, sm = r0["path"], r0["smoothed"]
        out["nodes"] = r0["nodes"]
    else:
        kw, wf = step_kwargs(label, env, cfg)
        r = RP.search(grid, w, s, g, **kw)
        raw = r["path"]
        greported = r["cost"] if r["success"] else None
        out.update(nodes=r["expanded"], reopened=r["reopened"], los_calls=r["los_calls"],
                   g_reported=greported)
        sm = None
        if raw:
            mode = SMOOTH[label]
            if mode == "orig":
                sm = P0.smooth_path(grid, raw)
            elif mode == "los":
                sm = G.smooth_los(grid, raw)
            else:
                sm = G.smooth(grid, w, raw)
    out["success"] = bool(raw)
    if not raw:
        out["reason"] = "no path"
        return out
    raw = [(int(p[0]), int(p[1])) for p in raw]
    sm = [(int(p[0]), int(p[1])) for p in sm]
    if raw[0] != s or raw[-1] != g or sm[0] != s or sm[-1] != g:
        fails.append("path endpoints differ from query")
    raw_exact = not G.validate_path(grid, raw, s, g)
    sm_exact = not G.validate_path(grid, sm, s, g)
    model_cost = math.fsum(0.5 * math.hypot(b[0] - a[0], b[1] - a[1]) * (w[a] + w[b])
                           for a, b in zip(raw, raw[1:]))
    raw_int = G.path_cost(grid, w, raw) if raw_exact else None
    sm_int = G.path_cost(grid, w, sm) if sm_exact else None
    out.update(raw_valid_exact=raw_exact, sm_valid_exact=sm_exact, model_cost=model_cost,
               raw_cost=raw_int, cost=sm_int, c8=c8, bound=wf,
               ratio_raw=None if raw_int is None else raw_int / c8,
               ratio=None if sm_int is None else sm_int / c8,
               length=G.path_length(sm), raw_length=G.path_length(raw), n_way=len(sm))
    if raw_int is not None:
        out["underestimate"] = raw_int - model_cost
    for th in G.TURN_THRESHOLDS:
        out["turns%g" % th] = G.turns(sm, th)
    if hs and sm_exact:
        ex, cells = G.exposure(grid, sm, field, mask)
        out.update(exposure=ex, hot_cells=cells)
    if label in (L0, L1):
        out["raw_valid_model"] = all(G.bres_ok(grid, w, a, b) for a, b in zip(raw, raw[1:]))
    if label == L1 and not out["raw_valid_model"]:
        fails.append("L1 raw path violates its own collision model")
    if label in (L1, L2) and not _close(greported, model_cost):
        fails.append("g differs from endpoint chain cost")
    if label not in (B0, L0, L1):
        if not raw_exact:
            fails.append("raw path collides")
        if not sm_exact:
            fails.append("smoothed path collides")
    if label in (L3, L4) and raw_int is not None and not _close(greported, raw_int):
        fails.append("g differs from integrated chain cost")
    if label in (B1, L5, L6, L7, L8) and raw_int is not None and raw_int > greported + TOL * max(1.0, greported):
        fails.append("integrated chain cost above g")
    if wf is not None and raw_int is not None and raw_int > wf * c8 + TOL * max(1.0, c8):
        fails.append("raw bound violated")
    if label in (B1, L6, L7, L8) and sm_int is not None and raw_int is not None:
        if sm_int > raw_int + TOL * max(1.0, raw_int):
            fails.append("smoothing raised cost")
        if wf is not None and sm_int > wf * c8 + TOL * max(1.0, c8):
            fails.append("smoothed bound violated")
    if fails:
        out["invariant_fail"] = fails
    return out


def run_bridge(spec):
    signal.signal(signal.SIGALRM, K._alarm)
    signal.alarm(int(spec.get("timeout", 1800)))
    env = spec["env"]
    base = {k: spec.get(k) for k in ("stage", "env", "seed", "query")}
    recs = []
    done = set()
    try:
        grid, s, g, hs = K.build_map(spec)
        w = K.weight_for(grid, hs, env, "full")
        field, mask = G.hotspot_reference(grid, hs)
        c8 = RP.search(grid, w, s, g)["cost"]
        base.update(start=list(s), goal=list(g), n_hot=len(hs))
        for m in spec["methods"]:
            label = m["label"]
            key = spec["key_prefix"] + "|" + label + "|" + m["w"]
            rec = dict(base, key=key, method=label, wkey=m["w"])
            try:
                rec.update(_evaluate(label, grid, w, s, g, hs, env, spec["cfg"], c8, field, mask))
            except K.InstanceTimeout:
                raise
            except Exception as ex:
                rec.update(success=False, error=repr(ex), tb=traceback.format_exc()[-800:])
            recs.append(rec)
            done.add(key)
    except K.InstanceTimeout:
        pass
    except Exception as ex:
        err = repr(ex)
        tb = traceback.format_exc()[-800:]
        for m in spec["methods"]:
            key = spec["key_prefix"] + "|" + m["label"] + "|" + m["w"]
            if key not in done:
                recs.append(dict(base, key=key, method=m["label"], wkey=m["w"], success=False, error=err, tb=tb))
                done.add(key)
    finally:
        signal.alarm(0)
    for m in spec["methods"]:
        key = spec["key_prefix"] + "|" + m["label"] + "|" + m["w"]
        if key not in done:
            recs.append(dict(base, key=key, method=m["label"], wkey=m["w"], success=False, error="timeout"))
    return recs
