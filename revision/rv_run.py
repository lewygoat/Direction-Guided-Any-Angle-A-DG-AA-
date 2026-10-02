import cProfile
import io
import json
import math
import multiprocessing as mp
import os
import platform
import pstats
import sys
import time
import traceback

import numpy as np

import rv_core as K
import rv_planners as RP
import rv_geom as G

RES = K.RES
STATUS = os.path.join(RES, "status.json")
ALERT = os.path.join(RES, "ALERT.txt")
CONFIG = os.path.join(RES, "selected_config.json")
N_PROC = 8
EPS_GRID = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 2.0]
BETA_GRID = [0.05, 0.1, 0.15, 0.2, 0.3]
W_MAX = 1.5
FRONT_EPS = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
FRONT_BETA = [0.1, 0.15, 0.2, 0.3]
MAIN_METHODS = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS",
                "Weighted Theta*", "Weighted Theta* (matched W)", "Proposed",
                "Proposed (fixed eps)", "Proposed (beta=0)", "Proposed (grid only)",
                "Proposed (no reopen)"]
BOUNDED = {"Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "Weighted Theta*",
           "Weighted Theta* (matched W)", "Proposed", "Proposed (fixed eps)",
           "Proposed (beta=0)", "Proposed (grid only)"}


def M(labels, w="full"):
    return [{"label": x, "w": w} for x in labels]


def spec(stage, env, seed, methods, cfg, **kw):
    s = {"stage": stage, "env": env, "seed": seed, "methods": methods, "cfg": cfg}
    s.update(kw)
    s["key_prefix"] = "%s|%s|%s|d%s|s%s|q%s|i%s|h%s" % (
        stage, env, seed, kw.get("density"), kw.get("size"), kw.get("query", "diag"),
        kw.get("inflate"), int(bool(kw.get("hotsel"))))
    if kw.get("map"):
        s["key_prefix"] += "|m" + kw["map"]
    return s


def stage_specs(stage, cfg):
    out = []
    if stage == "tune":
        labels = ["Traditional A*"] + ["Prop@%g@%g" % (e, b) for e in EPS_GRID for b in BETA_GRID]
        for d in (0.1, 0.2, 0.3, 0.4):
            for sd in range(9000, 9015):
                out.append(spec(stage, "general", sd, M(labels), cfg, density=d))
        for sd in range(9000, 9030):
            out.append(spec(stage, "warehouse", sd, M(labels), cfg))
    elif stage == "main":
        for env in ("general", "warehouse"):
            meth = M(MAIN_METHODS)
            if env == "warehouse":
                meth = meth + M(["Proposed"], "nohot") + M(["Proposed"], "noenv") + M(["Traditional A*", "Theta*"], "unit")
            for sd in range(100):
                for q in ["diag", 0, 1, 2, 3, 4]:
                    extra = {"gen": "legacy"} if env == "general" else {}
                    out.append(spec(stage, env, sd, meth, cfg, query=q, **extra))
    elif stage == "frontier":
        labels = ["Traditional A*"] + ["WTheta*@%g" % e for e in FRONT_EPS] + \
                 ["Prop@%g@%g" % (e, b) for e in FRONT_EPS for b in FRONT_BETA]
        for env in ("general", "warehouse"):
            for sd in range(100):
                extra = {"gen": "legacy"} if env == "general" else {}
                out.append(spec(stage, env, sd, M(labels), cfg, **extra))
    elif stage == "density":
        labels = ["Traditional A*", "Theta*", "Weighted Theta*", "Proposed", "Proposed (fixed eps)"] + \
                 ["PropFix@%g@%g" % (e, cfg["beta"]) for e in sorted(set([1.05] + EPS_GRID))]
        for d in (0.1, 0.2, 0.3, 0.4):
            for sd in range(100):
                out.append(spec(stage, "general", sd, M(labels), cfg, density=d))
        for sd in range(100, 200):
            out.append(spec(stage, "warehouse", sd, M(labels), cfg))
    elif stage == "hotspot":
        meth = []
        for lam in (0.0, 0.25, 0.5, 1.0):
            meth += M(["Proposed"], "p:0.3:%g:2.0" % lam)
        for sg in (1.0, 4.0):
            meth += M(["Proposed"], "p:0.3:0.5:%g" % sg)
        for a in (0.0, 0.6):
            meth += M(["Proposed"], "p:%g:0.5:2.0" % a)
        meth += M(["Theta*", "Traditional A*"], "p:0.3:0.0:2.0")
        for sd in range(2000, 2100):
            out.append(spec(stage, "warehouse", sd, meth, cfg, hotsel=True, uncertainty=True))
    elif stage == "footprint":
        labels = ["Traditional A*", "Theta*", "Weighted Theta*", "Proposed"]
        for env in ("general", "warehouse"):
            for sd in range(100):
                for rho in (None, 0.3, 0.5):
                    extra = {"gen": "legacy"} if env == "general" else {}
                    out.append(spec(stage, env, sd, M(labels), cfg, inflate=rho, clearance=True, **extra))
    elif stage == "scale":
        labels = ["Traditional A*", "Theta*", "JPS", "Weighted Theta*", "Proposed"]
        for S in (20, 50, 100, 150):
            for sd in range(30):
                out.append(spec(stage, "general", sd, M(labels), cfg, density=0.2, size=S, timeout=3600))
        for S in (50, 100, 150):
            for sd in range(30):
                out.append(spec(stage, "warehouse", sd, M(labels), cfg, size=S, timeout=3600))
    elif stage == "frontier2":
        labels = ["Prop@%g@%g" % (e, cfg["beta"]) for e in FRONT_EPS]
        for env in ("general", "warehouse"):
            for sd in range(100):
                extra = {"gen": "legacy"} if env == "general" else {}
                out.append(spec(stage, env, sd, M(labels), cfg, **extra))
    elif stage in ("timing_nb_main", "timing_nb_scale"):
        labels = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS",
                  "Weighted Theta*", "Proposed"]
        if stage == "timing_nb_main":
            for env in ("general", "warehouse"):
                for sd in range(100):
                    extra = {"gen": "legacy"} if env == "general" else {}
                    out.append(spec(stage, env, sd, M(labels), cfg, reps=10, engine="numba", **extra))
        else:
            for S in (20, 50, 100, 150):
                for sd in range(30):
                    out.append(spec(stage, "general", sd, M(labels), cfg, density=0.2, size=S,
                                    reps=10, engine="numba", timeout=7200))
            for S in (50, 100, 150):
                for sd in range(30):
                    out.append(spec(stage, "warehouse", sd, M(labels), cfg, size=S,
                                    reps=10, engine="numba", timeout=7200))
    elif stage == "bridge":
        import rv_bridge
        for env in ("general", "warehouse"):
            for sd in range(100):
                for q in ["diag", 0, 1, 2, 3, 4]:
                    extra = {"gen": "legacy"} if env == "general" else {}
                    out.append(spec(stage, env, sd, M(rv_bridge.LABELS), cfg, query=q, runner="bridge", **extra))
    elif stage in ("footprint_r1", "footprint_r1q"):
        labels = ["Traditional A*", "Theta*", "Weighted Theta*", "Proposed"]
        for env in ("general", "warehouse"):
            for sd in range(100):
                extra = {"gen": "legacy"} if env == "general" else {}
                out.append(spec(stage, env, sd, M(labels), cfg, inflate=1.0, clearance=True,
                                fpsel=(stage == "footprint_r1q"), **extra))
    elif stage == "timing_nb_movingai":
        import rv_movingai as MA
        labels = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS",
                  "Weighted Theta*", "Proposed"]
        for mp_ in MA.MAPS:
            for qi in range(len(MA.load_queries(mp_))):
                out.append(spec(stage, "movingai", qi, M(labels, "unit"), cfg, map=mp_,
                                reps=10, engine="numba", timeout=7200))
    elif stage == "kin":
        base_labels = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*",
                       "Weighted Theta*", "Weighted Theta* (matched W)", "Proposed", "Proposed (beta=0)"]
        for env in ("general", "warehouse"):
            for sd in range(100):
                for q in ["diag", 0, 1, 2, 3, 4]:
                    extra = {"gen": "legacy"} if env == "general" else {}
                    for rho in (None, 0.5):
                        labels = base_labels + (["JPS"] if rho is None else [])
                        out.append(spec(stage, env, sd, M(labels), cfg, query=q, inflate=rho,
                                        kin=True, clearance=True, **extra))
    elif stage == "movingai":
        import rv_movingai as MA
        labels = ["Traditional A*", "Weighted A*", "Theta*", "JPS", "Weighted Theta*",
                  "Weighted Theta* (matched W)", "Proposed", "Proposed (beta=0)"]
        for mp_ in MA.MAPS:
            for qi in range(len(MA.load_queries(mp_))):
                out.append(spec(stage, "movingai", qi, M(labels, "unit"), cfg, map=mp_, timeout=3600))
    elif stage in ("timing_main", "timing_scale"):
        labels = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS",
                  "Weighted Theta*", "Proposed"]
        if stage == "timing_main":
            for env in ("general", "warehouse"):
                for sd in range(100):
                    extra = {"gen": "legacy"} if env == "general" else {}
                    out.append(spec(stage, env, sd, M(labels), cfg, reps=10, **extra))
        else:
            for S in (20, 50, 100, 150):
                for sd in range(30):
                    out.append(spec(stage, "general", sd, M(labels), cfg, density=0.2, size=S, reps=10, timeout=7200))
            for S in (50, 100, 150):
                for sd in range(30):
                    out.append(spec(stage, "warehouse", sd, M(labels), cfg, size=S, reps=10, timeout=7200))
    else:
        raise KeyError(stage)
    return out


def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    good = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            good.append(line)
            done[r["key"]] = r
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(good) + ("\n" if good else ""))
    os.replace(path + ".tmp", path)
    return done


class Guard:
    def __init__(self):
        self.n = self.err = self.bound_bad = self.invalid = self.gc_bad = 0
        self.alerts = []

    def feed(self, r):
        self.n += 1
        if r.get("error"):
            self.err += 1
            if len(self.alerts) < 20:
                self.alerts.append("ERROR %s %s" % (r["key"], r["error"]))
        if "rho_used" in r and abs(float(r["rho_used"]) - float(r.get("inflate") or 0.0)) > 1e-12:
            self.gc_bad += 1
            self.alerts.append("RADIUS MISMATCH %s rho_used=%s" % (r["key"], r["rho_used"]))
        if r.get("invariant_fail"):
            self.gc_bad += 1
            self.alerts.append("BRIDGE INVARIANT %s %s" % (r["key"], r["invariant_fail"]))
        if r.get("nb_equal") is False:
            self.gc_bad += 1
            self.alerts.append("NUMBA MISMATCH %s" % r["key"])
        if r.get("success") and r.get("valid") is False:
            self.invalid += 1
            self.alerts.append("INVALID PATH %s %s" % (r["key"], r.get("problems")))
        if r.get("success") and r.get("g_consistent") is False:
            self.gc_bad += 1
            self.alerts.append("G-COST MISMATCH %s" % r["key"])
        if r.get("bound_ok") is False and (r["method"] in BOUNDED or "@" in r["method"]):
            self.bound_bad += 1
            self.alerts.append("BOUND VIOLATION %s cost=%.4f c8=%.4f W=%s" % (
                r["key"], r.get("cost", math.nan), r.get("c8", math.nan), r.get("bound")))

    def fatal(self):
        if self.invalid or self.bound_bad or self.gc_bad:
            return "invariant violated"
        if self.n >= 200 and self.err > 0.02 * self.n:
            return "error rate %.1f%%" % (100.0 * self.err / self.n)
        return None


def write_status(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    K.atomic_json(STATUS, kw)


def alert(msg):
    with open(ALERT, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print("ALERT " + msg, flush=True)


def run_stage(stage, cfg, stages, t_global):
    specs = stage_specs(stage, cfg)
    n_proc = 1 if stage.startswith("timing") else N_PROC
    path = os.path.join(RES, stage + ".jsonl")
    done = load_done(path)
    todo = []
    for s in specs:
        keys = [s["key_prefix"] + "|" + m["label"] + "|" + m["w"] for m in s["methods"]]
        if not all(k in done for k in keys):
            todo.append(s)
    guard = Guard()
    for r in done.values():
        guard.feed(r)
    total = len(specs)
    finished = total - len(todo)
    print("[%s] %d instances, %d already done" % (stage, total, finished), flush=True)
    t0 = time.time()
    n_new = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_proc, maxtasksperchild=50) as pool, open(path, "a") as f:
        for recs in pool.imap_unordered(K.dispatch, todo, chunksize=1):
            for r in recs:
                if r["key"] in done:
                    continue
                f.write(json.dumps(r, default=float) + "\n")
                done[r["key"]] = r
                guard.feed(r)
            f.flush()
            os.fsync(f.fileno())
            finished += 1
            n_new += 1
            el = time.time() - t0
            eta = el / n_new * (total - finished) if n_new else None
            write_status(state="running", stage=stage, stages=stages, done=finished, total=total, n_proc=n_proc,
                         records=len(done), errors=guard.err, invalid=guard.invalid,
                         bound_violations=guard.bound_bad, g_mismatch=guard.gc_bad,
                         stage_elapsed_s=round(el), stage_eta_s=None if eta is None else round(eta),
                         total_elapsed_s=round(time.time() - t_global), cfg=cfg,
                         recent_alerts=guard.alerts[-5:])
            why = guard.fatal()
            if why:
                for a in guard.alerts[-20:]:
                    alert(a)
                alert("stage %s halted: %s" % (stage, why))
                pool.terminate()
                return False
    for a in guard.alerts:
        if a.startswith("ERROR"):
            alert(a)
    print("[%s] finished in %.0fs, errors=%d" % (stage, time.time() - t0, guard.err), flush=True)
    return True


def select_config():
    path = os.path.join(RES, "tune.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    trad = {}
    for r in rows:
        if r["method"] == "Traditional A*" and r.get("success"):
            trad[r["key"].rsplit("|", 2)[0]] = r["nodes"]
    table = []
    for e in EPS_GRID:
        for b in BETA_GRID:
            lab = "Prop@%g@%g" % (e, b)
            nr, cr, tr = [], [], []
            fails = 0
            for r in rows:
                if r["method"] != lab:
                    continue
                pk = r["key"].rsplit("|", 2)[0]
                if not r.get("success") or pk not in trad:
                    fails += 1
                    continue
                nr.append(r["nodes"] / trad[pk])
                cr.append(r["ratio"])
                tr.append(r["turns18"])
            table.append({"eps_max": e, "beta": b, "W": (1.0 + 2.0 * b) * e, "node_ratio": float(np.mean(nr)),
                          "cost_ratio": float(np.mean(cr)), "turns18": float(np.mean(tr)),
                          "n": len(nr), "fails": fails})
    ok = [t for t in table if t["cost_ratio"] <= 1.02 and t["fails"] == 0 and t["W"] <= W_MAX + 1e-12]
    if not ok:
        raise RuntimeError("no tuning configuration satisfies the locked constraint")
    best = min(ok, key=lambda t: (round(t["node_ratio"], 4), t["turns18"]))
    cfg = {"eps_max": best["eps_max"], "beta": best["beta"]}
    K.atomic_json(CONFIG, {"selected": cfg, "rule": "min pooled node ratio s.t. cost ratio <= 1.02 and W <= %g; tie -> turns" % W_MAX,
                           "table": table})
    return cfg


def profile_stage(cfg):
    out_txt = os.path.join(RES, "profile.txt")
    out_json = os.path.join(RES, "profile.json")
    if os.path.exists(out_json):
        return True
    summary = {}
    buf = io.StringIO()
    for env in ("general", "warehouse"):
        for label in ("Traditional A*", "Theta*", "Proposed"):
            kw, _ = K.method_kwargs(label, cfg)
            pr = cProfile.Profile()
            tot = 0.0
            for sd in range(30):
                extra = {"gen": "legacy"} if env == "general" else {}
                grid, s, g, hs = K.build_map(dict({"env": env, "seed": sd}, **extra))
                w = K.weight_for(grid, hs, env, "full")
                t0 = time.perf_counter()
                pr.enable()
                RP.search(grid, w, s, g, **kw)
                pr.disable()
                tot += time.perf_counter() - t0
            st = pstats.Stats(pr)
            cats = {"segment_cost_LOS": 0.0, "step_cost": 0.0, "heuristic_key": 0.0, "heap": 0.0}
            for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
                if name in ("seg_cost", "walk"):
                    cats["segment_cost_LOS"] += tt
                elif name == "step_cost":
                    cats["step_cost"] += tt
                elif name == "key":
                    cats["heuristic_key"] += tt
                elif name in ("heappush", "heappop", "<built-in method _heapq.heappush>", "<built-in method _heapq.heappop>") or "heap" in name:
                    cats["heap"] += tt
            prof_total = sum(v[2] for v in st.stats.values())
            cats["other_search_loop"] = max(0.0, prof_total - sum(cats.values()))
            summary["%s|%s" % (env, label)] = {"wall_s_30maps": tot, "profiled_s": prof_total,
                                               "share": {k: v / prof_total for k, v in cats.items()}}
            buf.write("==== %s | %s ====\n" % (env, label))
            ps = pstats.Stats(pr, stream=buf)
            ps.sort_stats("tottime").print_stats(12)
    with open(out_txt, "w") as f:
        f.write(buf.getvalue())
    K.atomic_json(out_json, summary)
    return True


def main():
    os.makedirs(RES, exist_ok=True)
    stages = sys.argv[1:] or ["tune", "main", "frontier", "density", "hotspot", "footprint", "scale", "profile"]
    t_global = time.time()
    meta = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "cpu": os.cpu_count(), "n_proc": N_PROC, "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    K.atomic_json(os.path.join(RES, "provenance.json"), meta)
    cfg = {"eps_max": 1.5, "beta": 0.15}
    for st in stages:
        try:
            if st != "tune" and os.path.exists(CONFIG):
                cfg = json.load(open(CONFIG))["selected"]
            if st == "profile":
                write_status(state="running", stage="profile", stages=stages, cfg=cfg,
                             total_elapsed_s=round(time.time() - t_global))
                profile_stage(cfg)
                continue
            if st != "tune":
                if not os.path.exists(CONFIG):
                    raise RuntimeError("selected_config.json missing; run tune first")
                cfg = json.load(open(CONFIG))["selected"]
            ok = run_stage(st, cfg, stages, t_global)
            if not ok:
                write_status(state="halted", stage=st, stages=stages, cfg=cfg,
                             total_elapsed_s=round(time.time() - t_global))
                sys.exit(2)
            if st == "tune":
                cfg = select_config()
                print("selected config", cfg, flush=True)
        except SystemExit:
            raise
        except Exception:
            alert("runner crashed in stage %s:\n%s" % (st, traceback.format_exc()))
            write_status(state="crashed", stage=st, stages=stages, cfg=cfg,
                         total_elapsed_s=round(time.time() - t_global))
            sys.exit(3)
    write_status(state="finished", stages=stages, cfg=cfg, total_elapsed_s=round(time.time() - t_global))
    print("ALL STAGES FINISHED", flush=True)


if __name__ == "__main__":
    main()
