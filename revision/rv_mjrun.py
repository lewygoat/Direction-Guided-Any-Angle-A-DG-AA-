import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback

import rv_core as K
import rv_geom as G
import rv_planners as RP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "mjsim.jsonl")
STATUS = os.path.join(RES, "mjsim_status.json")
ALERT = os.path.join(RES, "ALERT.txt")
LOG = os.path.join(RES, "run.log")
TOL = 1e-9
N_SEEDS = 5
METHODS_S0 = ["Traditional A*", "Weighted A*", "Theta*", "Weighted Theta*", "Proposed", "JPS"]
METHODS_S05 = ["Traditional A*", "Weighted A*", "Theta*", "Weighted Theta*", "Proposed"]
METHODS_S10 = ["Traditional A*", "Theta*", "Weighted Theta*", "Proposed"]
WALL_TIMEOUT = 600


class RunTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise RunTimeout()


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def alert(msg):
    with open(ALERT, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    log("ALERT " + msg)


def write_status(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=1)
    os.replace(tmp, STATUS)


def load_cfg():
    return json.load(open(os.path.join(RES, "selected_config.json")))["selected"]


def collect_sources(queries):
    tasks = []
    for ln in open(os.path.join(RES, "kin.jsonl")):
        r = json.loads(ln)
        if r.get("env") != "warehouse" or not r.get("success"):
            continue
        if str(r.get("query")) not in queries:
            continue
        rho = float(r.get("rho_used") or 0.0)
        if rho == 0.0 and r["method"] in METHODS_S0:
            tasks.append({"set": "S0", "r_body": 0.5, "src": r})
        elif rho == 0.5 and r["method"] in METHODS_S05:
            tasks.append({"set": "S05", "r_body": 0.5, "src": r})
    for ln in open(os.path.join(RES, "footprint_r1q.jsonl")):
        r = json.loads(ln)
        if r.get("env") == "warehouse" and r.get("success") and r["method"] in METHODS_S10:
            tasks.append({"set": "S10", "r_body": 1.0, "src": r})
    return tasks


def replan_s10(src, cfg):
    spec = {"env": "warehouse", "seed": src["seed"]}
    grid, s, g, hs = K.build_map(spec)
    s2, g2, qsel = K.select_clear_query(grid, src["seed"], 1.0)
    if s2 is None or list(s2) != list(src["start"]) or list(g2) != list(src["goal"]):
        raise ValueError("S10 query mismatch")
    if "clr%s" % qsel != src["query"]:
        raise ValueError("S10 query id mismatch")
    w = K.weight_for(grid, hs, "warehouse", src.get("wkey") or "full")
    kw, _ = K.method_kwargs(src["method"], cfg)
    r = RP.search(grid, w, tuple(s2), tuple(g2), rho=1.0, **kw)
    if not r["success"]:
        raise ValueError("S10 replan failed")
    sm = G.smooth(grid, w, r["path"], rho=1.0)
    L = G.path_length(sm)
    c = G.path_cost(grid, w, sm, 1.0)
    if abs(L - src["length"]) > TOL or abs(c - src["cost"]) > TOL:
        raise ValueError("S10 replan differs from record: L %.12f vs %.12f, c %.12f vs %.12f"
                         % (L, src["length"], c, src["cost"]))
    return grid, sm


def prepare(task, cfg):
    src = task["src"]
    if task["set"] == "S10":
        return replan_s10(src, cfg)
    q = src.get("query")
    spec = {"env": "warehouse", "seed": src["seed"], "query": q if q == "diag" else int(q)}
    grid, s, g, hs = K.build_map(spec)
    if list(s) != list(src["start"]) or list(g) != list(src["goal"]):
        raise ValueError("endpoint mismatch with source record")
    path = [tuple(p) for p in src["path_sm"]]
    if tuple(path[0]) != tuple(s) or tuple(path[-1]) != tuple(g):
        raise ValueError("stored path endpoints differ from query")
    if abs(G.path_length(path) - src["length"]) > TOL:
        raise ValueError("stored path length differs from record")
    return grid, path


def run_task(args):
    task, cfg = args
    import rv_mjsim as S
    src = task["src"]
    base = {"set": task["set"], "r_body": task["r_body"], "src_key": src["key"], "method": src["method"],
            "seed_map": src["seed"], "query": src.get("query"), "rho_plan": float(src.get("rho_used") or 0.0),
            "plan_length": src["length"]}
    out = []
    try:
        grid, path = prepare(task, cfg)
    except Exception as ex:
        return [dict(base, k=k, key="%s|%s|%d" % (task["set"], src["key"], k), error="prep: " + repr(ex),
                     fatal=True) for k in range(N_SEEDS)]
    for k in range(N_SEEDS):
        key = "%s|%s|%d" % (task["set"], src["key"], k)
        rec = dict(base, k=k, key=key)
        sd = S.seed_for(src["key"], k)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(WALL_TIMEOUT)
        try:
            t0 = time.perf_counter()
            r = S.simulate(grid, path, task["r_body"], seed=sd)
            rec["wall_s"] = time.perf_counter() - t0
            rec.update(r)
            if k == 0:
                r2 = S.simulate(grid, path, task["r_body"], seed=sd)
                rec["repeat_equal"] = (json.dumps(r, sort_keys=True) == json.dumps(r2, sort_keys=True))
        except RunTimeout:
            rec["error"] = "wall timeout"
        except Exception as ex:
            rec["error"] = repr(ex)
            rec["tb"] = traceback.format_exc()[-600:]
        finally:
            signal.alarm(0)
        out.append(rec)
    return out


def done_keys():
    s = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            try:
                s.add(json.loads(ln)["key"])
            except Exception:
                pass
    return s


def main(queries, n_proc, limit=None, out_path=None):
    global OUT
    if out_path:
        OUT = out_path
    cfg = load_cfg()
    tasks = collect_sources(queries)
    if limit:
        tasks = tasks[:limit]
    have = done_keys()
    todo = [t for t in tasks if any("%s|%s|%d" % (t["set"], t["src"]["key"], k) not in have for k in range(N_SEEDS))]
    total = len(tasks)
    log("[mjsim] %d paths (%d todo), queries=%s, seeds=%d" % (total, len(todo), sorted(queries), N_SEEDS))
    cnt = {"errors": 0, "fatal": 0, "nonrepeat": 0, "diverged": 0}
    done = total - len(todo)
    t0 = time.time()
    halted = None
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc, maxtasksperchild=20) as pool, open(OUT, "a") as f:
        for recs in pool.imap_unordered(run_task, [(t, cfg) for t in todo], chunksize=1):
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
                if r.get("error"):
                    cnt["errors"] += 1
                    if r["error"] in ("diverged", "badqacc"):
                        cnt["diverged"] += 1
                if r.get("fatal"):
                    cnt["fatal"] += 1
                if r.get("k") == 0 and r.get("repeat_equal") is False:
                    cnt["nonrepeat"] += 1
            f.flush()
            done += 1
            el = time.time() - t0
            rate = (done - (total - len(todo))) / el if el > 0 else 0
            write_status(state="running", stage="mjsim", done=done, total=total, n_proc=n_proc,
                         eta_s=int((total - done) / rate) if rate else None, elapsed_s=int(el), **cnt)
            if cnt["fatal"] or cnt["nonrepeat"]:
                halted = "fatal=%d nonrepeat=%d" % (cnt["fatal"], cnt["nonrepeat"])
                alert("mjsim halted: " + halted + " last=" + recs[0]["key"])
                pool.terminate()
                break
    write_status(state="halted" if halted else "finished", stage="mjsim", done=done, total=total,
                 n_proc=n_proc, elapsed_s=int(time.time() - t0), **cnt)
    log("[mjsim] %s in %ds %s" % ("HALTED" if halted else "finished", time.time() - t0, cnt))


if __name__ == "__main__":
    qs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["diag", "0", "1", "2", "3", "4"]
    npr = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    op = sys.argv[4] if len(sys.argv) > 4 else None
    main(set(qs), npr, lim, op)
