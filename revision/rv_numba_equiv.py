import json
import math
import sys
import time

import numpy as np

import rv_core as K
import rv_numba as NB
import rv_planners as RP

LABELS = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS",
          "Weighted Theta*", "Proposed", "Proposed (no reopen)"]


def compare(kw, grid, w, s, g):
    ref = RP.jps(grid, s, g) if kw is None else RP.search(grid, w, s, g, **kw)
    st, cost, ex, ro, los, path = NB.run_nb(kw, grid, w, s, g)
    bad = []
    if bool(st == NB.ST_OK) != bool(ref["success"]):
        bad.append("status %s vs %s" % (st, ref["success"]))
        return bad
    if ex != ref["expanded"]:
        bad.append("expanded %d vs %d" % (ex, ref["expanded"]))
    if ro != ref["reopened"]:
        bad.append("reopened %d vs %d" % (ro, ref["reopened"]))
    if los != ref["los_calls"]:
        bad.append("los %d vs %d" % (los, ref["los_calls"]))
    if ref["success"]:
        if abs(cost - ref["cost"]) > 1e-9 * max(1.0, ref["cost"]):
            bad.append("cost %.12f vs %.12f" % (cost, ref["cost"]))
        if [tuple(map(int, p)) for p in path] != [tuple(p) for p in ref["path"]]:
            bad.append("path differs")
    return bad


def main():
    cfg = json.load(open("results/selected_config.json"))["selected"]
    t0 = time.time()
    NB.warmup()
    print("jit warmup %.1fs" % (time.time() - t0))
    specs = []
    for env in ("general", "warehouse"):
        for sd in range(40):
            extra = {"gen": "legacy"} if env == "general" else {}
            for q in ("diag", 0, 1):
                specs.append(dict({"env": env, "seed": sd, "query": q}, **extra))
    for S in (100, 150):
        for sd in range(3):
            specs.append({"env": "general", "seed": sd, "density": 0.2, "size": S})
            specs.append({"env": "warehouse", "seed": sd, "size": S})
    for d in (0.1, 0.4):
        for sd in range(10):
            specs.append({"env": "general", "seed": sd, "density": d})
    rng = np.random.default_rng(11)
    n = 0
    fails = []
    for sp in specs:
        grid, s, g, hs = K.build_map(sp)
        w = K.weight_for(grid, hs, sp["env"], "full")
        for lab in LABELS:
            kw, _ = K.method_kwargs(lab, cfg)
            n += 1
            b = compare(kw, grid, w, s, g)
            if b:
                fails.append((sp, lab, b))
    for k in range(300):
        m = int(rng.integers(6, 16))
        grid = (rng.random((m, m)) < rng.uniform(0.15, 0.45)).astype(np.int8)
        w = 1.0 + rng.random((m, m)) * rng.choice([0.0, 1.0, 4.0])
        free = np.argwhere(grid == 0)
        if len(free) < 2:
            continue
        s = tuple(int(x) for x in free[0])
        g = tuple(int(x) for x in free[-1])
        for kw in ({}, {"anyangle": True, "eps": 2.0, "beta": 0.5, "adaptive": True},
                   {"anyangle": True, "eps": 1.3, "beta": 0.05, "adaptive": True, "reopen": False},
                   {"moves": RP.NB16}, None):
            n += 1
            b = compare(kw, grid, w if kw is not None else np.ones_like(w), s, g)
            if b:
                fails.append(({"random": k}, str(kw), b))
    print("compared %d runs, mismatches %d, %.1fs" % (n, len(fails), time.time() - t0))
    for f in fails[:15]:
        print("  ", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
