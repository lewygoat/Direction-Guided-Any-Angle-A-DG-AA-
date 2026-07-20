import json
import numpy as np
import environments as E
import planners as P

N = 30
EPS_WH = 1.3


def _run(seed, **kw):
    grid, s, g, hs = E.make_warehouse(seed)
    hot_set = set(hs) if hs else set()
    r = P.proposed_v3(grid, s, g, hs, warehouse=True, eps=EPS_WH, beta=0.15, **kw)
    hits = sum(1 for p in (r["smoothed"] or []) if tuple(p) in hot_set)
    return r["nodes"], r["turns"], r["length"], hits


def sweep(param, values, fixed):
    out = {}
    for val in values:
        kw = dict(fixed)
        kw[param] = val
        rows = [_run(seed, **kw) for seed in range(N)]
        arr = np.array(rows, float)
        out[val] = {
            "nodes": float(arr[:, 0].mean()),
            "turns": float(arr[:, 1].mean()),
            "length": float(arr[:, 2].mean()),
            "hotspot_hits": float(arr[:, 3].mean()),
        }
    return out


def main():
    res = {}
    print("alpha (warehouse env weight)...")
    res["alpha"] = sweep("alpha", [0.0, 0.15, 0.3, 0.45, 0.6], {"lam": P.LAMBDA, "sigma": P.SIGMA})
    print("lambda (hotspot penalty)...")
    res["lambda"] = sweep("lam", [0.0, 0.25, 0.5, 0.75, 1.0], {"alpha": P.ALPHA, "sigma": P.SIGMA})
    print("sigma (hotspot radius)...")
    res["sigma"] = sweep("sigma", [1.0, 2.0, 3.0, 4.0], {"alpha": P.ALPHA, "lam": P.LAMBDA})
    with open("supp_part3.json", "w") as f:
        json.dump(res, f, indent=2)
    print("done part3")


if __name__ == "__main__":
    main()
