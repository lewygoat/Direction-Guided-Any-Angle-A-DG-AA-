import json
import math
import numpy as np
from multiprocessing import Pool
from scipy import stats
import environments as E
import planners as P

N = 30
EPS_GEN = 1.1


def _stat(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "p": 1.0, "d": 0.0}
    t, p = stats.ttest_rel(a, b)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd else 0.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "p": float(p), "d": float(d)}


def _one_density(args):
    seed, ratio = args
    grid, s, g, _ = E.make_general_density(seed, ratio)
    ra = P.astar(grid, s, g)
    rp = P.proposed_v3(grid, s, g, warehouse=False, eps=EPS_GEN, beta=0.15)
    return ratio, ra["nodes"], rp["nodes"], ra["turns"], rp["turns"], ra["length"], rp["length"]


def robustness_density():
    ratios = [0.10, 0.20, 0.30, 0.40]
    jobs = [(s, r) for r in ratios for s in range(N)]
    with Pool(10) as pool:
        rows = pool.map(_one_density, jobs)
    out = {}
    for r in ratios:
        sub = [x for x in rows if x[0] == r]
        an = [x[1] for x in sub]; pn = [x[2] for x in sub]
        at = [x[3] for x in sub]; pt = [x[4] for x in sub]
        out[r] = {
            "trad_nodes": float(np.mean(an)), "prop_nodes": float(np.mean(pn)),
            "node_red": float((np.mean(an) - np.mean(pn)) / np.mean(an) * 100),
            "nodes_stat": _stat(an, pn),
            "trad_turns": float(np.mean(at)), "prop_turns": float(np.mean(pt)),
        }
    return out


def _one_multiquery(args):
    seed, q = args
    grid, _, _, _ = E.make_general_density(seed, 0.20)
    s, g = E.random_query(grid, seed * 100 + q)
    ra = P.astar(grid, s, g)
    rp = P.proposed_v3(grid, s, g, warehouse=False, eps=EPS_GEN, beta=0.15)
    return ra["nodes"], rp["nodes"], ra["turns"], rp["turns"]


def robustness_multiquery():
    jobs = [(s, q) for s in range(N) for q in range(5)]
    with Pool(10) as pool:
        rows = pool.map(_one_multiquery, jobs)
    an = [x[0] for x in rows]; pn = [x[1] for x in rows]
    at = [x[2] for x in rows]; pt = [x[3] for x in rows]
    return {
        "n_queries": len(rows),
        "trad_nodes": float(np.mean(an)), "prop_nodes": float(np.mean(pn)),
        "node_red": float((np.mean(an) - np.mean(pn)) / np.mean(an) * 100),
        "nodes_stat": _stat(an, pn),
        "trad_turns": float(np.mean(at)), "prop_turns": float(np.mean(pt)),
        "turns_stat": _stat(at, pt),
    }


def _one_eps(args):
    seed, eps = args
    grid, s, g, _ = E.make_general_density(seed, 0.20)
    rp = P.proposed_v3(grid, s, g, warehouse=False, eps=eps, beta=0.15)
    return eps, rp["nodes"], rp["turns"], rp["length"]


def sensitivity_eps():
    epss = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
    jobs = [(s, e) for e in epss for s in range(N)]
    with Pool(10) as pool:
        rows = pool.map(_one_eps, jobs)
    out = {}
    for e in epss:
        sub = [x for x in rows if abs(x[0] - e) < 1e-9]
        out[e] = {
            "nodes": float(np.mean([x[1] for x in sub])),
            "turns": float(np.mean([x[2] for x in sub])),
            "length": float(np.mean([x[3] for x in sub])),
            "bound": float(1 + 2 * 0.15) * e,
        }
    return out


def _one_beta(args):
    seed, beta = args
    grid, s, g, _ = E.make_general_density(seed, 0.20)
    rp = P.proposed_v3(grid, s, g, warehouse=False, eps=EPS_GEN, beta=beta)
    return beta, rp["nodes"], rp["turns"], rp["length"]


def sensitivity_beta():
    betas = [0.0, 0.1, 0.15, 0.2, 0.3, 0.5]
    jobs = [(s, b) for b in betas for s in range(N)]
    with Pool(10) as pool:
        rows = pool.map(_one_beta, jobs)
    out = {}
    for b in betas:
        sub = [x for x in rows if abs(x[0] - b) < 1e-9]
        out[b] = {
            "nodes": float(np.mean([x[1] for x in sub])),
            "turns": float(np.mean([x[2] for x in sub])),
            "length": float(np.mean([x[3] for x in sub])),
        }
    return out


def main():
    res = {}
    print("robustness density...")
    res["density"] = robustness_density()
    print("robustness multiquery...")
    res["multiquery"] = robustness_multiquery()
    print("sensitivity eps...")
    res["eps"] = sensitivity_eps()
    print("sensitivity beta...")
    res["beta"] = sensitivity_beta()
    with open("supp_part1.json", "w") as f:
        json.dump(res, f, indent=2)
    print("done part1")


if __name__ == "__main__":
    main()
