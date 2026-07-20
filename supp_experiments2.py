import json
import math
import numpy as np
from multiprocessing import Pool
from scipy import stats
import environments as E
import planners as P

EPS_GEN = 1.1
EPS_WH = 1.3


def _stat(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"p": 1.0, "d": 0.0}
    t, p = stats.ttest_rel(a, b)
    sd = diff.std(ddof=1)
    return {"p": float(p), "d": float(diff.mean() / sd if sd else 0.0)}


def _one_scale(args):
    seed, S = args
    grid, s, g, _ = E.make_general_size(seed, S, 0.20)
    ra = P.astar(grid, s, g)
    rp = P.proposed_v3(grid, s, g, warehouse=False, eps=EPS_GEN, beta=0.15)
    return S, ra["nodes"], rp["nodes"], ra["time"] if "time" in ra else 0.0


def scalability():
    import time
    sizes = [20, 50, 100, 150]
    NS = 15
    out = {}
    for S in sizes:
        an, pn, at, pt = [], [], [], []
        for seed in range(NS):
            grid, s, g, _ = E.make_general_size(seed, S, 0.20)
            t0 = time.perf_counter(); ra = P.astar(grid, s, g); t1 = time.perf_counter()
            t2 = time.perf_counter(); rp = P.proposed_v3(grid, s, g, warehouse=False, eps=EPS_GEN, beta=0.15); t3 = time.perf_counter()
            an.append(ra["nodes"]); pn.append(rp["nodes"])
            at.append((t1 - t0) * 1000); pt.append((t3 - t2) * 1000)
        out[S] = {
            "trad_nodes": float(np.mean(an)), "trad_nodes_sd": float(np.std(an, ddof=1)),
            "prop_nodes": float(np.mean(pn)), "prop_nodes_sd": float(np.std(pn, ddof=1)),
            "node_red": float((np.mean(an) - np.mean(pn)) / np.mean(an) * 100),
            "trad_time": float(np.mean(at)), "prop_time": float(np.mean(pt)),
            "time_red": float((np.mean(at) - np.mean(pt)) / np.mean(at) * 100),
            "n": NS,
        }
    return out


def _turns_at(path, thresh):
    if path is None or len(path) < 3:
        return 0
    t = 0
    for i in range(1, len(path) - 1):
        v1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        v2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        if math.degrees(math.acos(dot)) > thresh:
            t += 1
    return t


def turn_threshold():
    threshs = [13, 18, 23, 28]
    NS = 30
    trad = {th: [] for th in threshs}
    prop = {th: [] for th in threshs}
    for seed in range(NS):
        grid, s, g, _ = E.make_general_density(seed, 0.20)
        ra = P.astar(grid, s, g)
        rp = P.proposed_v3(grid, s, g, warehouse=False, eps=EPS_GEN, beta=0.15)
        for th in threshs:
            trad[th].append(_turns_at(ra["smoothed"], th))
            prop[th].append(_turns_at(rp["smoothed"], th))
    out = {}
    for th in threshs:
        a = np.array(trad[th], float); b = np.array(prop[th], float)
        out[th] = {
            "trad": float(a.mean()), "prop": float(b.mean()),
            "red": float((a.mean() - b.mean()) / a.mean() * 100) if a.mean() else 0.0,
            "stat": _stat(a, b),
        }
    return out


def hotspot_ablation():
    NS = 30
    on_nodes, off_nodes, on_turns, off_turns = [], [], [], []
    on_hot, off_hot = [], []
    for seed in range(NS):
        grid, s, g, hs = E.make_warehouse(seed)
        hot_set = set(hs) if hs else set()
        r_on = P.proposed_v3(grid, s, g, hs, warehouse=True, eps=EPS_WH, beta=0.15)
        r_off = P.proposed_v3(grid, s, g, None, warehouse=True, eps=EPS_WH, beta=0.15)
        on_nodes.append(r_on["nodes"]); off_nodes.append(r_off["nodes"])
        on_turns.append(r_on["turns"]); off_turns.append(r_off["turns"])
        on_hot.append(sum(1 for p in (r_on["smoothed"] or []) if tuple(p) in hot_set))
        off_hot.append(sum(1 for p in (r_off["smoothed"] or []) if tuple(p) in hot_set))
    return {
        "n": NS,
        "on_nodes": float(np.mean(on_nodes)), "off_nodes": float(np.mean(off_nodes)),
        "on_turns": float(np.mean(on_turns)), "off_turns": float(np.mean(off_turns)),
        "on_hotspot_hits": float(np.mean(on_hot)), "off_hotspot_hits": float(np.mean(off_hot)),
        "hotspot_hits_stat": _stat(np.array(off_hot, float), np.array(on_hot, float)),
    }


def main():
    res = {}
    print("scalability...")
    res["scalability"] = scalability()
    print("turn threshold...")
    res["turn_threshold"] = turn_threshold()
    print("hotspot ablation...")
    res["hotspot_ablation"] = hotspot_ablation()
    with open("supp_part2.json", "w") as f:
        json.dump(res, f, indent=2)
    print("done part2")


if __name__ == "__main__":
    main()
