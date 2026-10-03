import json
import os
import re

import numpy as np

import rv_stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis")
CFG_LABEL = "Prop@1.3@0.05"


def _mean(t, m):
    v = [x[0] for x in t.get(m, {}).values()]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def frontier():
    recs = S.load("frontier") + S.load("frontier2")
    out = {}
    for env in sorted({r["env"] for r in recs}):
        rs = [r for r in recs if r["env"] == env]
        tn = S.table(rs, "nodes")
        tt = S.table(rs, "turns18")
        tr = S.table(rs, "ratio")
        pts = []
        for m in sorted(tn):
            pts.append({"method": m, "nodes": _mean(tn, m)[0], "turns18": _mean(tt, m)[0],
                        "ratio": _mean(tr, m)[0], "n": _mean(tn, m)[1]})
        for p in pts:
            p["dominated"] = any(q is not p and q["nodes"] <= p["nodes"] and q["turns18"] <= p["turns18"]
                                 and (q["nodes"] < p["nodes"] or q["turns18"] < p["turns18"]) for q in pts)
        ns = {p["n"] for p in pts}
        out[env] = {"points": pts, "n_values": sorted(ns)}
    return out


def density():
    recs = S.load("density")
    out = {}
    for d in sorted({str(r.get("density")) for r in recs}):
        rs = [r for r in recs if str(r.get("density")) == d]
        blk = {}
        fams = {}
        for metric in ("nodes", "ratio", "turns18"):
            t = S.table(rs, metric)
            fixed = sorted(m for m in t if m.startswith("PropFix@"))
            fams[metric] = S.family(t, "Proposed", [m for m in t if m != "Proposed"], metric)
            blk[metric] = {m: _mean(t, m)[0] for m in t}
            blk[metric + "_fixed"] = fixed
        tn = S.table(rs, "nodes")
        fixed = [m for m in tn if m.startswith("PropFix@")]
        oracle = min(fixed, key=lambda m: _mean(tn, m)[0]) if fixed else None
        blk["oracle_fixed"] = oracle
        blk["tests"] = fams
        blk["n_excluded"] = {m: r["excluded"] for r in fams["nodes"] for m in [r["other"]]}
        out[d] = blk
    return out


def hotspot():
    recs = S.load("hotspot")
    out = {"nominal": "Proposed|p:0.3:0.5:2.0", "lambda0": "Proposed|p:0.3:0:2.0"}
    for metric in ("exposure", "hot_cells", "nodes", "length", "cost", "exp_shift1", "exp_shift2",
                   "valid_add0.01", "valid_add0.03"):
        t = S.table(recs, metric)
        if out["nominal"] not in t:
            continue
        others = [m for m in t if m != out["nominal"]]
        out[metric] = {"summary": S.summary(t), "vs_nominal": S.family(t, out["nominal"], others, metric)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"frontier": frontier(), "density": density(), "hotspot": hotspot()}
    with open(os.path.join(OUT, "stages1.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    fr = res["frontier"]
    for env, blk in fr.items():
        print("== frontier", env, "n per method", blk["n_values"])
        nd = sorted((p for p in blk["points"] if not p["dominated"]), key=lambda p: p["nodes"])
        print("   nondominated:", [(p["method"], round(p["nodes"], 1), round(p["turns18"], 2)) for p in nd])
        sel = [p for p in blk["points"] if p["method"] == CFG_LABEL]
        print("   selected cfg:", [(p["method"], round(p["nodes"], 1), round(p["turns18"], 2), p["dominated"]) for p in sel])
    for d, blk in res["density"].items():
        n = blk["nodes"]
        print("== density", d, "oracle", blk["oracle_fixed"], "excluded", set(blk["n_excluded"].values()))
        for m in ("Traditional A*", "Theta*", "Weighted Theta*", "Proposed (fixed eps)", "Proposed", blk["oracle_fixed"]):
            if m in n:
                print("   %-22s nodes=%8.1f ratio=%.4f turns=%.2f" % (m, n[m], blk["ratio"].get(m, float("nan")),
                                                                     blk["turns18"].get(m, float("nan"))))
        for r in blk["tests"]["nodes"]:
            if r["other"] in ("Traditional A*", "Proposed (fixed eps)", "Weighted Theta*", blk["oracle_fixed"]):
                print("   P vs %-20s pct=%+6.1f p_holm=%.2g w_holm=%.2g" % (r["other"], r["pct_vs_b"], r["p_holm"], r["w_holm"]))
    hs = res["hotspot"]
    print("== hotspot")
    for metric in ("exposure", "hot_cells", "nodes", "length", "exp_shift1", "valid_add0.03"):
        if metric not in hs:
            continue
        for r in hs[metric]["vs_nominal"]:
            print("   %-12s nominal vs %-28s n=%d diff=%+.3f ci=[%.3g,%.3g] p_holm=%.2g w_holm=%.2g d=%+.2f excl=%d" % (
                metric, r["other"], r["n"], r["diff"], r["ci"][0], r["ci"][1], r["p_holm"], r["w_holm"], r["d"], r["excluded"]))


if __name__ == "__main__":
    main()
