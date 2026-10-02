import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_BOOT = 10000
BOOT_SEED = 12345
ALPHA = 0.05


def load(stage, path=None):
    fp = path or os.path.join(RES, stage + ".jsonl")
    out = []
    with open(fp) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def inst_id(r):
    return (r.get("env"), r.get("seed"), str(r.get("query")), r.get("density"), r.get("size"),
            r.get("inflate"), r.get("map"))


def cluster_id(r):
    return (r.get("env"), r.get("map"), r.get("seed") if r.get("env") != "movingai" else None,
            r.get("density"), r.get("size"))


def label(r):
    wk = r.get("wkey") or "full"
    return r["method"] if wk == "full" else "%s|%s" % (r["method"], wk)


def table(recs, metric, method_key=label, ok=lambda r: r.get("success")):
    t = {}
    for r in recs:
        if not ok(r):
            continue
        v = r.get(metric)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        m = method_key(r) if callable(method_key) else r[method_key]
        k = inst_id(r)
        cell = t.setdefault(m, {})
        if k in cell:
            raise ValueError("duplicate record for %s %r" % (m, k))
        cell[k] = (float(v), cluster_id(r))
    return t


def pair(t, a, b):
    ta, tb = t.get(a, {}), t.get(b, {})
    keys = sorted(set(ta) & set(tb), key=repr)
    x = np.array([ta[k][0] for k in keys])
    y = np.array([tb[k][0] for k in keys])
    cl = [ta[k][1] for k in keys]
    excl = len(set(ta) ^ set(tb))
    return x, y, cl, excl


def cluster_boot_ci(diff, clusters, n=N_BOOT, seed=BOOT_SEED):
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    ids = {}
    for i, c in enumerate(clusters):
        ids.setdefault(repr(c), []).append(i)
    groups = list(ids.values())
    sums = np.array([diff[g].sum() for g in groups])
    cnts = np.array([len(g) for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    G = len(groups)
    idx = rng.integers(0, G, size=(n, G))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_test(x, y, clusters):
    d = x - y
    n = len(d)
    res = {"n": n, "mean_a": float(x.mean()) if n else float("nan"),
           "mean_b": float(y.mean()) if n else float("nan"),
           "diff": float(d.mean()) if n else float("nan")}
    if n < 2 or np.allclose(d, 0):
        res.update(t_p=1.0, w_p=1.0, d=0.0)
    else:
        res["t_p"] = float(stats.ttest_rel(x, y).pvalue)
        try:
            res["w_p"] = float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            res["w_p"] = 1.0
        sd = d.std(ddof=1)
        res["d"] = float(d.mean() / sd) if sd > 0 else 0.0
    res["ci"] = cluster_boot_ci(d, clusters)
    mb = res["mean_b"]
    res["pct_vs_b"] = float(100.0 * (mb - res["mean_a"]) / mb) if n and mb != 0 else float("nan")
    return res


def holm(results, key="t_p", out="p_holm"):
    items = sorted((r[key], i) for i, r in enumerate(results) if r.get(key) is not None and math.isfinite(r[key]))
    k = len(items)
    run = 0.0
    for rank, (p, i) in enumerate(items):
        run = max(run, min(1.0, p * (k - rank)))
        results[i][out] = run
    return results


def family(t, ref, others, metric_name):
    rows = []
    for m in others:
        if m == ref or m not in t:
            continue
        x, y, cl, excl = pair(t, ref, m)
        r = paired_test(x, y, cl)
        r.update(ref=ref, other=m, metric=metric_name, excluded=excl)
        rows.append(r)
    holm(rows, "t_p", "p_holm")
    holm(rows, "w_p", "w_holm")
    return rows


def summary(t):
    s = {}
    for m, d in t.items():
        v = np.array([x[0] for x in d.values()])
        s[m] = {"n": len(v), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "median": float(np.median(v))}
    return s


def self_test():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    cl = [i // 4 for i in range(200)]
    lo, hi = cluster_boot_ci(x, cl, n=2000)
    assert lo < x.mean() < hi
    rows = [{"t_p": 0.01}, {"t_p": 0.04}, {"t_p": 0.03}]
    holm(rows)
    assert abs(rows[0]["p_holm"] - 0.03) < 1e-12
    assert abs(rows[2]["p_holm"] - 0.06) < 1e-12 and abs(rows[1]["p_holm"] - 0.06) < 1e-12
    a = np.arange(10.0)
    r = paired_test(a + 1, a, list(range(10)))
    assert r["diff"] == 1.0 and r["n"] == 10
    return True


if __name__ == "__main__":
    print("rv_stats self test", self_test())
