import json
import time
import math
import numpy as np
from multiprocessing import Pool
from scipy import stats

import environments as E
import planners as P

N_MAPS = 30
EPS_GEN = 1.1
EPS_WH = 1.3
METHODS = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS", "Proposed"]
METRICS = ["length", "nodes", "turns", "time"]


def _time_call(fn):
    t0 = time.perf_counter()
    r = fn()
    t1 = time.perf_counter()
    r["time"] = (t1 - t0) * 1000.0
    return r


def run_general(seed):
    grid, start, goal, _ = E.make_general(seed)
    out = {}
    out["Traditional A*"] = _time_call(lambda: P.astar(grid, start, goal))
    out["Weighted A*"] = _time_call(lambda: P.weighted_astar(grid, start, goal))
    out["16-neighbor A*"] = _time_call(lambda: P.astar16(grid, start, goal))
    out["Theta*"] = _time_call(lambda: P.theta_star(grid, start, goal))
    out["JPS"] = _time_call(lambda: P.jps(grid, start, goal))
    out["Proposed"] = _time_call(lambda: P.proposed_v3(grid, start, goal, warehouse=False, eps=EPS_GEN, beta=P.BETA))
    return {k: {m: v[m] for m in METRICS} for k, v in out.items()}


def run_warehouse(seed):
    grid, start, goal, hotspots = E.make_warehouse(seed)
    out = {}
    out["Traditional A*"] = _time_call(lambda: P.astar(grid, start, goal))
    out["Weighted A*"] = _time_call(lambda: P.weighted_astar(grid, start, goal))
    out["16-neighbor A*"] = _time_call(lambda: P.astar16(grid, start, goal))
    out["Theta*"] = _time_call(lambda: P.theta_star(grid, start, goal))
    out["JPS"] = _time_call(lambda: P.jps(grid, start, goal))
    out["Proposed"] = _time_call(lambda: P.proposed_v3(grid, start, goal, hotspots, warehouse=True, eps=EPS_WH, beta=P.BETA))
    return {k: {m: v[m] for m in METRICS} for k, v in out.items()}


def collect(runner):
    with Pool(processes=10) as pool:
        rows = pool.map(runner, list(range(N_MAPS)))
    data = {m: {met: np.array([row[m][met] for row in rows], dtype=float) for met in METRICS} for m in METHODS}
    return data


def cohen_d_paired(diff):
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return diff.mean() / sd


def ci95_meandiff(diff):
    n = len(diff)
    m = diff.mean()
    se = diff.std(ddof=1) / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return (m - tcrit * se, m + tcrit * se)


def compare(data, ref):
    res = {}
    for m in METHODS:
        if m == ref:
            continue
        res[m] = {}
        for met in METRICS:
            a = data[m][met]
            b = data[ref][met]
            diff = a - b
            if np.allclose(diff, 0):
                t, p = float("nan"), 1.0
                wp = 1.0
            else:
                t, p = stats.ttest_rel(a, b)
                try:
                    _, wp = stats.wilcoxon(a, b)
                except ValueError:
                    wp = float("nan")
            lo, hi = ci95_meandiff(diff)
            res[m][met] = {
                "mean_ref": float(b.mean()),
                "mean_other": float(a.mean()),
                "t": float(t),
                "p": float(p),
                "wilcoxon_p": float(wp),
                "cohen_d": float(cohen_d_paired(diff)),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
            }
    _holm(res)
    return res


def _holm(res):
    items = []
    for m in res:
        for met in res[m]:
            p = res[m][met]["p"]
            if not math.isnan(p):
                items.append((p, m, met))
    items.sort()
    k = len(items)
    for i, (p, m, met) in enumerate(items):
        adj = min(1.0, p * (k - i))
        res[m][met]["p_holm"] = float(adj)


def summarize(data):
    s = {}
    for m in METHODS:
        s[m] = {met: {"mean": float(data[m][met].mean()), "std": float(data[m][met].std(ddof=1))} for met in METRICS}
    return s


def improvement_vs(data, ref):
    imp = {}
    base = data[ref]
    for m in METHODS:
        imp[m] = {}
        for met in METRICS:
            bm = base[met].mean()
            om = data[m][met].mean()
            imp[m][met] = float((bm - om) / bm * 100.0) if bm != 0 else 0.0
    return imp


def main():
    print("Running general environment (n=%d)..." % N_MAPS)
    gen = collect(run_general)
    print("Running warehouse environment (n=%d)..." % N_MAPS)
    wh = collect(run_warehouse)

    result = {
        "config": {
            "n_maps": N_MAPS, "grid": "50x50", "general_obstacle_ratio": 0.20,
            "R_d": P.R_D, "rho_th": P.RHO_TH, "alpha": P.ALPHA, "beta": P.BETA,
            "lambda": P.LAMBDA, "sigma": P.SIGMA, "eps_general": EPS_GEN, "eps_warehouse": EPS_WH,
            "method": "proposed_v3 (any-angle + direction-guided weighted heuristic)",
            "weighted_eps": P.WEIGHTED_EPS, "turn_thresh_deg": P.TURN_THRESH_DEG,
        },
        "general": {
            "summary": summarize(gen),
            "improvement_vs_traditional": improvement_vs(gen, "Traditional A*"),
            "compare_vs_proposed": compare(gen, "Proposed"),
            "compare_vs_traditional": compare(gen, "Traditional A*"),
            "raw": {m: {met: gen[m][met].tolist() for met in METRICS} for m in METHODS},
        },
        "warehouse": {
            "summary": summarize(wh),
            "improvement_vs_traditional": improvement_vs(wh, "Traditional A*"),
            "compare_vs_proposed": compare(wh, "Proposed"),
            "compare_vs_traditional": compare(wh, "Traditional A*"),
            "raw": {m: {met: wh[m][met].tolist() for met in METRICS} for m in METHODS},
        },
    }

    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)

    write_txt(result)
    write_tables(result)
    print("Done. results.json, results.txt, tables.tex written.")


def _fmt(mean, std):
    return "%.2f±%.2f" % (mean, std)


def _fmt_tex(mean, std):
    return "$%.2f\\pm%.2f$" % (mean, std)


def write_txt(result):
    lines = []
    cfg = result["config"]
    lines.append("Reproduction of Improved A* Path Planning - Real Simulation Results")
    lines.append("=" * 72)
    lines.append("Config: %s" % json.dumps(cfg))
    lines.append("Paired design: all methods run on the SAME %d maps per environment." % cfg["n_maps"])
    lines.append("NOTE on JPS: optimizes pure distance only; no environment/hotspot")
    lines.append("weights and no turn cost, so its objective differs from Proposed.")
    lines.append("Nodes = pops from OPEN into CLOSED (JPS: jump-point expansions).")
    lines.append("")
    for env in ["general", "warehouse"]:
        block = result[env]
        lines.append("#" * 72)
        lines.append("ENVIRONMENT: %s" % env.upper())
        lines.append("#" * 72)
        lines.append("")
        lines.append("Mean +/- Std (n=%d)" % cfg["n_maps"])
        header = "%-16s %16s %14s %12s %12s" % ("Method", "Length", "Nodes", "Turns", "Time(ms)")
        lines.append(header)
        lines.append("-" * len(header))
        for m in METHODS:
            s = block["summary"][m]
            lines.append("%-16s %16s %14s %12s %12s" % (
                m, _fmt(s["length"]["mean"], s["length"]["std"]),
                _fmt(s["nodes"]["mean"], s["nodes"]["std"]),
                _fmt(s["turns"]["mean"], s["turns"]["std"]),
                _fmt(s["time"]["mean"], s["time"]["std"]),
            ))
        lines.append("")
        lines.append("Improvement vs Traditional A* (positive = better/lower):")
        for m in METHODS:
            if m == "Traditional A*":
                continue
            imp = block["improvement_vs_traditional"][m]
            lines.append("  %-16s length %+6.1f%%  nodes %+6.1f%%  turns %+6.1f%%  time %+6.1f%%" % (
                m, imp["length"], imp["nodes"], imp["turns"], imp["time"]))
        lines.append("")
        for ref_key, ref_name in [("compare_vs_proposed", "Proposed"), ("compare_vs_traditional", "Traditional A*")]:
            lines.append("Paired t-test of each method vs %s:" % ref_name)
            comp = block[ref_key]
            for m in METHODS:
                if m == ref_name or m not in comp:
                    continue
                lines.append("  %s:" % m)
                for met in METRICS:
                    c = comp[m][met]
                    lines.append("    %-7s p=%.4g d=%+.3f 95%%CI[%+.3f,%+.3f] (%s mean=%.3f, %s mean=%.3f)" % (
                        met, c["p"], c["cohen_d"], c["ci95_lo"], c["ci95_hi"],
                        m, c["mean_other"], ref_name, c["mean_ref"]))
            lines.append("")
    with open("results.txt", "w") as f:
        f.write("\n".join(lines))


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def write_tables(result):
    out = []
    out.append("% Auto-generated by run_experiments.py - real simulation output")
    out.append("% Significance vs Proposed (paired t-test): * p<0.05, ** p<0.01, *** p<0.001")
    for env, label, tabnum in [("general", "General environment (50$\\times$50, 20\\% obstacles)", "2"),
                               ("warehouse", "Warehouse environment (50$\\times$50)", "3")]:
        block = result[env]
        comp = block["compare_vs_proposed"]
        out.append("")
        out.append("\\begin{table}[htbp]")
        out.append("\\centering")
        out.append("\\caption{%s, mean$\\pm$std over %d maps. Replaces Table %s.}" % (label, result["config"]["n_maps"], tabnum))
        out.append("\\label{tab:repro_%s}" % env)
        out.append("\\begin{tabular}{lcccc}")
        out.append("\\toprule")
        out.append("Method & Path length & Nodes & Turns & Time (ms) \\\\")
        out.append("\\midrule")
        for m in METHODS:
            s = block["summary"][m]
            if m == "Proposed":
                cells = [_fmt_tex(s[x]["mean"], s[x]["std"]) for x in METRICS]
            else:
                cells = []
                for x in METRICS:
                    star = _sig(comp[m][x]["p"]) if m in comp else ""
                    cells.append(_fmt_tex(s[x]["mean"], s[x]["std"]) + star)
            name = "\\textbf{Proposed}" if m == "Proposed" else m.replace("*", "$^{*}$").replace("A$^{*}$", "A*")
            name = m.replace("*", "$^{\\ast}$")
            if m == "Proposed":
                name = "\\textbf{Proposed}"
            out.append("%s & %s & %s & %s & %s \\\\" % (name, cells[0], cells[1], cells[2], cells[3]))
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")

    out.append("")
    out.append("\\begin{table}[htbp]")
    out.append("\\centering")
    out.append("\\caption{Competitor vs Proposed: paired $t$-test on the general environment (n=%d). $\\Delta$ is competitor mean minus Proposed mean.}" % result["config"]["n_maps"])
    out.append("\\label{tab:repro_stats}")
    out.append("\\begin{tabular}{llrrr}")
    out.append("\\toprule")
    out.append("Method & Metric & $\\Delta$ & $p$ & Cohen's $d$ \\\\")
    out.append("\\midrule")
    comp = result["general"]["compare_vs_proposed"]
    for m in METHODS:
        if m == "Proposed":
            continue
        for met in METRICS:
            c = comp[m][met]
            delta = c["mean_other"] - c["mean_ref"]
            nm = m if met == "length" else ""
            out.append("%s & %s & %+.2f & %.3g & %+.2f \\\\" % (nm.replace("*", "$^{\\ast}$"), met, delta, c["p"], c["cohen_d"]))
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    out.append("\\end{table}")

    with open("tables.tex", "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
