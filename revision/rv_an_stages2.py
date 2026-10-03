import json
import math
import os
import sys

import numpy as np

import rv_stats as S
import rv_kin as K

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis")
RES = os.path.join(HERE, "results")


def stars(p):
    if p is None or not math.isfinite(p):
        return ""
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def bold_best(vals, fmt, higher_is_better=False):
    """Return list of formatted strings; mark best bold, second underline."""
    finite = [(i, v) for i, v in enumerate(vals) if v is not None and math.isfinite(v)]
    if not finite:
        return ["--"] * len(vals)
    finite_sorted = sorted(finite, key=lambda x: x[1], reverse=higher_is_better)
    best_i = finite_sorted[0][0]
    second_i = finite_sorted[1][0] if len(finite_sorted) > 1 else None
    out = []
    for i, v in enumerate(vals):
        if v is None or not math.isfinite(v):
            out.append("--")
            continue
        s = fmt(v)
        if i == best_i:
            out.append("\\mathbf{%s}" % s)
        elif i == second_i:
            out.append("\\underline{%s}" % s)
        else:
            out.append(s)
    return out


# ── scale analysis ────────────────────────────────────────────────────────────

def analyze_scale():
    recs = S.load("scale")
    trecs = S.load("timing_nb_scale")
    bad = [r for r in trecs if not r.get("nb_equal", True)]
    if bad:
        sys.exit("timing_nb_scale nb_equal false: %d" % bad)

    methods = ["Traditional A*", "Theta*", "Weighted Theta*", "JPS", "Proposed"]
    sizes = [20, 50, 100, 150]
    envs = ["general", "warehouse"]

    rows = []
    for env in envs:
        env_sizes = sizes if env == "general" else [50, 100, 150]
        for sz in env_sizes:
            block = {}
            for r in recs:
                if r.get("env") == env and r.get("size") == sz and r.get("success"):
                    m = r["method"]
                    if m in methods:
                        block.setdefault(m, []).append(r)
            tblock = {}
            for r in trecs:
                if r.get("env") == env and r.get("size") == sz:
                    m = r["method"]
                    if m in methods:
                        tblock.setdefault(m, []).append(r)
            row = {"env": env, "size": sz}
            for m in methods:
                rs = block.get(m, [])
                row[m] = {
                    "nodes_mean": float(np.mean([r["nodes"] for r in rs])) if rs else None,
                    "n": len(rs),
                }
                ts = tblock.get(m, [])
                row[m]["t_median_mean"] = float(np.mean([r["t_median"] for r in ts])) if ts else None
            rows.append(row)

    # node reduction vs Traditional A*
    for row in rows:
        base = row.get("Traditional A*", {}).get("nodes_mean")
        prop = row.get("Proposed", {}).get("nodes_mean")
        if base and prop:
            row["node_reduction_pct"] = 100.0 * (base - prop) / base
        tbase = row.get("Traditional A*", {}).get("t_median_mean")
        tprop = row.get("Proposed", {}).get("t_median_mean")
        if tbase and tprop:
            row["time_reduction_pct"] = 100.0 * (tbase - tprop) / tbase

    # write tex table
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{Node expansion and time across map sizes (compiled single-process timing, $n=30$ per cell). "
             "Node reduction: percentage fewer nodes than Traditional A*.}\\label{tab:scale}",
             "\\begin{tabular}{llrrrrrrr}", "\\toprule",
             "Env & Size & Trad.~A* & Theta* & W.Theta* & JPS & Proposed & Node red. & Time red. \\\\",
             "\\midrule"]
    for row in rows:
        cells = []
        for m in methods:
            v = row.get(m, {}).get("nodes_mean")
            cells.append("%.0f" % v if v is not None else "--")
        nr = row.get("node_reduction_pct")
        tr = row.get("time_reduction_pct")
        lines.append("%s & %d & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            row["env"].capitalize()[:3], row["size"],
            cells[0], cells[1], cells[2], cells[3], cells[4],
            ("%.1f\\%%" % nr) if nr is not None else "--",
            ("%.1f\\%%" % tr) if tr is not None else "--",
        ))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "tab_scale.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return rows


# ── footprint analysis ────────────────────────────────────────────────────────

def analyze_footprint():
    results = {}
    for stage, label in [("footprint", "rho0+0.5"), ("footprint_r1", "rho1.0_orig"),
                         ("footprint_r1q", "rho1.0_clr")]:
        recs = S.load(stage)
        methods = ["Traditional A*", "Theta*", "Weighted Theta*", "Proposed"]
        envs = list({r.get("env") for r in recs})
        inflates = sorted({str(r.get("inflate")) for r in recs})
        block = {}
        for r in recs:
            key = (r.get("env"), str(r.get("inflate")), r["method"])
            if r.get("success") and r["method"] in methods:
                block.setdefault(key, []).append(r)
        rows = []
        for env in sorted(envs):
            for infl in inflates:
                for m in methods:
                    rs = block.get((env, infl, m), [])
                    if not rs:
                        continue
                    rows.append({
                        "env": env, "inflate": infl, "method": m,
                        "n": len(rs),
                        "success_rate": len(rs) / max(1, sum(
                            1 for r in recs
                            if r.get("env") == env and str(r.get("inflate")) == infl
                            and r["method"] == m)),
                        "min_clear_mean": float(np.mean([r["min_clear"] for r in rs
                                                          if r.get("min_clear") is not None])),
                        "frac_below": {
                            k: float(np.mean([r[k] for r in rs if r.get(k) is not None]))
                            for k in ["frac_below0.3", "frac_below0.5"]
                            if any(r.get(k) is not None for r in rs)
                        },
                        "nodes_mean": float(np.mean([r["nodes"] for r in rs])),
                        "length_mean": float(np.mean([r["length"] for r in rs])),
                    })
        results[stage] = rows

    # write summary table
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{AGV footprint analysis. rho: planning radius (cells). "
             "Min\\_clear: mean minimum chassis clearance minus radius (cells). "
             "Frac$_{<r}$: fraction of path length with clearance $< r$.}\\label{tab:footprint}",
             "\\begin{tabular}{llllrrrr}", "\\toprule",
             "Stage & Env & $\\rho$ & Method & $n$ & Min clear & Frac$_{<0.5}$ & Nodes \\\\",
             "\\midrule"]
    for stage in ["footprint", "footprint_r1q"]:
        for row in results[stage]:
            fb = row["frac_below"].get("frac_below0.5", float("nan"))
            lines.append("%s & %s & %s & %s & %d & %.3f & %.3f & %.0f \\\\" % (
                stage.replace("footprint_r1q", "rho1.0\\_clr").replace("footprint", "rho0+0.5"),
                row["env"][:3], row["inflate"], row["method"].replace("*", "$^{*}$"),
                row["n"], row["min_clear_mean"],
                fb if math.isfinite(fb) else float("nan"),
                row["nodes_mean"],
            ))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "tab_footprint.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return results


# ── MovingAI analysis ─────────────────────────────────────────────────────────

def analyze_movingai():
    recs = S.load("movingai")
    trecs = S.load("timing_nb_movingai")
    bad = [r for r in trecs if not r.get("nb_equal", True)]
    if bad:
        sys.exit("timing_nb_movingai nb_equal false: %d" % len(bad))

    methods = ["Traditional A*", "Theta*", "Weighted Theta*", "Weighted Theta* (matched W)",
               "JPS", "Proposed", "Proposed (beta=0)"]
    metrics = ["nodes", "length", "ratio", "turns18"]

    t = {}
    for metric in metrics:
        t[metric] = S.table(recs, metric, method_key="method")
    tt = S.table(trecs, "t_median", method_key="method")

    rows = {}
    for metric in metrics:
        fam = S.family(t[metric], "Proposed", [m for m in methods if m != "Proposed"], metric)
        rows[metric] = fam

    # summary per map
    maps = sorted({r.get("map") for r in recs if r.get("map")})
    summary_rows = []
    for mp in maps:
        mr = [r for r in recs if r.get("map") == mp and r.get("success")]
        sz = None
        for r in mr:
            sz = r.get("size")
            break
        block = {}
        for r in mr:
            block.setdefault(r["method"], []).append(r)
        row = {"map": mp, "size": sz}
        for m in ["Traditional A*", "Proposed"]:
            rs = block.get(m, [])
            row[m] = {
                "nodes": float(np.mean([r["nodes"] for r in rs])) if rs else None,
                "ratio": float(np.mean([r["ratio"] for r in rs])) if rs else None,
            }
        if row["Traditional A*"].get("nodes") and row["Proposed"].get("nodes"):
            row["node_red"] = 100.0 * (row["Traditional A*"]["nodes"] - row["Proposed"]["nodes"]) / row["Traditional A*"]["nodes"]
        summary_rows.append(row)

    # write tex
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{MovingAI warehouse benchmark (unit-cost, no hotspots). "
             "Stars: Holm-adjusted paired $t$-test vs Proposed.}\\label{tab:movingai}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Method & Nodes & Length & $C/C^*_8$ & Turns & Time (ms) \\\\",
             "\\midrule"]
    smry = {metric: S.summary(t[metric]) for metric in metrics}
    tsmry = S.summary(tt)
    p_map = {}
    for metric in metrics:
        for row in rows[metric]:
            p_map[(row["other"], metric)] = row.get("p_holm")
    for m in methods:
        cells = []
        for metric in metrics:
            s = smry[metric].get(m)
            p = p_map.get((m, metric))
            if s:
                cells.append("$%.2f\\pm%.2f%s$" % (s["mean"], s["std"], stars(p) if m != "Proposed" else ""))
            else:
                cells.append("--")
        ts = tsmry.get(m)
        cells.append("$%.3f\\pm%.3f$" % (ts["mean"], ts["std"]) if ts else "--")
        name = m.replace("*", "$^{*}$")
        if m == "Proposed":
            name = "\\textbf{Proposed}"
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "tab_movingai.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"by_metric": rows, "by_map": summary_rows}


# ── bridge analysis ───────────────────────────────────────────────────────────

def analyze_bridge():
    recs = S.load("bridge")

    # B0/L0/L1 have cost=null (sm_valid_exact=False under new collision model).
    # Use model_cost (always present) and derive model_ratio = model_cost / c8.
    for r in recs:
        mc = r.get("model_cost")
        c8v = r.get("c8")
        if mc is not None and c8v is not None and c8v > 0:
            r["model_ratio"] = float(mc) / float(c8v)

    ladder = ["L0 original code", "L1 original semantics re-implemented",
              "L2 + exact collision", "L3 + integrated cost",
              "L4 + cost-based parent", "L5 + reopening",
              "L6 + cost-monotone smoothing", "L7 + adaptive eps",
              "L8 + selected config (= Proposed)"]
    baselines = ["B0 Traditional A* (original code)", "B1 Traditional A* (revised)"]
    metrics = ["nodes", "length", "model_cost", "model_ratio"]

    t = {}
    for metric in metrics:
        t[metric] = S.table(recs, metric)

    rows = []
    for metric in metrics:
        all_m = ladder + baselines
        present = [m for m in all_m if m in t[metric]]
        fam = S.family(t[metric], "L8 + selected config (= Proposed)", present, metric)
        rows.extend(fam)

    # consecutive step differences (nodes)
    tn = t.get("nodes", {})
    steps = []
    for i in range(1, len(ladder)):
        prev, curr = ladder[i - 1], ladder[i]
        if prev in tn and curr in tn:
            x, y, cl, excl = S.pair(tn, curr, prev)
            r = S.paired_test(x, y, cl)
            r.update(from_step=prev, to_step=curr, metric="nodes")
            steps.append(r)

    # write tex
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{Attribution ladder: mean nodes and operational cost at each algorithmic step. "
             "Evaluated on the same maps and queries as the main experiment.}\\label{tab:bridge}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "Step & Nodes & Length & Model cost & $C_m/C^*_8$ \\\\",
             "\\midrule"]
    for m in baselines + ladder:
        cells = []
        for metric in metrics:
            s = t[metric].get(m)
            if s:
                v = float(np.mean([x[0] for x in s.values()]))
                cells.append("%.1f" % v if metric == "nodes" else "%.3f" % v)
            else:
                cells.append("--")
        short = m.replace("L0 original code", "L0 orig").replace("L1 original semantics re-implemented", "L1 re-impl") \
                 .replace("L2 + exact collision", "L2 +exact LOS") \
                 .replace("L3 + integrated cost", "L3 +int.~cost") \
                 .replace("L4 + cost-based parent", "L4 +cost parent") \
                 .replace("L5 + reopening", "L5 +reopen") \
                 .replace("L6 + cost-monotone smoothing", "L6 +mono.~smooth") \
                 .replace("L7 + adaptive eps", "L7 +adaptive $\\varepsilon$") \
                 .replace("L8 + selected config (= Proposed)", "\\textbf{L8 (= Proposed)}") \
                 .replace("B0 Traditional A* (original code)", "B0 Trad.~A* (orig)") \
                 .replace("B1 Traditional A* (revised)", "B1 Trad.~A* (rev)")
        lines.append(short + " & " + " & ".join(cells) + " \\\\")
        if m == baselines[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "tab_bridge.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"vs_l8": rows, "steps": steps}


# ── E15 kinematic sensitivity ─────────────────────────────────────────────────

def analyze_kin_sensitivity():
    recs = S.load("kin")
    methods = ["Traditional A*", "Weighted A*", "Theta*", "Weighted Theta*", "Proposed"]
    envs_rhos = [("general", 0.0), ("general", 0.5), ("warehouse", 0.0), ("warehouse", 0.5)]
    params = {
        "v_max":  [0.5, 1.0, 1.5, 2.0],
        "a_max":  [0.25, 0.5, 1.0],
        "w_max":  [0.5, 1.0, 2.0],
    }
    nominal = {"v_max": 1.0, "a_max": 0.5, "w_max": 1.0}
    metrics_kin = ["kin_time", "kin_energy"]

    # re-compute kinematic profile with varied parameters
    # paths are stored in kin.jsonl as path_sm
    results = {}
    for env, rho in envs_rhos:
        base_recs = [r for r in recs
                     if r.get("env") == env
                     and abs(float(r.get("rho_used") or 0.0) - rho) < 1e-9
                     and r.get("success") and r.get("path_sm")
                     and r["method"] in methods]
        for param, values in params.items():
            for val in values:
                tag = (env, rho, param, val)
                varied = {}
                for r in base_recs:
                    path = [tuple(p) for p in r["path_sm"]]
                    kw = dict(nominal)
                    kw[param] = val
                    if param == "w_max":
                        kw["aw_max"] = val
                    # temporarily override module-level constants
                    old = {k: getattr(K, k.upper()) for k in kw}
                    for k, v2 in kw.items():
                        setattr(K, k.upper(), v2)
                    try:
                        pr = K.profile(path)
                    except Exception:
                        pr = None
                    for k, v2 in old.items():
                        setattr(K, k.upper(), v2)
                    if pr is None:
                        continue
                    inst = (r.get("env"), r.get("seed"), str(r.get("query")),
                            r.get("density"), r.get("size"), r.get("inflate"), r.get("map"))
                    for metric in metrics_kin:
                        varied.setdefault(metric, {}).setdefault(r["method"], {})[inst] = (
                            float(pr[metric]), (r.get("env"), r.get("seed")))
                # paired comparisons Proposed vs others
                for metric in metrics_kin:
                    tv = varied.get(metric, {})
                    fam = S.family(tv, "Proposed",
                                   [m for m in methods if m != "Proposed" and m in tv],
                                   metric)
                    results[tag + (metric,)] = fam

    # robustness summary: for each (env, rho, metric), count settings where
    # Proposed is significantly better/worse/same vs Traditional A*
    summary = {}
    for (env, rho, param, val, metric), fam in results.items():
        for row in fam:
            if row["other"] != "Traditional A*":
                continue
            key = (env, rho, metric)
            summary.setdefault(key, {"better": 0, "worse": 0, "ns": 0, "total": 0})
            summary[key]["total"] += 1
            p = row.get("p_holm", 1.0)
            d = row.get("d", 0.0)
            if p < 0.05:
                if d < 0:
                    summary[key]["better"] += 1
                else:
                    summary[key]["worse"] += 1
            else:
                summary[key]["ns"] += 1

    # write tex
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{E15 kinematic parameter sensitivity. Each cell shows the number of "
             "parameter settings (out of 10) where Proposed is significantly faster/more "
             "energy-efficient than Traditional A* (Holm $p<0.05$, $d<0$), "
             "tied (ns), or slower/higher-energy ($d>0$).}\\label{tab:kin_sens}",
             "\\begin{tabular}{llllrrrr}", "\\toprule",
             "Env & $\\rho$ & Metric & Better & NS & Worse & Total \\\\",
             "\\midrule"]
    for (env, rho, metric), s in sorted(summary.items()):
        lines.append("%s & %.1f & %s & %d & %d & %d & %d \\\\" % (
            env[:3], rho, metric.replace("_", "\\_"),
            s["better"], s["ns"], s["worse"], s["total"]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "tab_kin_sens.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"by_tag": {str(k): v for k, v in results.items()}, "summary": {str(k): v for k, v in summary.items()}}


# ── MuJoCo simulation analysis ────────────────────────────────────────────────

def analyze_mjsim():
    recs = S.load("mjsim", os.path.join(RES, "mjsim.jsonl"))
    methods_s0 = ["Traditional A*", "Weighted A*", "Theta*", "Weighted Theta*", "Proposed", "JPS"]
    methods_s05 = ["Traditional A*", "Weighted A*", "Theta*", "Weighted Theta*", "Proposed"]
    methods_s10 = ["Traditional A*", "Theta*", "Weighted Theta*", "Proposed"]

    set_methods = {"S0": methods_s0, "S05": methods_s05, "S10": methods_s10}
    metrics_sim = ["reached", "clean", "contacts", "contact_time", "min_clear",
                   "exec_time", "energy", "xt_mean", "xt_max"]

    results = {}
    for sim_set, methods in set_methods.items():
        set_recs = [r for r in recs if r.get("set") == sim_set]
        # average 5 seeds per path
        path_avg = {}
        for r in set_recs:
            k = (r.get("set"), r.get("src_key"), r.get("method"))
            path_avg.setdefault(k, []).append(r)
        avg_recs = []
        for (s, sk, m), rs in path_avg.items():
            base = {"set": s, "src_key": sk, "method": m,
                    "env": rs[0].get("env") or "warehouse",
                    "seed": rs[0].get("seed_map"),
                    "query": rs[0].get("query"),
                    "inflate": rs[0].get("r_body"),
                    "map": rs[0].get("map"),
                    "success": True}
            for metric in metrics_sim:
                vals = [r.get(metric) for r in rs if r.get(metric) is not None
                        and (not isinstance(r.get(metric), float) or math.isfinite(r.get(metric)))]
                if vals:
                    base[metric] = float(np.mean(vals)) if metric not in ("reached", "clean") \
                        else float(np.mean([float(v) for v in vals]))
            avg_recs.append(base)

        t = {}
        for metric in metrics_sim:
            t[metric] = S.table(avg_recs, metric, ok=lambda r: True)

        fam_results = {}
        for metric in ["reached", "clean", "contacts", "exec_time", "energy", "xt_max"]:
            if "Proposed" not in t.get(metric, {}):
                continue
            others = [m for m in methods if m != "Proposed" and m in t.get(metric, {})]
            fam_results[metric] = S.family(t[metric], "Proposed", others, metric)

        # summary table
        smry = {metric: S.summary(t[metric]) for metric in metrics_sim if metric in t}
        results[sim_set] = {"summary": smry, "comparisons": fam_results}

    # write tex
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{MuJoCo physics simulation results (5 noise seeds per path, "
             "averaged). Reached: fraction reaching goal. Clean: reached with no contact. "
             "Stars: Holm-adjusted paired $t$-test vs Proposed.}\\label{tab:mjsim}",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Set & Method & Reached & Clean & Contacts & Exec (s) & Energy & XTrack \\\\",
             "\\midrule"]
    for sim_set, methods in set_methods.items():
        res = results.get(sim_set, {})
        smry = res.get("summary", {})
        comp = res.get("comparisons", {})
        p_map = {}
        for metric, fam in comp.items():
            for row in fam:
                p_map[(row["other"], metric)] = row.get("p_holm")
        for m in methods:
            cells = []
            for metric in ["reached", "clean", "contacts", "exec_time", "energy", "xt_max"]:
                s = smry.get(metric, {}).get(m)
                p = p_map.get((m, metric))
                if s:
                    fmt_v = "%.2f" % s["mean"]
                    st = stars(p) if m != "Proposed" else ""
                    cells.append("$%s%s$" % (fmt_v, st))
                else:
                    cells.append("--")
            name = m.replace("*", "$^{*}$")
            if m == "Proposed":
                name = "\\textbf{Proposed}"
            lines.append("%s & %s & %s \\\\" % (
                sim_set if m == methods[0] else "", name, " & ".join(cells)))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT, "tab_mjsim.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    print("scale...", flush=True)
    scale = analyze_scale()
    print("footprint...", flush=True)
    footprint = analyze_footprint()
    print("movingai...", flush=True)
    movingai = analyze_movingai()
    print("bridge...", flush=True)
    bridge = analyze_bridge()
    print("kin_sensitivity...", flush=True)
    kin_sens = analyze_kin_sensitivity()
    print("mjsim...", flush=True)
    mjsim = analyze_mjsim()

    out = {
        "scale": [[{k: v for k, v in r.items() if k not in ("Traditional A*", "Theta*",
                   "Weighted Theta*", "JPS", "Proposed")} for r in scale]],
        "footprint_stages": list(footprint.keys()),
        "movingai_maps": [r["map"] for r in movingai["by_map"]],
        "bridge_steps": len(bridge["steps"]),
        "kin_sens_summary": kin_sens["summary"],
        "mjsim_sets": list(mjsim.keys()),
    }
    with open(os.path.join(OUT, "stages2.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    written = [x for x in sorted(os.listdir(OUT)) if not x.startswith("._")]
    print("written", written)


if __name__ == "__main__":
    main()
