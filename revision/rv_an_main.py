import json
import os
import sys

import numpy as np

import rv_stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis")
REF = "Proposed"
BASELINES = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS",
             "Weighted Theta*", "Weighted Theta* (matched W)"]
ABLATIONS = ["Proposed (beta=0)", "Proposed (fixed eps)", "Proposed (grid only)", "Proposed (no reopen)"]
METRICS = ["nodes", "reopened", "length", "cost", "ratio", "turns18"]
TIME_METRIC = "t_median"
NAMES = {"nodes": "Nodes", "reopened": "Reopened", "length": "Length", "cost": "Op. cost", "ratio": "$C/C^*_8$",
         "turns18": "Turns", "t_median": "Time (ms)"}


def fmt(v, metric):
    if metric == "ratio":
        return "%.4f" % v
    if metric in ("nodes",):
        return "%.1f" % v
    return "%.2f" % v


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def analyze_env(recs, trecs, env, others):
    recs = [r for r in recs if r.get("env") == env and (r.get("wkey") or "full") == "full"]
    trecs = [r for r in trecs if r.get("env") == env]
    res = {"env": env, "n_success": {}, "metrics": {}}
    for m in [REF] + others:
        res["n_success"][m] = sum(1 for r in recs if r["method"] == m and r.get("success"))
    for metric, rs in [(x, recs) for x in METRICS] + [(TIME_METRIC, trecs)]:
        t = S.table(rs, metric)
        if REF not in t:
            continue
        cols = [m for m in others if m in t]
        res["metrics"][metric] = {"summary": S.summary(t), "vs_ref": S.family(t, REF, cols, metric)}
    return res


def write_tex(res, others, path, caption, label):
    lines = ["\\begin{table}[htbp]", "\\centering\\small",
             "\\caption{%s}\\label{%s}" % (caption, label),
             "\\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}", "\\toprule",
             "Method & " + " & ".join(NAMES[m] for m in METRICS + [TIME_METRIC]) + " \\\\", "\\midrule"]
    rank = {}
    for metric in METRICS + [TIME_METRIC]:
        blk = res["metrics"].get(metric)
        if blk is None:
            continue
        vals = sorted({round(blk["summary"][m]["mean"], 12) for m in others + [REF] if m in blk["summary"]})
        rank[metric] = vals[:2]
    for m in others + [REF]:
        cells = []
        for metric in METRICS + [TIME_METRIC]:
            blk = res["metrics"].get(metric)
            if blk is None or m not in blk["summary"]:
                cells.append("--")
                continue
            s = blk["summary"][m]
            p = None
            for row in blk["vs_ref"]:
                if row["other"] == m:
                    p = row.get("p_holm")
            body = "%s\\pm%s%s" % (fmt(s["mean"], metric), fmt(s["std"], metric), stars(p))
            mv = round(s["mean"], 12)
            best = rank.get(metric, [])
            if best and mv == best[0]:
                body = "\\mathbf{%s}" % body
            elif len(best) > 1 and mv == best[1]:
                body = "\\underline{%s}" % body
            cells.append("$%s$" % body)
        name = m.replace("*", "$^{*}$")
        if m == REF:
            name = "\\textbf{Proposed}"
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = S.load("main")
    trecs = S.load("timing_nb_main")
    bad = [r for r in trecs if not r.get("nb_equal", True)]
    if bad:
        sys.exit("timing records with nb_equal false: %d" % len(bad))
    out = {}
    for env in ("general", "warehouse"):
        others = BASELINES + ABLATIONS
        res = analyze_env(recs, trecs, env, others)
        out[env] = res
        write_tex(res, BASELINES, os.path.join(OUT, "tab_main_%s.tex" % env),
                  "%s environment, mean$\\pm$std over %s queries; stars: Holm-adjusted paired $t$-test vs Proposed." %
                  (env.capitalize(), res["n_success"][REF]), "tab:main_%s" % env)
        write_tex(res, ABLATIONS, os.path.join(OUT, "tab_ablation_%s.tex" % env),
                  "Component ablation, %s environment; stars: Holm-adjusted paired $t$-test vs full method." % env,
                  "tab:abl_%s" % env)
    with open(os.path.join(OUT, "main.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("written", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
