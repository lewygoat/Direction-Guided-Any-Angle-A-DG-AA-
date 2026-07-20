import json

d1 = json.load(open("supp_part1.json"))
d2 = json.load(open("supp_part2.json"))
out = []

out.append("% Supplementary tables - real simulation output (n as noted)")

out.append("\\begin{table}[htbp]\\centering")
out.append("\\caption{Robustness across obstacle density (general, $50\\times50$, $n=30$ maps each).}")
out.append("\\label{tab:robust_density}")
out.append("\\begin{tabular}{ccccc}\\toprule")
out.append("Density & Traditional nodes & Proposed nodes & Reduction & $p$ \\\\\\midrule")
for r, v in d1["density"].items():
    out.append("%.0f\\%% & %.1f & %.1f & %.1f\\%% & %.1e \\\\" % (
        float(r) * 100, v["trad_nodes"], v["prop_nodes"], v["node_red"], v["nodes_stat"]["p"]))
out.append("\\bottomrule\\end{tabular}\\end{table}")

out.append("")
out.append("\\begin{table}[htbp]\\centering")
out.append("\\caption{Scalability across map size (general, 20\\% obstacles, $n=15$ each). Node reduction grows with map size; on tiny maps preprocessing is not amortized.}")
out.append("\\label{tab:scalability}")
out.append("\\begin{tabular}{ccccc}\\toprule")
out.append("Size & Traditional nodes & Proposed nodes & Node red. & Time red. \\\\\\midrule")
for S, v in d2["scalability"].items():
    out.append("$%s\\times%s$ & %.0f$\\pm$%.0f & %.0f$\\pm$%.0f & %.1f\\%% & %.1f\\%% \\\\" % (
        S, S, v["trad_nodes"], v["trad_nodes_sd"], v["prop_nodes"], v["prop_nodes_sd"],
        v["node_red"], v["time_red"]))
out.append("\\bottomrule\\end{tabular}\\end{table}")

out.append("")
out.append("\\begin{table}[htbp]\\centering")
out.append("\\caption{Sensitivity of the heuristic inflation $\\varepsilon$ (general, $n=30$, $\\beta=0.15$). Larger $\\varepsilon$ expands fewer nodes at a larger suboptimality bound $(1+2\\beta)\\varepsilon$.}")
out.append("\\label{tab:sens_eps}")
out.append("\\begin{tabular}{ccccc}\\toprule")
out.append("$\\varepsilon$ & Nodes & Turns & Length & Bound $(1+2\\beta)\\varepsilon$ \\\\\\midrule")
for e, v in d1["eps"].items():
    mark = " (used)" if abs(float(e) - 1.1) < 1e-9 else ""
    out.append("%.2f%s & %.1f & %.2f & %.2f & %.2f \\\\" % (float(e), mark, v["nodes"], v["turns"], v["length"], v["bound"]))
out.append("\\bottomrule\\end{tabular}\\end{table}")

out.append("")
out.append("\\begin{table}[htbp]\\centering")
out.append("\\caption{Sensitivity of the direction penalty $\\beta$ (general, $n=30$, $\\varepsilon=1.1$).}")
out.append("\\label{tab:sens_beta}")
out.append("\\begin{tabular}{cccc}\\toprule")
out.append("$\\beta$ & Nodes & Turns & Length \\\\\\midrule")
for b, v in d1["beta"].items():
    mark = " (used)" if abs(float(b) - 0.15) < 1e-9 else ""
    out.append("%.2f%s & %.1f & %.2f & %.2f \\\\" % (float(b), mark, v["nodes"], v["turns"], v["length"]))
out.append("\\bottomrule\\end{tabular}\\end{table}")

out.append("")
out.append("\\begin{table}[htbp]\\centering")
out.append("\\caption{Turn count under different angle thresholds (general, $n=30$). The turn difference is not statistically significant at any threshold, so no turn-reduction claim is made against Traditional A*.}")
out.append("\\label{tab:turn_thresh}")
out.append("\\begin{tabular}{ccccc}\\toprule")
out.append("Threshold & Traditional & Proposed & Diff. & $p$ \\\\\\midrule")
for th, v in d2["turn_threshold"].items():
    out.append("$%s^\\circ$ & %.2f & %.2f & %.1f\\%% & %.3f \\\\" % (th, v["trad"], v["prop"], v["red"], v["stat"]["p"]))
out.append("\\bottomrule\\end{tabular}\\end{table}")

out.append("")
out.append("\\begin{table}[htbp]\\centering")
h = d2["hotspot_ablation"]
out.append("\\caption{Hotspot-weight ablation (warehouse, $n=%d$). The hotspot term steers paths away from congested cells at the cost of extra node expansion.}" % h["n"])
out.append("\\label{tab:hotspot_ablation}")
out.append("\\begin{tabular}{cccc}\\toprule")
out.append("Configuration & Nodes & Turns & Hotspot cells traversed \\\\\\midrule")
out.append("With hotspot weight & %.1f & %.2f & %.2f \\\\" % (h["on_nodes"], h["on_turns"], h["on_hotspot_hits"]))
out.append("Without hotspot weight & %.1f & %.2f & %.2f \\\\" % (h["off_nodes"], h["off_turns"], h["off_hotspot_hits"]))
out.append("\\bottomrule\\end{tabular}")
out.append("\\end{table}")

d3 = json.load(open("supp_part3.json"))

out.append("")
out.append("\\begin{table}[htbp]\\centering")
out.append("\\caption{Sensitivity of warehouse parameters $\\alpha$ (environmental weight), $\\lambda$ (hotspot penalty), and $\\sigma$ (hotspot radius) (warehouse, $n=30$, $\\varepsilon=1.3$, $\\beta=0.15$). ``Hotspot cells'' is the mean number of congested cells the final path traverses.}")
out.append("\\label{tab:sens_warehouse}")
out.append("\\begin{tabular}{lcccc}\\toprule")
out.append("Parameter & Nodes & Turns & Length & Hotspot cells \\\\\\midrule")
for name, sym, used in [("alpha", "\\alpha", 0.3), ("lambda", "\\lambda", 0.5), ("sigma", "\\sigma", 2.0)]:
    out.append("\\multicolumn{5}{l}{\\emph{Varying $%s$}} \\\\" % sym)
    for k, v in d3[name].items():
        mark = " (used)" if abs(float(k) - used) < 1e-9 else ""
        out.append("$%s=%.2f$%s & %.1f & %.2f & %.2f & %.3f \\\\" % (sym, float(k), mark, v["nodes"], v["turns"], v["length"], v["hotspot_hits"]))
    out.append("\\midrule")
out[-1] = "\\bottomrule"
out.append("\\end{tabular}\\end{table}")

open("supp_tables.tex", "w").write("\n".join(out))
print("supp_tables.tex written")
