import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "build", "fig6_pareto.png")

m = json.load(open(os.path.join(HERE, "analysis", "main.json")))["general"]["metrics"]
nd = m["nodes"]["summary"]
tn = m["turns18"]["summary"]
f = json.load(open(os.path.join(HERE, "analysis", "stages1.json")))["frontier"]["general"]
pts = {p["method"]: p for p in f["points"]}

baselines = [
    ("Traditional A*", "o", "#555555"),
    ("Weighted A*", "s", "#1f77b4"),
    ("16-neighbor A*", "^", "#2ca02c"),
    ("Theta*", "D", "#9467bd"),
    ("JPS", "v", "#8c564b"),
    ("Weighted Theta*", "P", "#ff7f0e"),
]

fig, ax = plt.subplots(figsize=(7.2, 5.0))

# Weighted Theta* frontier sweep (eps)
wt = [("WTheta*@%s" % e) for e in ["1", "1.05", "1.1", "1.2", "1.3", "1.5", "2"]]
wt = [pts[k] for k in wt if k in pts]
ax.plot([p["nodes"] for p in wt], [p["turns18"] for p in wt],
        "-", color="#ff7f0e", alpha=0.5, lw=1.4, zorder=1,
        label="Weighted Theta* frontier ($\\varepsilon$ sweep)")

# Proposed frontier sweep (beta at eps=1.3)
pr = [("Prop@1.3@%s" % b) for b in ["0.05", "0.1", "0.15", "0.2", "0.3"]]
pr = [pts[k] for k in pr if k in pts]
ax.plot([p["nodes"] for p in pr], [p["turns18"] for p in pr],
        "--", color="#d62728", alpha=0.6, lw=1.4, zorder=1,
        label="Proposed frontier ($\\beta$ sweep, $\\varepsilon_{\\max}{=}1.3$)")

# baseline scatter
for name, mk, col in baselines:
    if name in nd:
        ax.scatter(nd[name]["mean"], tn[name]["mean"], marker=mk, s=90,
                   color=col, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(name, (nd[name]["mean"], tn[name]["mean"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5)

# proposed as a star
ax.scatter(nd["Proposed"]["mean"], tn["Proposed"]["mean"], marker="*", s=360,
           color="#d62728", edgecolor="black", linewidth=0.8, zorder=4,
           label="Proposed (selected)")
ax.annotate("Proposed", (nd["Proposed"]["mean"], tn["Proposed"]["mean"]),
            textcoords="offset points", xytext=(8, -14), fontsize=9, fontweight="bold")

ax.set_xlabel("Expanded nodes (mean, general environment, lower is better)")
ax.set_ylabel("Path turns at $18^\\circ$ (mean, lower is better)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
