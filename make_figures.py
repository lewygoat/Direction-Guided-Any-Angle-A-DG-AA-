import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ".."
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.dpi": 300})

CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}

res = json.load(open("results.json"))
p1 = json.load(open("supp_part1.json"))
p2 = json.load(open("supp_part2.json"))


def fig_pareto():
    methods = ["Traditional A*", "Weighted A*", "16-neighbor A*", "Theta*", "JPS", "Proposed"]
    cols = [CB["black"], CB["orange"], CB["green"], CB["purple"], CB["blue"], CB["vermillion"]]
    mk = ["o", "s", "^", "D", "v", "*"]
    s = res["general"]["summary"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, c, k in zip(methods, cols, mk):
        x = s[m]["nodes"]["mean"]; y = s[m]["turns"]["mean"]
        sz = 320 if m == "Proposed" else 120
        ax.scatter(x, y, c=c, marker=k, s=sz, label=m, edgecolors="k", linewidths=0.6, zorder=3)
    ax.set_xlabel("Expanded nodes (fewer is better)")
    ax.set_ylabel("Path turns (fewer is better)")
    ax.set_title("Node--turn trade-off (general, $n=30$)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT + "/fig6_pareto.png")
    plt.close(fig)


def fig_scalability():
    sc = p2["scalability"]
    sizes = list(sc.keys())
    red = [sc[s]["node_red"] for s in sizes]
    treds = [sc[s]["time_red"] for s in sizes]
    x = range(len(sizes))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar([i - 0.2 for i in x], red, width=0.4, color=CB["blue"], label="Node reduction", edgecolor="k", linewidth=0.5)
    ax.bar([i + 0.2 for i in x], treds, width=0.4, color=CB["orange"], label="Time reduction", edgecolor="k", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["%sx%s" % (s, s) for s in sizes])
    ax.set_xlabel("Map size")
    ax.set_ylabel("Reduction vs Traditional A* (%)")
    ax.set_title("Scalability of the advantage")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "/fig7_scalability.png")
    plt.close(fig)


def fig_density():
    dd = p1["density"]
    ratios = list(dd.keys())
    red = [dd[r]["node_red"] for r in ratios]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([float(r) * 100 for r in ratios], red, "-o", color=CB["green"], markersize=8, linewidth=2, markeredgecolor="k")
    ax.set_xlabel("Obstacle density (\\%)")
    ax.set_ylabel("Node reduction vs Traditional A* (\\%)")
    ax.set_title("Robustness across obstacle density ($n=30$)")
    for r in ratios:
        ax.annotate("%.1f%%" % dd[r]["node_red"], (float(r) * 100, dd[r]["node_red"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "/fig8_density.png")
    plt.close(fig)


def fig_eps():
    ee = p1["eps"]
    epss = [float(e) for e in ee.keys()]
    nodes = [ee[e]["nodes"] for e in ee.keys()]
    bound = [ee[e]["bound"] for e in ee.keys()]
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    ax1.plot(epss, nodes, "-o", color=CB["blue"], markersize=7, linewidth=2, label="Expanded nodes", markeredgecolor="k")
    ax1.set_xlabel("Heuristic inflation $\\varepsilon$")
    ax1.set_ylabel("Expanded nodes", color=CB["blue"])
    ax1.tick_params(axis="y", labelcolor=CB["blue"])
    ax2 = ax1.twinx()
    ax2.plot(epss, bound, "--s", color=CB["vermillion"], markersize=7, linewidth=2, label="Suboptimality bound", markeredgecolor="k")
    ax2.set_ylabel("Suboptimality bound $(1+2\\beta)\\varepsilon$", color=CB["vermillion"])
    ax2.tick_params(axis="y", labelcolor=CB["vermillion"])
    ax2.grid(False)
    ax1.set_title("Speed--optimality trade-off (general, $n=30$)")
    fig.tight_layout()
    fig.savefig(OUT + "/fig9_eps.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_pareto()
    fig_scalability()
    fig_density()
    fig_eps()
    print("figures written: fig6_pareto, fig7_scalability, fig8_density, fig9_eps")
