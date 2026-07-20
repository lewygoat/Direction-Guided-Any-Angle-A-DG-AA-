#!/usr/bin/env python3
"""
Ablation: Any-Angle Line-of-Sight Contribution.
Compares full method (with LOS) vs grid-only variant (no LOS).
"""
import json
import numpy as np
from environments import make_general, make_warehouse
from planners import proposed_v3, proposed_grid_only
from scipy.stats import ttest_rel

np.random.seed(42)

def run_anyangle_ablation():
    results = {}

    # General environment
    print("Running any-angle ablation: general (n=30)...")
    full_nodes, full_lens, full_turns = [], [], []
    grid_nodes, grid_lens, grid_turns = [], [], []

    for i in range(30):
        grid_map, start, goal, _ = make_general(seed=3000+i)

        info_full = proposed_v3(grid_map, start, goal, warehouse=False, eps=1.1)
        full_nodes.append(info_full["nodes"])
        full_lens.append(info_full["length"])
        full_turns.append(info_full["turns"])

        info_grid = proposed_grid_only(grid_map, start, goal, warehouse=False, eps=1.1)
        grid_nodes.append(info_grid["nodes"])
        grid_lens.append(info_grid["length"])
        grid_turns.append(info_grid["turns"])

    p_nodes = ttest_rel(grid_nodes, full_nodes, alternative='less').pvalue
    p_len = ttest_rel(grid_lens, full_lens, alternative='greater').pvalue

    results["general"] = {
        "full": {
            "nodes": {"mean": np.mean(full_nodes), "sd": np.std(full_nodes, ddof=1)},
            "length": {"mean": np.mean(full_lens), "sd": np.std(full_lens, ddof=1)},
            "turns": {"mean": np.mean(full_turns)}
        },
        "grid_only": {
            "nodes": {"mean": np.mean(grid_nodes), "sd": np.std(grid_nodes, ddof=1)},
            "length": {"mean": np.mean(grid_lens), "sd": np.std(grid_lens, ddof=1)},
            "turns": {"mean": np.mean(grid_turns)}
        },
        "p_nodes": p_nodes,
        "p_len": p_len
    }

    print(f"General: grid-only={np.mean(grid_nodes):.1f} nodes, full={np.mean(full_nodes):.1f} nodes")
    print(f"         grid-only={np.mean(grid_lens):.2f} length, full={np.mean(full_lens):.2f} length, p={p_len:.3f}")

    # Warehouse environment
    print("\nRunning any-angle ablation: warehouse (n=30)...")
    full_nodes_w, full_lens_w = [], []
    grid_nodes_w, grid_lens_w = [], []

    for i in range(30):
        grid_map, start, goal, hotspots = make_warehouse(seed=3000+i)

        info_full = proposed_v3(grid_map, start, goal, hotspots, warehouse=True, eps=1.3)
        full_nodes_w.append(info_full["nodes"])
        full_lens_w.append(info_full["length"])

        info_grid = proposed_grid_only(grid_map, start, goal, hotspots, warehouse=True, eps=1.3)
        grid_nodes_w.append(info_grid["nodes"])
        grid_lens_w.append(info_grid["length"])

    p_nodes_w = ttest_rel(grid_nodes_w, full_nodes_w, alternative='less').pvalue
    p_len_w = ttest_rel(grid_lens_w, full_lens_w, alternative='greater').pvalue

    results["warehouse"] = {
        "full": {
            "nodes": {"mean": np.mean(full_nodes_w), "sd": np.std(full_nodes_w, ddof=1)},
            "length": {"mean": np.mean(full_lens_w), "sd": np.std(full_lens_w, ddof=1)}
        },
        "grid_only": {
            "nodes": {"mean": np.mean(grid_nodes_w), "sd": np.std(grid_nodes_w, ddof=1)},
            "length": {"mean": np.mean(grid_lens_w), "sd": np.std(grid_lens_w, ddof=1)}
        },
        "p_nodes": p_nodes_w,
        "p_len": p_len_w
    }

    print(f"Warehouse: grid-only={np.mean(grid_nodes_w):.1f} nodes, full={np.mean(full_nodes_w):.1f} nodes")
    print(f"           grid-only={np.mean(grid_lens_w):.2f} length, full={np.mean(full_lens_w):.2f} length, p={p_len_w:.3f}")

    with open("ablation_anyangle.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nablation_anyangle.json written successfully")
    return results

if __name__ == "__main__":
    run_anyangle_ablation()
