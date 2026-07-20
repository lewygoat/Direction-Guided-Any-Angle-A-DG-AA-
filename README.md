# Reproduction Code for AJSE Submission

**Paper**: A Scalable Direction-Guided Any-Angle A* Algorithm for Efficient Warehouse AGV Path Planning  
**Authors**: Shaofang Mou  
**Target Journal**: Arabian Journal for Science and Engineering (AJSE)

---

## File Structure

```
reproduction/
├── README.md                      # This file
├── planners.py                    # All path planning algorithm implementations
├── environments.py                # Map generators (general & warehouse)
├── run_experiments.py             # Main experiments (Tables 1-2, Figures 6-9)
├── supp_experiments.py            # Density, scalability, ablation (Tables 3-6)
├── supp_experiments2.py           # Scalability sweep (Figure 7)
├── supp_experiments3.py           # Sensitivity analysis (Tables 7-8)
├── supp_experiments_adaptive.py   # Distance-adaptive inflation (Table 9)
├── ablation_anyangle.py           # Any-angle LOS ablation (Section 4.6)
├── gen_supp_tables.py             # Generate LaTeX table code
├── make_figures.py                # Generate all figures (PNG)
├── results.json                   # Main experiment raw data
├── supp_part1.json                # Density & ablation data
├── supp_part2.json                # Scalability data
├── supp_part3.json                # Sensitivity analysis data
├── adaptive_final.json            # Adaptive inflation data
└── ablation_anyangle.json         # Any-angle ablation data
```

---

## Quick Start

### Requirements
```bash
pip install numpy scipy matplotlib
```

### Reproduce All Results
```bash
# Main results (Tables 1-2, takes ~10 min)
python3 run_experiments.py

# Supplementary experiments (Tables 3-6, ~15 min)
python3 supp_experiments.py
python3 supp_experiments2.py
python3 supp_experiments3.py

# New experiments added after review
python3 supp_experiments_adaptive.py   # Table 9
python3 ablation_anyangle.py           # Section 4.6 any-angle ablation

# Generate figures (requires above JSON files)
python3 make_figures.py
```

### Verify Results Match Paper
```bash
# Check main table values
python3 -c "import json; d=json.load(open('results.json')); \
    print('Table 1 Proposed nodes:', d['general']['summary']['Proposed']['nodes']['mean'])"
# Expected: 110.9 (matches Table 1)

# Check any-angle ablation
python3 -c "import json; d=json.load(open('ablation_anyangle.json')); \
    print('Warehouse full:', d['warehouse']['full']['nodes']['mean'], 
          'grid-only:', d['warehouse']['grid_only']['nodes']['mean'])"
# Expected: full=90.6, grid-only=184.9 (matches Section 4.6)
```

---

## Algorithm Implementations in `planners.py`

| Function | Description | Paper Reference |
|----------|-------------|-----------------|
| `astar(grid, start, goal)` | Traditional A* (8-neighbor) | Baseline, Tables 1-2 |
| `weighted_astar(grid, start, goal, eps)` | Weighted A* with ε inflation | Baseline, Tables 1-2 |
| `astar_16n(grid, start, goal)` | 16-neighbor A* | Baseline, Tables 1-2 |
| `theta_star(grid, start, goal)` | Theta* (any-angle, grid-based) | Baseline, Tables 1-2 |
| `jps(grid, start, goal)` | Jump Point Search | Baseline, Tables 1-2 |
| `proposed_v3(grid, start, goal, ...)` | **Proposed method (final)** | Main method, all tables |
| `proposed_adaptive(grid, start, goal, ...)` | Proposed with adaptive ε | Table 9 |
| `proposed_grid_only(grid, start, goal, ...)` | Proposed without any-angle LOS | Section 4.6 ablation |

---

## Model Name & Abbreviations

### Full Name
**DG-AA\*** (Direction-Guided Any-Angle A*)

### Alternative Names (for tables/figures)
- **Proposed** (used in paper tables)
- **DG-AA\* (β=0.15)** (with direction penalty)
- **DG-AA\* (β=0)** (weighted any-angle baseline, no direction penalty)

### Baselines Compared
- **Traditional A\*** — Standard A* with 8-neighbor expansion
- **Weighted A\*** — Traditional A* with heuristic inflation ε
- **16-neighbor A\*** — A* with extended neighborhood
- **Theta\*** — Any-angle path planning on grids (Nash et al. 2007)
- **JPS** — Jump Point Search (Harabor & Grastien 2011)

---

## Key Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Heuristic inflation | ε | 1.1 (general), 1.3 (warehouse) | Weighted A* constant |
| Direction penalty weight | β | 0.15 | Direction guidance strength |
| Env weight scale | α | 0.3 | Obstacle proximity penalty |
| Hotspot weight | λ | 0.5 | Congestion avoidance weight |
| Hotspot radius | σ | 2.0 | Gaussian spread for hotspots |

---

## Data Format

All JSON files follow this structure:
```json
{
  "environment_name": {
    "planner_name": {
      "nodes": {"mean": float, "sd": float},
      "length": {"mean": float, "sd": float},
      "turns": {"mean": float},
      "time": {"mean": float}
    }
  }
}
```

---

## Citation

If you use this code, please cite:
```bibtex
@article{mou2025scalable,
  title={A Scalable Direction-Guided Any-Angle A* Algorithm for Efficient Warehouse AGV Path Planning},
  author={Mou, Shaofang},
  journal={Arabian Journal for Science and Engineering},
  year={2025},
  note={Under review}
}
```

---

## Contact

**Author**: Shaofang Mou  
**Email**: 374638996@qq.com  
**ORCID**: 0000-0003-2491-1485

---

## License

This code is provided for reproducibility and research purposes. Commercial use requires permission from the author.
