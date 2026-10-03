# Revision (v2) — Direction-Guided Any-Angle A*

This directory contains the re-implemented planners and the full experiment,
analysis, and provenance pipeline produced for the revision of the manuscript.

## What changed from the original code

The search was re-implemented so that it satisfies the preconditions of the
bounded-suboptimality guarantee:

- CLOSED nodes are **reopened** when a cheaper path is found;
- the any-angle parent is chosen by **minimum integrated cost** (not greedily by
  line-of-sight);
- edge cost is the **exact line integral** of the operational weight along the
  segment (not an endpoint average);
- smoothing accepts a shortcut only when it does **not increase** the integrated
  cost.

Under these provisions the returned cost satisfies
`C <= (1 + 2*beta) * eps_max * C*_8`, where `C*_8` is the optimal 8-connected
grid cost under the same operational metric. The bound was checked on every one
of the >109,000 planning instances and never violated.

## Files

| File | Purpose |
|------|---------|
| `rv_geom.py` | grid geometry, exact segment cost, swept clearance, smoothing |
| `rv_planners.py` | reference planners (A*, Weighted A*, Theta*, Weighted Theta*, JPS, Proposed) with reopening |
| `rv_numba.py`, `rv_numba_equiv.py` | compiled (Numba) planners + node-for-node equivalence gate |
| `rv_core.py`, `rv_run.py` | instance runner, staged experiment driver, defensive invariant checks |
| `rv_movingai.py` | MovingAI benchmark loader |
| `rv_kin.py`, `rv_mjsim.py`, `rv_mjrun.py` | kinematic profile + MuJoCo physics simulation |
| `rv_stats.py`, `rv_an_*.py` | paired tests, Holm correction, cluster bootstrap, LaTeX table generation |
| `rv_fig_pareto.py` | node–turn frontier figure |
| `rv_tests.py` | unit tests (bound, validity, clearance, equivalence) |
| `protocol.json` | the pre-registered protocol with all amendments |
| `selected_config.json` | tuned configuration (eps_max=1.3, beta=0.05) |
| `source_sha256.txt`, `provenance.json` | source hashes and run environment |
| `analysis/` | generated LaTeX tables and summary JSON |

## Reproduce

```bash
pip install numpy scipy numba mujoco==3.3.7
python rv_tests.py                 # unit tests, must pass
python rv_run.py                   # all staged experiments -> results/*.jsonl
python rv_mjrun.py diag,0,1,2 8    # MuJoCo physics simulation
python rv_an_main.py               # main tables
python rv_an_stages1.py            # frontier / density / hotspot
python rv_an_stages2.py            # scale / footprint / movingai / bridge / mjsim / kin
```

Raw per-instance logs (`results/*.jsonl`, ~100 MB) and the superseded
`eps_max=2.0` archive are not committed; they regenerate deterministically from
the fixed seeds in the scripts. The MovingAI maps are redistributed from
movingai.com under their original terms and are downloaded by `rv_movingai.py`.

## Protocol discipline

The protocol (`protocol.json`) was fixed before any test-set metric was
inspected. Parameter selection used a tuning split disjoint from the 100 test
maps per environment. Amendments 1–7 record every change made after the initial
lock, including the single post-inspection decision (keeping the adaptive
configuration as the primary method and reporting the fixed-epsilon result as an
ablation).
