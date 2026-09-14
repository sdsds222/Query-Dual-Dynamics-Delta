# QD³ reproduction

Journal: *Complex & Intelligent Systems*. Author information is withheld in
this review copy to preserve the journal's double-blind process.

This folder contains the clean source and prepared data for the experiments in
“Query Dual Dynamics Delta: Read-Driven Modulation of Delta-Rule Writes in
Fixed-Capacity Fast-Weight Memory.” The internal method label `dqd` denotes QD³.

## Setup

Run all commands from this directory.

```powershell
python -m pip install -r requirements.txt
python test_dqd.py
```

The test should end with `All DQD invariants passed.` Use `--device cpu` if
CUDA is unavailable.

## Quick check

```powershell
python run_dqd_experiments.py --suite main --datasets synthetic --conditions concentrated --device cpu --quick --output results/quick_check
```

`--quick` checks the pipeline only and is not used for reported results.

## Full experiments

Main controlled experiments:

```powershell
python run_dqd_experiments.py --suite main --device cuda --val-seeds 8 --test-seeds 16 --output results/main
```

Ablations:

```powershell
python run_dqd_experiments.py --suite ablation --device cuda --val-seeds 8 --test-seeds 16 --output results/ablation
```

Robustness experiments:

```powershell
python run_dqd_experiments.py --suite robustness --device cuda --val-seeds 8 --test-seeds 16 --output results/robustness
```

Held-out ordered and shuffled WikiText-2 experiments:

```powershell
python run_wikitext2_ordered.py --device cuda --val-seeds 8 --test-seeds 16 --output results/wikitext2_ordered
```

WikiText-2 observed-frequency and uniform controls:

```powershell
python run_wikitext2_experiment.py --device cuda --val-seeds 8 --test-seeds 16 --output results/wikitext2_frequency
```

State-matched Dual Trace control:

```powershell
python run_dual_trace_experiments.py --suite main --device cuda --val-seeds 8 --test-seeds 16 --baseline-results reported_results/main_blind_v1 --output results/dual_trace
```

The Dual Trace run reuses the archived per-seed baselines after verifying their
seeds and experimental settings; it computes only the additional control.

Each run writes `manifest.json`, `validation_grid.csv`,
`selected_hyperparameters.json`, `per_seed.csv`, `summary.csv`, and
`paired.csv`. Hyperparameters are selected on validation seeds and then frozen
for the 16 paired test seeds. The included `.npz` files are the exact prepared
arrays used by these commands. `reported_results/` contains the complete outputs
used in the manuscript, including the Dual Trace control.

The root source files are the consolidated implementation used for reproduction.
`recorded_sources/pre_dual/` preserves the exact earlier source revisions whose
SHA-256 values appear in the original main and WikiText-2 manifests; the later
Dual Trace source hashes match the consolidated `dqd_core.py` and
`run_dual_trace_experiments.py` in this folder.
