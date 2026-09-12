# QD³ reproduction

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

Each run writes `manifest.json`, `validation_grid.csv`,
`selected_hyperparameters.json`, `per_seed.csv`, `summary.csv`, and
`paired.csv`. Hyperparameters are selected on validation seeds and then frozen
for the 16 paired test seeds. The included `.npz` files are the exact prepared
arrays used by these commands.
