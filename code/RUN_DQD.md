# Running the Dual Query Delta experiments

Run all commands from this `qtfw` directory in the `qwen_imdb` environment.
The old `run_v2.py` results are pilot evidence; paper tables come only from the
new commands below.

## 0. Mandatory verification

```powershell
python test_dqd.py
```

The command must end with `All DQD invariants passed.` It checks every method,
batched versus individual execution, exact `beta=0` reduction, source
isolation, deterministic replay, and NumPy/Torch/CUDA parity.

An optional short pipeline check is:

```powershell
python run_dqd_experiments.py --suite main --datasets synthetic --conditions concentrated --device cuda --quick --output results/pipeline_check_local
```

`--quick` uses old development seeds and must never be cited in the paper.

## 1. Final main experiment -- run first and only once

```powershell
python run_dqd_experiments.py --suite main --device cuda --val-seeds 8 --test-seeds 16 --output results/main_blind_v1
```

This produces the primary concentrated-workload results, uniform controls,
the matched low-rank write-trace comparison, full query trace, stationary query
moment reference, and weighted linear reference. Seeds 2001--2008 select the
hyper-parameters; previously unseen seeds 10001--10016 form the blind test.

## 2. Mechanism and rank ablations

```powershell
python run_dqd_experiments.py --suite ablation --device cuda --val-seeds 8 --test-seeds 16 --output results/ablation_blind_v1
```

This compares diagonal, last-query, fixed-random-subspace, full-moment, and DQD
ranks 1/2/4/8/16. The method list was fixed before seeing the blind results.

## 3. Query-skew and moving-hotspot robustness

```powershell
python run_dqd_experiments.py --suite robustness --device cuda --val-seeds 8 --test-seeds 16 --output results/robustness_blind_v1
```

For the two real-data-derived workloads this sweeps the query-frequency power
from exactly uniform (`alpha=0`) through the observed workload (`alpha=1`) and
beyond (`alpha=1.25`). For the spatial task it includes a hotspot that changes
halfway through the stream.

## 4. Optional moving-hotspot decay study

Run only after the three fixed suites above. These runs measure the expected
adaptation/stability trade-off rather than searching for a winning decay.

```powershell
python run_dqd_experiments.py --suite robustness --datasets synthetic --conditions moving_hotspot --device cuda --lam 1.0 --val-seeds 8 --test-seeds 16 --output results/hotspot_lam_1000
python run_dqd_experiments.py --suite robustness --datasets synthetic --conditions moving_hotspot --device cuda --lam 0.999 --val-seeds 8 --test-seeds 16 --output results/hotspot_lam_0999
python run_dqd_experiments.py --suite robustness --datasets synthetic --conditions moving_hotspot --device cuda --lam 0.99 --val-seeds 8 --test-seeds 16 --output results/hotspot_lam_0990
```

## Outputs

Every results directory contains:

- `manifest.json`: command, environment, GPU, configuration, dataset shapes,
  and SHA-256 hashes;
- `validation_grid.csv`: complete validation grid, including losing settings;
- `selected_hyperparameters.json`: selections made before test scoring;
- `per_seed.csv`: all blind-test observations;
- `summary.csv`: means, sample standard deviations, confidence intervals, and
  gains relative to plain Delta;
- `paired.csv`: paired differences, confidence intervals, t statistics, p
  values when SciPy is installed, and win counts.

Do not rerun a completed blind directory under the same version. If code or the
protocol changes, give the run a new version and describe the change.
