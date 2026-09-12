# QD³ experiment reproduction package

This archive reproduces the experiments reported in **Query Dual Dynamics
Delta: Read-Driven Write Allocation for Fixed-Capacity Fast-Weight Memory**.
It is self-contained for evaluation: the prepared arrays, executable scripts,
invariant tests, and reported CSV outputs are included.

QD³ is the paper name. The short internal label `dqd` is retained in filenames,
method identifiers, and result tables so that the released code remains
consistent with the recorded experiment outputs.

## Package contents

| File or directory | Purpose |
|---|---|
| `dqd_core.py` | NumPy and Torch implementations of all memory operators |
| `run_dqd_experiments.py` | Main controlled experiments, ablations, and robustness suites |
| `run_wikitext2_ordered.py` | Held-out ordered and shuffled WikiText-2 experiments |
| `run_wikitext2_experiment.py` | WikiText-2 observed-frequency and uniform controls |
| `test_dqd.py` | Invariants, determinism, source isolation, and backend agreement |
| `prepare_data.py` | Optional reconstruction of the source-token and module-import workloads |
| `prepare_wikitext2.py` | Optional construction of the WikiText-2 frequency workload |
| `prepare_wikitext2_ordered.py` | Optional addition of held-out ordered token sequences |
| `*.npz` | Exact prepared arrays used for the paper |
| `reported_results/` | Validation grids, frozen settings, per-seed scores, summaries, and manifests from the reported runs |

The experiment scripts contain executable code only. Explanatory material is
kept in this README so that the numerical implementation is easy to inspect.

## Environment

The reported runs used Python 3.10.20, NumPy 2.0.1, Torch 2.5.1, and an NVIDIA
GeForce RTX 3070 Ti Laptop GPU. Python 3.10 or newer is recommended. CPU-only
execution is supported through `--device cpu` when Torch is installed, or
`--device numpy` without Torch.

Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

For a particular CUDA version, install the matching official Torch build before
running the requirements command. No network access is needed after the
dependencies have been installed because all prepared arrays are included.

## Mandatory implementation check

Run this command from the extracted package directory:

```powershell
python test_dqd.py
```

Successful execution ends with:

```text
All DQD invariants passed.
```

The test covers every declared operator, batched versus individual execution,
exact reduction to Plain Delta at `beta=0`, deterministic replay, separation of
query and write sources, and NumPy/Torch/CUDA agreement. CUDA is tested only
when it is available.

## Short end-to-end check

The following command exercises data generation, validation selection, frozen
test evaluation, paired statistics, and file output in a few minutes or less:

```powershell
python run_dqd_experiments.py --suite main --datasets synthetic --conditions concentrated --device cpu --quick --output reproduced_results/smoke_test
```

The `--quick` flag uses small development settings. It verifies the pipeline but
must not be used to reproduce or cite the paper results.

## Full paper experiments

Use a new output directory for every run. The scripts refuse to overwrite an
existing directory.

### Main controlled workloads

```powershell
python run_dqd_experiments.py --suite main --device cuda --val-seeds 8 --test-seeds 16 --output reproduced_results/main
```

This command evaluates the concentrated and uniform spatial fields, the
source-token profiles, and the module-import graph. It includes Plain Delta,
matched write-derived traces, QD³, the dense query trace, the stationary query
moment reference, and the weighted linear reference.

### Mechanism and rank ablations

```powershell
python run_dqd_experiments.py --suite ablation --device cuda --val-seeds 8 --test-seeds 16 --output reproduced_results/ablation
```

This suite evaluates the diagonal trace, last-query signal, fixed random
subspace, dense query trace, and learned ranks 1, 2, 4, 8, and 16.

### Query-skew and moving-workload tests

```powershell
python run_dqd_experiments.py --suite robustness --device cuda --val-seeds 8 --test-seeds 16 --output reproduced_results/robustness
```

This suite includes the moving spatial hotspot and query-frequency powers from
uniform access through stronger-than-observed concentration on both controlled
real-data-derived workloads.

### Held-out WikiText-2 sequence

```powershell
python run_wikitext2_ordered.py --device cuda --val-seeds 8 --test-seeds 16 --output reproduced_results/wikitext2_ordered
```

The default conditions are `ordered` and `shuffled`. Query prefixes and scored
tokens occupy non-overlapping portions of disjoint held-out sequence blocks.

### WikiText-2 frequency and uniform controls

```powershell
python run_wikitext2_experiment.py --device cuda --val-seeds 8 --test-seeds 16 --output reproduced_results/wikitext2_frequency
```

This supplementary run compares the observed held-out frequency distribution
with uniform queries over the same memory items.

To run on a CPU, replace `--device cuda` with `--device cpu`. Small floating-point
differences across devices are expected; the implementation tests use an
absolute backend tolerance of `2e-4`.

## Validation and test protocol

Hyperparameters are selected separately for every method using eight validation
seeds. The selected setting is then frozen before evaluation on 16 paired test
seeds. Controlled workloads use validation seeds 2001--2008 and test seeds
10001--10016. WikiText-2 uses validation seeds 4001--4008 and test seeds
30001--30016. Methods within a seed receive identical write orders, query
sequences, and evaluation samples.

The controlled search uses:

- `eta` in `{0.02, 0.05, 0.1, 0.2, 0.4}`;
- `beta` in `{1, 3, 10, 30, 100}`; and
- low ranks in `{2, 4, 8}` for the main suite.

The WikiText-2 search uses `eta` in `{0.02, 0.05, 0.1}`, `beta` in
`{1, 3, 10, 30}`, and ranks in `{1, 2, 4}`. Fixed operator settings are
`eta_u=0.05`, `lambda=0.999`, clipping at `1.9`, and one query event per three
writes.

## Output files

Each output directory contains:

| File | Contents |
|---|---|
| `manifest.json` | Command, environment, seeds, settings, data shapes, hashes, and completion status |
| `validation_grid.csv` | Every validation candidate, including settings not selected |
| `selected_hyperparameters.json` | Frozen setting selected from validation data |
| `per_seed.csv` | One blind-test observation per method and seed |
| `summary.csv` | Mean, sample standard deviation, confidence interval, and gain from Plain Delta |
| `paired.csv` | Paired differences, confidence intervals, test statistics, and win counts |

Compare reproduced files with the matching directory under
`reported_results/`. The principal reported QD³ mean NRMSE values are 0.5040
for the concentrated spatial field, 0.6278 for source-token profiles, 0.6056
for the module-import graph, and 0.1674 for ordered WikiText-2. The corresponding
reductions from Plain Delta are 31.01%, 18.66%, 32.13%, and 2.62%.

## Data provenance

`word.npz` and `module.npz` are controlled workloads derived from source-code
text and module co-import structure in the Python standard-library installation
used for the reported experiments. Rebuilding them on another installation can
change the available source collection. Exact reproduction must therefore use
the included arrays; `prepare_data.py` is supplied only to document and repeat
the construction procedure on another source installation.

`wikitext2_qd3.npz` contains keys and values built only from the WikiText-2
training split, with validation and test frequencies retained separately.
`wikitext2_ordered_qd3.npz` additionally stores covered validation and test token
sequences. The ordered runner selects a query prefix followed by a disjoint
scored segment inside a seed-specific held-out block.

Rebuilding the WikiText-2 arrays is optional. Place the official parquet files
at the following paths before running the preparation scripts:

```text
data/wikitext2/train-00000-of-00001.parquet
data/wikitext2/validation-00000-of-00001.parquet
data/wikitext2/test-00000-of-00001.parquet
```

Then run:

```powershell
python prepare_wikitext2.py
python prepare_wikitext2_ordered.py
```

The prepared WikiText-2 files retain hashes of their raw sources. The archived
result manifests retain the hashes recorded when the paper experiments were
run. Code comments and docstrings were removed for this reviewer package only;
this changes file hashes but not the executable numerical statements.
