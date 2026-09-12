# Query Dual Dynamics Delta (QD³)

This directory is the self-contained paper and reproducibility package for
**“Query Dual Dynamics Delta: Read-Driven Write Allocation for Fixed-Capacity
Fast-Weight Memory.”** The manuscript is written for submission to *Neural
Processing Letters*.

## Directory map

- `manuscript/`: English submission source and PDF, a Chinese reading
  translation (`main_zh.tex` and `main_zh.pdf`), Springer Nature class/style
  files, bibliography, and publication figures.
- `code/`: exact experiment operators, runners, data preparation scripts, unit
  tests, figure generator, prepared arrays, and a local copy of every reported
  result directory. The prepared arrays are duplicated in `data/` so data and
  code can also be archived separately.
- `results/`: validation grids, frozen test outputs, paired statistics,
  manifests, and hashes used by the manuscript.
- `data/`: prepared arrays used by the reported runs and their provenance notes.

Internal filenames retain the earlier short name `dqd`; the paper name and
operator are QD³ (Query Dual Dynamics Delta). No numerical result depends on
this naming change.

## Build the manuscript

From `manuscript/`, run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `manuscript/main.pdf`. The checked 15-page build uses numeric
citations, A4 pages, three vector figures, and no unresolved references or
overfull boxes.

The Chinese reading version preserves the equations, data, and figure files but
translates the surrounding prose. Build it with XeLaTeX:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
bibtex main_zh
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```

## Verify the implementation

Create or activate a Python environment, install the listed dependencies, and
then work from `code/`:

```powershell
python -m pip install -r requirements.txt
```

Run the invariant suite:

```powershell
python test_dqd.py
```

For a short end-to-end check:

```powershell
python run_dqd_experiments.py --suite main --datasets synthetic --conditions concentrated --device cpu --quick --output results/reproduction_quick
```

The full controlled experiments are intentionally validation-separated. To
avoid overwriting the archived blind outputs, use new destination names:

```powershell
python run_dqd_experiments.py --suite main --device cuda --val-seeds 8 --test-seeds 16 --output results/reproduction_main
python run_dqd_experiments.py --suite ablation --device cuda --val-seeds 8 --test-seeds 16 --output results/reproduction_ablation
python run_dqd_experiments.py --suite robustness --device cuda --val-seeds 8 --test-seeds 16 --output results/reproduction_robustness
python run_wikitext2_ordered.py --device cuda --val-seeds 8 --test-seeds 16 --output results/reproduction_wikitext2_ordered
```

The broader frequency-profile WikiText-2 control can be reproduced with:

```powershell
python run_wikitext2_experiment.py --device cuda --val-seeds 8 --test-seeds 16 --output results/reproduction_wikitext2_frequency
```

`code/RUN_DQD.md` records the original suite commands and interruption-safe
continuation commands. The result manifests preserve the exact arguments,
chosen validation settings, seed-level measurements, and source hashes.

## Regenerate the figures

From `code/`, run:

```powershell
python make_qd3_figures.py
```

This writes PDF and high-resolution PNG versions to `code/figures/`. The six
publication copies used by LaTeX are frozen in `manuscript/figures/`.

## Submission checklist

The scientific content, equations, numerical tables, bibliography, and layout
have been compiled and checked. Before submission, the authors should still:

1. confirm the author order, institutional affiliation, and both email addresses;
2. replace or confirm the provisional funding, author-contribution, data
   availability, and competing-interest statements;
3. deposit the code/data package in a stable repository and replace the
   provisional availability wording with its DOI or permanent URL;
4. check the current journal portal for any cover-letter, graphical-abstract,
   or source-archive requirements; and
5. perform a final authorship-approved language and claim audit.

The paper deliberately makes a bounded claim: historical reads improve future
write allocation when read demand contains information not already present in
the write stream. Uniform-query controls identify that boundary without turning
it into a universal-performance claim.
