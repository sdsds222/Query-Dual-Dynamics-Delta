# Data provenance

This folder contains the prepared arrays used by the reported experiments.

| File | Role | Construction |
|---|---|---|
| `word.npz` | Controlled source-token profiles | Token counts and short-/long-window co-occurrence profiles extracted from locally installed standard-library source files by `code/prepare_data.py`. |
| `module.npz` | Controlled module-import graph | Module co-import contexts and empirical import frequencies extracted from the same source collection by `code/prepare_data.py`. |
| `wikitext2_qd3.npz` | WikiText-2 frequency-profile control | Vocabulary, context-derived keys/values, and validation/test frequencies built from the official WikiText-2 train/validation/test splits. |
| `wikitext2_ordered_qd3.npz` | Main held-out token-order experiment | The same WikiText-2 memory items with contiguous, held-out validation and test token sequences retained for causal evaluation. |

The first two artifacts are real-data-derived controlled workloads, not
standard benchmarks. Their purpose is to isolate read/write distribution
mismatch in two different structures (token statistics and a graph). The
WikiText-2 ordered experiment supplies the external held-out sequence result.

Prepared WikiText-2 arrays embed source-file hashes and diagnostic metadata.
Every result directory also contains a manifest with hashes of the arrays and
runner used for that experiment. Rebuilding `word.npz` or `module.npz` on a
different software installation can change the source collection; exact
reproduction should therefore use the archived arrays in this package.
