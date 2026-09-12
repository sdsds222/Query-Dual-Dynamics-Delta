# Dual Query Delta: preregistered experiment plan

This document fixes the questions, comparisons, and reporting rules before the
paper experiments are run. Negative and null results are retained.

## Central claim

Dual Query Delta (DQD) maintains a low-rank trace of historical queries and
uses that trace to modulate subsequent delta-rule writes. Under a fixed memory
budget, DQD should improve error under a non-uniform query workload when the
write and query distributions differ. It should provide no material advantage
when the query trace contains no information beyond the write stream.

The mechanism is a query-second-moment proxy for workload importance. It is not
claimed to implement the exact density ratio or exact query-weighted least
squares.

## Research questions

1. **RQ1 -- main effect.** Does DQD reduce workload-weighted recall error over
   the plain delta rule on concentrated workloads?
2. **RQ2 -- source of information.** Does a query trace outperform a
   capacity- and estimator-matched trace learned from writes?
3. **RQ3 -- mechanism.** How much is contributed by adaptive low-rank
   subspace learning, rank, and accumulated history?
4. **RQ4 -- boundary and robustness.** How does the effect change with query
   skew, uniform queries, query rate, and a moving hotspot?
5. **RQ5 -- practical value.** What accuracy is gained per extra state element
   and per unit of runtime in a fixed-capacity streaming memory?

## Data

- **Synthetic spatial field.** Uniform write locations, an independent query
  stream, and an independent test stream. Concentrated, uniform, skew-sweep,
  and moving-hotspot workloads are generated without mixing query points into
  the write set.
- **Python-token workload (`word.npz`).** Existing local artifact: semantic
  keys from short-window co-occurrence, values from long-window profiles, and
  query probabilities from observed token frequency.
- **Module-import workload (`module.npz`).** Existing local artifact: keys and
  values from co-import structure and query probabilities from import counts.

The two `.npz` datasets are described as real-data-derived controlled workloads,
not as end-to-end real applications. Their hashes and array shapes are written
to every run manifest.

## Main methods

- `plain`: standard delta rule.
- `write_diag`: the previous diagonal write-frequency control, retained for
  continuity with the exploratory results.
- `write_trace`: the decisive matched control. It has the same adaptive
  low-rank estimator, state size, update count, normalization, and
  hyper-parameter grid as DQD, but observes writes instead of queries.
- `dqd`: low-rank historical-query trace.
- `full_query_trace`: online full second-moment trace; more state, no low-rank
  approximation.
- `query_moment_reference`: receives the true stationary query second moment
  and uses the same scalar-gain form. This is a reference, not an upper bound.
- `weighted_linear_reference`: optimal linear least-squares map under the test
  workload. This is a linear-model reference, not a bound on arbitrary memory
  systems.

## Ablations

- diagonal query trace;
- last-query-only gain;
- fixed random query subspace;
- DQD ranks 1, 2, 4, 8, and 16;
- query frequency and trace-decay sensitivity.

## Selection and statistics

- Earlier seeds 1--16 and 101--104 are treated as pilot/development evidence
  because their outcomes were inspected while the experiment was designed.
- Final hyper-parameters are selected only on new validation seeds beginning at
  2001. Final blind tests use previously unseen seeds beginning at 10001.
- Methods use paired write orders, query streams, and test samples.
- Report every per-seed score, mean, sample standard deviation, paired
  difference, 95% confidence interval, paired t statistic, and win count.
- The primary comparisons are fixed in advance: `dqd` vs `plain` and `dqd` vs
  `write_trace`. Other comparisons are secondary and labelled as such.
- The uniform condition and all negative results are retained.

## Implementation safeguards

- NumPy and Torch/CUDA paths must agree within `2e-4` on every method.
- `dqd(beta=0)` and `write_trace(beta=0)` must reproduce `plain` exactly.
- Synthetic write, query, reference, and test samples use disjoint RNG streams.
- The matched write trace receives exactly as many trace observations as DQD.
- The plain timing path does not update unused trace state.
- Runs save the complete configuration, source/data hashes, selected
  hyper-parameters, summaries, paired statistics, and raw per-seed results.

## Application interpretation

The intended setting is a fixed-capacity memory with continued writes and a
query workload that is concentrated or changes over time: long-running robot
maps, embedded retrieval services, agent memories, and streaming sensor stores.
Repeated reads act as unlabeled workload feedback, so later observations near
frequently requested directions receive stronger corrective writes. DQD is not
expected to help a build-once/read-only store without a warm query log, or a
stream in which every item is queried and written with the same distribution.
