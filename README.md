# Query-Traced Fast Weights

**The informative signal is the *distribution* of queries, not the query itself — and it should modulate *writing*, not reading.**

A fast-weight memory has fixed capacity and cannot store everything. Every existing write rule decides what to keep from properties of the **current token**. This note argues it should be decided from **what has actually been asked**, and shows that this signal can be tracked in a second, tiny fast weight that uses the *same delta rule* as the main one.

> ⚠️ **Status: research note / work in progress.** Results below are on two synthetic task families. No real-data or end-to-end validation yet. Published early to document the idea and its boundaries. Feedback and pointers to prior art are very welcome — see *Prior art* below for what has already been checked.

---

## The problem

A fast weight is a fixed matrix `M`: write updates it, read returns `M q`. Its central tension is that capacity is fixed while the stream is not — entries interfere, and the memory cannot keep everything.

So the real question is: **given that it cannot keep everything, what should it keep?**

The delta-rule family (DeltaNet, Gated DeltaNet, Mamba, RWKV-7, …) answers from the current token — a learned gate `β_t`. None of them know which contents will be *asked about repeatedly*, because in all of them the query only reads; **it never influences how memory is written**.

Our answer: keep what will be queried.

*Analogy.* A library's shelves are finite. Current practice orders books by properties of the books. We instead track which sections readers actually borrow from, and give new arrivals in those sections the better shelf space.

---

## The mechanism

Two memories with **identical form** — one learns *what is stored*, the other *what is asked* — coupled through a single scalar.

```
state:  M ∈ R^{d_v×d_k}    main memory
        U ∈ R^{d_k×r}      query subspace      (r = 4)
        s ∈ R^r            energy per direction

on write (k, v):
    p = Uᵀk
    h = Σ s·p²  / norm                  # heat: does k lie where queries land?
    M += η·(1 + β·h)·(v − Mk)kᵀ         # plain delta write, times one scalar

on read (q):
    out = Mq                            # read path unchanged, returns immediately
    p = Uᵀq                             # --- side effect, can be asynchronous ---
    U += η_u·(q − Up)pᵀ                 # same delta rule, learning (q,q)
    s  = λ·s + p⊙p
```

The auxiliary memory is **the delta rule applied to queries**: it converges to the principal subspace of the query distribution (Oja's rule, no orthogonalisation needed — pure outer-product updates).

**Properties.** No new operator (no QR, no PCA routine, no random projection). `β = 0` recovers the plain delta rule exactly. Cold start has `s = 0 ⟹ g = 1`, so the memory *is* a plain fast weight until queries accumulate — no switching logic. Cost: `r·d_k + r` extra state (≈ +51% at r=4, +25% at r=2) and one r-dim projection per write.

**Interpretation.** The resulting `M` approximates the least-squares fit weighted by the **query** distribution instead of the **write** distribution — an online, O(1) form of importance weighting under covariate shift.

---

## Results

Two task families with no structure in common. Writes are uniform over the domain; queries are concentrated. Metric = error **under the actual query distribution** ("are the questions people actually ask answered well?"). All baselines get the same hyper-parameter search budget.

| | plain delta | **ours (r=4)** | oracle (knows the true query distribution) | fraction of oracle recovered |
|---|---|---|---|---|
| **A. continuous 2-D field**, concentrated queries | 0.7041 | **0.4824 (+31.5%)** | 0.4709 (+33.1%) | **95%** |
| **B. discrete recall**, Zipf queries, random keys | 0.8709 | **0.6929 (+20.4%)** | 0.6699 (+23.1%) | **88%** |
| either family, **uniform queries** | — | **+0.0%** | +0.0% | (clean degradation) |

Absolute gains differ because the tasks' oracles differ; what transfers is that the online trace recovers ~90% of the oracle in both.

---

## What the ablations rule out

Each row answers a "wouldn't something simpler work?" objection.

Numbers are (task A / task B), as fraction of the oracle's gain recovered.

| Objection | Control | A | B | recovers |
|---|---|---|---|---|
| Just use write frequency | write-frequency gain | +1.2% | +0.0% | **~0** — hot-spot information exists *only* in the queries |
| Just use the current query | Q-Delta-style per-step coupling | +0.0% | −0.0% | **0** — instantaneous coupling carries none of it |
| Just keep a per-channel counter | diagonal trace | +2.2% | +2.4% | ~8% — a low-rank *subspace* is required |
| Just compress with a fixed projection | random projection, r=32 | +19.6% | +11.4% | 55% — the subspace must be found adaptively |
| Just remember the last few queries | sliding window, r=4 | +23.4% | +13.1% | **65%** — recency is a decent proxy, but strictly worse |
| — | **ours, r=4** | **+31.5%** | **+20.4%** | **92%** |
| You cherry-picked the setting | uniform queries | +0.0% | +0.0% | — degrades cleanly, boundary explicit |

Three controls separate the mechanism *categorically*: write frequency, per-step query
coupling, and the uniform-query boundary all sit at zero. The remaining two — recency and
random projection — are **weaker estimators of the same quantity**, not different
mechanisms: recency recovers 65% of what the accumulated low-rank trace recovers. We state
this as a difference of degree, not of kind.

---

## Prior art (checked)

12 search angles, 14 papers read in full.

| Work | What it occupies | Relation |
|---|---|---|
| **Q-Delta** (arXiv:2606.08804) | First to argue the query should enter state evolution; couples the **current** `q_t` into the decay term | **Nearest neighbour.** Direction precedes us; mechanism differs (instantaneous vs accumulated distribution). Empirically +0.0% vs our +17.5% on task B |
| QED (2608.13668) | Query-derived erase *direction*, applied at write time, no cross-step accumulation | different |
| CCQ (2606.01294) | Query modifies the **read** only; accumulates **key** covariance | different (states the "read does not modify the write" assumption explicitly) |
| KDN (2609.07816) | Kalman gain; uncertainty comes from **writes** | different |
| HOLA (2607.02303) | Exact KV side-cache; admission by residual magnitude | different |
| H2O / SnapKV / Ada-KV | Access-driven eviction of **discrete tokens** | **Same spirit.** Differs in object (discrete cache vs continuous state) and action (eviction vs weighted write). Note their recency/frequency heuristics are *not* refuted here: our sliding-window control recovers 65% of the gain, so the contribution is a better estimator, not a new category |
| KLA, PDN, OSDN, GDN-2, MesaNet, LLA, Log-Linear Attention | Unrelated to query statistics | different |
| Covariate shift / importance weighting (Sugiyama et al.) | Statistical ancestor of the weighting | cited as theory, not architecture |
| Adaptive indexing / database cracking | Cross-domain ancestor: reorganising structure by query workload | cited as motivation |

---

## Scope and limitations

**Be explicit about where this does nothing.**

- **It is useless in language modelling**, and that is structural: there, every token both writes and queries, so the query distribution ≈ the write distribution — exactly the condition under which we measure **+0.0%**. It also cannot show up in perplexity, which averages over all tokens rather than weighting hot queries. We believe this is the main reason the idea has not been explored: the community working on these rules works on language.
- **Heat only affects subsequent writes.** Deployment modes: (A) *interleaved* writes and reads — works; (B) *build-then-query* — does not, unless writing continues; (C) *warm-started from a query log* — works from the first write.
- Validated only on synthetic tasks so far. No real data, no end-to-end training.
- Requires writes and queries to actually differ in distribution.

**Where it should help:** build-then-serve retrieval, streaming spatial mapping where a robot revisits regions, long-running agent memory — settings where writes cover everything but queries concentrate, and where hot-query accuracy is what matters.

---

## Reproduce

```bash
python experiments.py          # both task families + all six controls
```

`qtfw.py` is a ~60-line reference implementation with no dependencies beyond NumPy.

---

## Citing

If this is useful, please cite as a note and link here. If you know of prior work that
overlaps — especially anything accumulating query statistics to modulate a recurrent
state's write — please open an issue; I would rather find out early.

MIT licensed.
