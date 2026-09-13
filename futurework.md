# Future Work: Dual-Trace Demand–Supply Modulation for QD³

## Motivation

The current QD³ operator uses **read history** to estimate which memory addresses are likely to matter in future queries. A natural extension is to model both sides of the memory workload:

- **Read history = demand:** where the memory is actually being queried.
- **Write history = supply:** where the memory has already received frequent update opportunities.

The key idea is not simply to add read and write heat together, but to estimate their **mismatch**. An address that is queried frequently but written rarely should receive a stronger correction when a new write finally arrives.

## Dual Traces

Let $q_j \in \mathbb{R}^{d_k}$ denote a query and $k_t \in \mathbb{R}^{d_k}$ a write key.

Maintain two exponentially decayed second-moment traces:

$$
Q^{(r)}_j
=
\lambda_r Q^{(r)}_{j-1}
+
q_j q_j^\top
$$

$$
Q^{(w)}_t
=
\lambda_w Q^{(w)}_{t-1}
+
k_t k_t^\top
$$

The read trace estimates historical **read demand**, while the write trace estimates historical **write supply**.

To keep the additional state small, both traces can be represented in low-rank form:

$$
Q^{(r)}
\approx
U_r \operatorname{diag}(a_r) U_r^\top
$$

$$
Q^{(w)}
\approx
U_w \operatorname{diag}(a_w) U_w^\top
$$

where

$$
U_r,U_w \in \mathbb{R}^{d_k \times r},
\qquad
r \ll d_k
$$

For an incoming write key $k_t$, the read and write heats are

$$
h_r(k_t)
=
k_t^\top Q^{(r)} k_t
\approx
\sum_{i=1}^{r}
a_{r,i}
\left(u_{r,i}^\top k_t\right)^2
$$

and

$$
h_w(k_t)
=
k_t^\top Q^{(w)} k_t
\approx
\sum_{i=1}^{r}
a_{w,i}
\left(u_{w,i}^\top k_t\right)^2
$$

## Demand–Supply Priority

After normalizing the two heats to comparable scales, define the raw mismatch priority as

$$
p_{\mathrm{raw}}(k_t)
=
\frac{\hat{h}_r(k_t)}
{\epsilon+\hat{h}_w(k_t)}
$$

A bounded version can then be used:

$$
p(k_t)
=
\frac{p_{\mathrm{raw}}(k_t)}
{1+p_{\mathrm{raw}}(k_t)}
$$

Its interpretation is:

| Read demand | Write supply | Priority |
|---|---|---|
| High | Low | Very high |
| High | High | Moderate |
| Low | High | Low |
| Low | Low | Low |

The mechanism therefore gives the highest priority to addresses that are **important to readers but under-served by writes**.

The effective Delta correction rate becomes

$$
\tilde{\eta}_t
=
\min
\left\{
\eta
\left(
1+\beta p(k_t)
\right),
c
\right\}
$$

and the memory update is

$$
M_t
=
M_{t-1}
+
\tilde{\eta}_t
\left(
v_t-M_{t-1}k_t
\right)
k_t^\top
$$

Setting

$$
\beta=0
$$

exactly recovers ordinary Delta. A cap

$$
c<2
$$

can preserve the same single-address non-expansion condition used in the current QD³ formulation.

## Interpretation

The extension can be viewed as a **demand–supply allocator**.

Read history estimates where memory accuracy is valuable:

$$
\text{Read Trace}
\rightarrow
\text{Demand}
$$

Write history estimates where corrective opportunities have already been abundant:

$$
\text{Write Trace}
\rightarrow
\text{Supply}
$$

The desired priority is therefore approximately

$$
\text{Priority}
\propto
\frac{\text{Demand}}
{\text{Supply}}
$$

The most important case is

$$
\text{high read demand}
+
\text{low write supply}
$$

because such an address is frequently needed but has received relatively few opportunities to be corrected.

## Future Evaluation

A future study should compare **read-only QD³**, **write-only trace**, **simple read-plus-write fusion**, **demand/supply mismatch modulation**, and **full-moment versus low-rank dual traces**.

The most informative setting is one in which

$$
p_{\mathrm{query}}
\neq
p_{\mathrm{write}}
$$

so that read demand and write supply are deliberately mismatched.

**Hypothesis:** the dual-trace formulation should provide the largest benefit when highly queried addresses receive relatively few write opportunities. When reads and writes are uniform or closely matched, its additional benefit should be small.
