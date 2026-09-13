# Future Work: Dual-Trace Demand–Supply Modulation for QD³

## Motivation

The current QD³ operator uses **read history** to estimate which memory addresses are likely to matter in future queries. A natural extension is to model both sides of the memory workload:

- **Read history = demand:** where the memory is actually being queried.
- **Write history = supply:** where the memory has already received frequent update opportunities.

The key idea is not simply to add read and write heat together, but to estimate their **mismatch**. An address that is queried frequently but written rarely should receive a stronger correction when a new write finally arrives.

---

## Dual traces

Let \(q_j\in\mathbb{R}^{d_k}\) denote a query and \(k_t\in\mathbb{R}^{d_k}\) a write key. Maintain two exponentially decayed second-moment traces:

\[
Q^{(r)}_j
=
\lambda_r Q^{(r)}_{j-1}
+
q_j q_j^\top ,
\]

\[
Q^{(w)}_t
=
\lambda_w Q^{(w)}_{t-1}
+
k_t k_t^\top .
\]

The first trace estimates historical **read demand**, while the second estimates historical **write supply**.

To keep the additional state small, both traces can be represented in low-rank form:

\[
Q^{(r)}
\approx
U_r \operatorname{diag}(a_r) U_r^\top ,
\qquad
Q^{(w)}
\approx
U_w \operatorname{diag}(a_w) U_w^\top ,
\]

with \(U_r,U_w\in\mathbb{R}^{d_k\times r}\) and \(r\ll d_k\).

For an incoming write key \(k_t\), the two historical heats are

\[
h_r(k_t)
=
k_t^\top Q^{(r)} k_t
\approx
\sum_{i=1}^{r} a_{r,i}(u_{r,i}^\top k_t)^2 ,
\]

\[
h_w(k_t)
=
k_t^\top Q^{(w)} k_t
\approx
\sum_{i=1}^{r} a_{w,i}(u_{w,i}^\top k_t)^2 .
\]

---

## Demand–supply priority

After normalizing the two heats to comparable scales, define a mismatch priority

\[
p_{\mathrm{raw}}(k_t)
=
\frac{\hat h_r(k_t)}
{\epsilon+\hat h_w(k_t)} .
\]

A bounded version can be used in practice:

\[
p(k_t)
=
\frac{p_{\mathrm{raw}}(k_t)}
{1+p_{\mathrm{raw}}(k_t)} .
\]

The interpretation is:

| Read demand | Write supply | Priority |
|---|---|---|
| high | low | very high |
| high | high | moderate |
| low | high | low |
| low | low | low |

Thus, the mechanism prioritizes addresses that are **important to readers but under-served by writes**.

The Delta update can then use

\[
\tilde{\eta}_t
=
\min\left\{
\eta\left(1+\beta p(k_t)\right),
c
\right\},
\]

\[
M_t
=
M_{t-1}
+
\tilde{\eta}_t
\left(v_t-M_{t-1}k_t\right)k_t^\top .
\]

As in the current QD³ design, setting \(\beta=0\) exactly recovers ordinary Delta, while the cap \(c<2\) can preserve the same single-address non-expansion condition.

---

## Expected value and evaluation

This extension reframes QD³ as a **demand–supply allocator**: read history estimates where accuracy is valuable, while write history estimates where corrective opportunities have already been abundant. The model therefore allocates extra correction specifically to regions with high demand and insufficient write exposure.

A future study should compare: **read-only QD³**, **write-only trace**, **simple read+write addition**, **demand/supply ratio**, and **full-moment versus low-rank dual traces**. The most important test case is a controlled workload in which read frequency and write frequency are deliberately mismatched.

**Hypothesis:** the dual-trace version should help most when \(p_{\mathrm{query}}\neq p_{\mathrm{write}}\), especially when highly queried addresses receive relatively few write opportunities. It is not expected to provide a large advantage when reads and writes are uniformly distributed or closely matched.