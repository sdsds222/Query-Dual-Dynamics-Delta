"""
Query-Traced Fast Weights (QTFW) -- reference implementation.

A fast-weight memory M has fixed capacity, so it cannot store everything.
Existing rules decide what to keep from the *current token*.  QTFW decides it
from the *distribution of past queries*, tracked in a second, tiny fast weight
that uses the very same delta rule.

    main  M += eta * g * (v - M k) k^T          what is stored
    aux   p  = U^T q ; U += eta_u (q - U p) p^T what is asked  (same delta rule)
          s  = lam * s + p * p
    couple g = 1 + beta * ||diag(sqrt(s)) U^T k||^2 / norm

Equivalent to fitting M by least squares weighted by the query distribution
rather than by the write distribution.

MIT licensed.  See README.md.
"""
import numpy as np


class QueryTracedMemory:
    """Fast-weight memory whose write gain is modulated by a query trace.

    Parameters
    ----------
    d_k, d_v : int      key / value dimension
    rank     : int      rank of the query trace (r=4 works well; r=2 is enough for most of the gain)
    eta      : float    main write step size
    beta     : float    coupling strength.  beta=0 recovers the plain delta rule exactly.
    eta_u    : float    aux (query-subspace) learning rate
    lam      : float    decay of the query-energy accumulator
    """

    def __init__(self, d_k, d_v, rank=4, eta=0.05, beta=10.0,
                 eta_u=0.05, lam=0.999, seed=0):
        rng = np.random.default_rng(seed)
        self.M = np.zeros((d_v, d_k))
        self.U = rng.normal(scale=0.1, size=(d_k, rank))
        self.s = np.zeros(rank)
        self.eta, self.beta, self.eta_u, self.lam = eta, beta, eta_u, lam
        self._norm = 1e-12                      # running max, keeps g scale-free

    # ---------------------------------------------------------------- write
    def heat(self, k):
        """How strongly k falls in the directions that have been queried."""
        p = self.U.T @ k
        return float((self.s * p * p).sum()) / max(self._norm, 1e-12)

    def write(self, k, v):
        """Store (k, v).  Costs one r-dim projection more than a plain delta write."""
        g = 1.0 + self.beta * self.heat(k)
        self.M += self.eta * g * np.outer(v - self.M @ k, k)

    # ----------------------------------------------------------------- read
    def read(self, q, trace=True):
        """Return M q.  The read path itself is unchanged; the trace update is a
        side effect and may be done asynchronously."""
        out = self.M @ q
        if trace:
            self.observe_query(q)
        return out

    def observe_query(self, q):
        """Delta-rule update of the query subspace (Oja without orthogonalisation)."""
        p = self.U.T @ q
        self.U += self.eta_u * np.outer(q - self.U @ p, p)
        self.s = self.lam * self.s + p * p
        p2 = self.U.T @ q
        self._norm = max(self._norm, float((self.s * p2 * p2).sum()))

    # ---------------------------------------------------------------- prior
    def set_query_prior(self, Q):
        """Warm-start the trace from a log of historical queries (deployment mode C).
        Without this the memory cold-starts as a plain delta rule (g == 1)."""
        for q in Q:
            self.observe_query(q)
