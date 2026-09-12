"""Reproduces the tables in README.md.

Two task families, six controls.  Writes are uniform over the domain; queries are
concentrated (or uniform, as the boundary check).  Metric = error under the ACTUAL
query distribution.  Every method gets the same hyper-parameter grid.

    python experiments.py            # ~5 min on a laptop CPU
    python experiments.py --full     # 3 seeds, wider grid
"""
import numpy as np, argparse

DK = 64
ETAS = [0.02, 0.05, 0.1, 0.2]
BETAS = [0, 1, 3, 10, 30]

# ────────────────────────────────────────────────────────────── task A: 2-D field
def task_a(seed, qmode, T=3000):
    DV = 8
    rg = np.random.default_rng(0)
    W = rg.normal(scale=20.0, size=(2, DK)); B = rg.uniform(0, 2*np.pi, DK)
    A1 = rg.normal(size=(DV, 16))
    def feat(P):
        F = np.cos(np.atleast_2d(P) @ W + B)
        return F / np.linalg.norm(F, axis=1, keepdims=True)
    def target(P):
        P = np.atleast_2d(P); ar = np.arange(1, 5)
        Z = np.concatenate([np.cos(2*np.pi*np.outer(P[:,0],ar)), np.sin(2*np.pi*np.outer(P[:,1],ar)),
                            np.cos(2*np.pi*np.outer(P[:,0]+P[:,1],ar)), np.sin(2*np.pi*np.outer(P[:,0]-P[:,1],ar))],1)
        return Z @ A1.T
    def qsample(n, sd):
        r = np.random.default_rng(sd)
        if qmode == "uniform": return r.uniform(0, 1, (n, 2))
        c = np.array([[.25,.30],[.70,.75]]); i = r.integers(0, 2, n)
        return np.clip(c[i] + 0.06*r.normal(size=(n,2)), 0, 1)
    r = np.random.default_rng(seed)
    Pw = r.uniform(0, 1, (T, 2))
    Kw = feat(Pw); Vw = target(Pw) + 0.05*r.normal(size=(T, DV))
    Kq = feat(qsample(T//3 + 2, seed+7))
    Pt = qsample(3000, seed+555); Kt, Vt = feat(Pt), target(Pt)
    Ko = feat(qsample(4000, seed+99))
    return Kw, Vw, Kq, Kt, Vt, Ko, DV

# ──────────────────────────────────────────────────────── task B: discrete recall
def task_b(seed, qmode, T=3000, N=300):
    DV = 16
    r = np.random.default_rng(seed)
    K = r.normal(size=(N, DK)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = r.normal(size=(N, DV))
    a = 0.0 if qmode == "uniform" else 1.2
    p = 1.0/np.arange(1, N+1)**a; p /= p.sum()
    pq = np.zeros(N); pq[r.permutation(N)] = p
    wi = r.integers(0, N, T); qi = r.choice(N, T//3 + 2, p=pq)
    ti = np.random.default_rng(seed+555).choice(N, 3000, p=pq)
    oi = np.random.default_rng(seed+99).choice(N, 4000, p=pq)
    return K[wi], V[wi], K[qi], K[ti], V[ti], K[oi], DV

TASKS = {"A (2-D field)": task_a, "B (discrete recall)": task_b}

# ───────────────────────────────────────────────────────────────────── the runner
def run(mode, Kw, Vw, Kq, DV, eta, beta, r=4, eta_u=0.05, lam=0.999, Ko=None):
    T = len(Kw)
    M = np.zeros((DV, DK)); rg = np.random.default_rng(0)
    U = rg.normal(scale=0.1, size=(DK, r)); s = np.zeros(r); nrm = 1e-12
    Aq = np.zeros((DK, DK)); mxf = 1e-12                         # full trace
    nw = np.zeros(DK)                                            # write frequency
    Qw = np.zeros((r, DK)); ptr = 0; mxw = 1e-12                 # sliding window
    dg = np.zeros(DK)                                            # diagonal trace
    R = np.random.default_rng(7).normal(scale=1/np.sqrt(32), size=(32, DK))
    G = np.zeros((32, 32)); mxs = 1e-12                          # random projection
    qcur = Kq[0]; qi = 0
    if mode == "oracle":
        Ao = (Ko.T @ Ko) / len(Ko); Ao /= np.trace(Ao)/DK
    for t in range(T):
        k = Kw[t]
        if   mode == "plain":  g = 1.0
        elif mode == "oracle": g = 1 + beta*float(k @ Ao @ k)
        elif mode == "write":  g = 1 + beta*float((nw/max(nw.max(),1e-12)) @ (k*k))
        elif mode == "window": g = 1 + beta*float((Qw@k)@(Qw@k))/r/max(mxw,1e-12)
        elif mode == "diag":   g = 1 + beta*float((dg/max(dg.max(),1e-12)) @ (k*k))
        elif mode == "proj":   Rk = R@k; g = 1 + beta*float(Rk@G@Rk)/max(mxs,1e-12)
        elif mode == "full":   g = 1 + beta*float(k @ Aq @ k)/max(mxf,1e-12)
        elif mode == "qdelta": g = 1.0
        else:                  g = 1 + beta*float(((U.T@k)**2*s).sum())/max(nrm,1e-12)
        if mode == "qdelta":                                     # per-step query coupling
            M -= eta*(M@k)[:,None]@k[None,:] + beta*eta*(M@qcur)[:,None]@k[None,:]
            M += eta*np.outer(Vw[t], k)
        else:
            M += eta*g*np.outer(Vw[t] - M@k, k)
        nw = 0.999*nw + k*k
        if t % 3 == 0:
            q = Kq[qi]; qi += 1; qcur = q; _ = M @ q
            p = U.T@q; U += eta_u*np.outer(q - U@p, p); s = lam*s + p*p
            p2 = U.T@q; nrm = max(nrm, float((s*p2*p2).sum()))
            Aq = lam*Aq + np.outer(q, q); mxf = max(mxf, float(q@Aq@q))
            Qw[ptr % r] = q; ptr += 1; mxw = max(mxw, float((Qw@q)@(Qw@q))/r)
            dg = lam*dg + q*q
            Rq = R@q; G = lam*G + np.outer(Rq, Rq); mxs = max(mxs, float(Rq@G@Rq))
    return M

def err(M, Kt, Vt):
    E = (Kt @ M.T) - Vt
    return float(np.sqrt((E**2).sum(1).mean()) / np.sqrt((Vt**2).sum(1).mean()))

MODES = [("plain","plain delta"),("write","write-frequency gain"),("window","sliding window r=4"),
         ("diag","diagonal trace"),("proj","random projection r=32"),("qdelta","Q-Delta (per-step query)"),
         ("ours","OURS (query trace r=4)"),("oracle","oracle (true query dist.)")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
    seeds = [1,2,3] if ap.parse_args().full else [1,2]
    for tname, tf in TASKS.items():
        for qmode in ["concentrated", "uniform"]:
            print(f"\n=== task {tname} | queries: {qmode} ===")
            base = None
            for key, label in MODES:
                grid = [0] if key == "plain" else BETAS
                best = min(np.mean([err(run(key, *tf(sd,qmode)[:3], tf(sd,qmode)[6], e, b,
                                            Ko=tf(sd,qmode)[5]), *tf(sd,qmode)[3:5])
                                    for sd in seeds]) for e in ETAS for b in grid)
                if base is None: base = best
                print(f"  {label:28s} {best:.4f}   {100*(base-best)/base:+6.1f}%", flush=True)
