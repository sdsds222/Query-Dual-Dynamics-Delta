"""Build the two real datasets from the local Python installation. No downloads.

  word     : word frequency (Zipf) from stdlib source; keys = distributional
             semantics (short-window co-occurrence, PCA), values = long-window profile
  module   : module dependency graph; keys = co-import context, values = coarse
             co-import profile; query distribution = real import frequency (power law)
"""
import glob, re, collections, numpy as np, sys, sysconfig, os

STDLIB = sysconfig.get_paths()["stdlib"]

def corpus(limit=4000):
    src = []
    for p in glob.glob(STDLIB + '/**/*.py', recursive=True)[:limit]:
        try: src.append(open(p, encoding='utf-8', errors='ignore').read())
        except Exception: pass
    return src

def build_word(V=3000, NC=256, DV=64, out='word.npz'):
    toks = re.findall(r"[a-z_]{2,}", " ".join(corpus()).lower())
    cnt = collections.Counter(toks)
    vocab = [w for w, _ in cnt.most_common(V)]; V = len(vocab)
    wid = {w: i for i, w in enumerate(vocab)}
    ck = {w: i for i, (w, _) in enumerate(cnt.most_common(NC))}
    cv = {w: i for i, (w, _) in enumerate(cnt.most_common(DV))}
    ids = [wid.get(t, -1) for t in toks]
    a_k = [ck.get(t, -1) for t in toks]; a_v = [cv.get(t, -1) for t in toks]
    CK = np.zeros((V, NC)); CV = np.zeros((V, DV))
    for i, a in enumerate(ids):
        if a < 0: continue
        for j in range(max(0, i-2), min(len(ids), i+3)):
            if j != i and a_k[j] >= 0: CK[a, a_k[j]] += 1
        for j in range(max(0, i-10), min(len(ids), i+11)):
            if j != i and a_v[j] >= 0: CV[a, a_v[j]] += 1
    CK = np.log1p(CK); CK /= np.maximum(np.linalg.norm(CK, axis=1, keepdims=True), 1e-9)
    CV /= np.maximum(CV.sum(1, keepdims=True), 1e-9)
    freq = np.array([cnt[w] for w in vocab], float); freq /= freq.sum()
    np.savez_compressed(out, CK=CK, CV=CV, freq=freq)
    return CK, CV, freq

def build_module(V=1200, NC=200, DV=48, out='module.npz'):
    pat = re.compile(r'^\s*(?:import\s+([a-zA-Z_][\w.]*)|from\s+([a-zA-Z_][\w.]*)\s+import)', re.M)
    docs = []
    for p in glob.glob(STDLIB + '/**/*.py', recursive=True)[:6000]:
        try: t = open(p, encoding='utf-8', errors='ignore').read()
        except Exception: continue
        m = {(a or b).split('.')[0] for a, b in pat.findall(t)}
        m = {x for x in m if x}
        if len(m) >= 2: docs.append(sorted(m))
    cnt = collections.Counter(x for d in docs for x in d)
    mods = [m for m, _ in cnt.most_common(V)]; V = len(mods)
    mid = {m: i for i, m in enumerate(mods)}
    cki = {m: i for i, (m, _) in enumerate(cnt.most_common(NC))}
    cvi = {m: i for i, (m, _) in enumerate(cnt.most_common(DV))}
    CK = np.zeros((V, NC)); CV = np.zeros((V, DV))
    for d in docs:
        for x in d:
            i = mid.get(x, -1)
            if i < 0: continue
            for y in d:
                if y != x and y in cki: CK[i, cki[y]] += 1
                if y != x and y in cvi and len(d) >= 4: CV[i, cvi[y]] += 1
    CK = np.log1p(CK); CK /= np.maximum(np.linalg.norm(CK, axis=1, keepdims=True), 1e-9)
    CV /= np.maximum(CV.sum(1, keepdims=True), 1e-9)
    freq = np.array([cnt[m] for m in mods], float); freq /= freq.sum()
    keep = (CV.sum(1) > 0) & (np.linalg.norm(CK, axis=1) > 0)
    CK, CV, freq = CK[keep], CV[keep], freq[keep]; freq /= freq.sum()
    np.savez_compressed(out, CK=CK, CV=CV, freq=freq)
    return CK, CV, freq

def diagnostics(CK, CV, freq, dk, name):
    """The two a-priori criteria: query concentration and key-subspace ratio."""
    V = len(freq); c = np.cumsum(np.sort(freq)[::-1])
    X = CK - CK.mean(0); _, _, Vt = np.linalg.svd(X, full_matrices=False)
    K = X @ Vt[:dk].T; K /= np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-9)
    ef = lambda A: (np.trace(A)**2) / np.trace(A @ A)
    r_all = ef((K.T @ K)/V); r_q = ef((K*freq[:, None]).T @ K)
    print(f"{name:8s} V={V:5d} dv={CV.shape[1]:3d} | query mass top1%={100*c[max(V//100-1,0)]:5.1f}% "
          f"top10%={100*c[V//10]:5.1f}% | key-subspace ratio {r_all/r_q:5.2f}x "
          f"(all {r_all:.1f} -> freq-weighted {r_q:.1f})")

if __name__ == "__main__":
    print("building word dataset ...");   a = build_word()
    print("building module dataset ..."); b = build_module()
    print()
    diagnostics(*a, 128, "word"); diagnostics(*b, 24, "module")
