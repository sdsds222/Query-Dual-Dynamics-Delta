"""Prepare a leakage-controlled WikiText-2 workload matching the word task."""
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "wikitext2"
TOKEN = re.compile(r"[a-z_]{2,}")


def read_tokens(split):
    path = DATA / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(path, columns=["text"])
    return TOKEN.findall(" ".join(frame.text.fillna("")).lower())


def effective_rank(moment):
    trace = float(np.trace(moment))
    return trace * trace / max(float(np.sum(moment * moment.T)), 1e-12)


def fit_error(k, v, fw, ew):
    fw, ew = fw / fw.sum(), ew / ew.sum()
    m, c = (k * fw[:, None]).T @ k, (k * fw[:, None]).T @ v
    ridge = 1e-6 * max(float(np.trace(m)) / k.shape[1], 1e-12)
    a = np.linalg.solve(m + ridge * np.eye(k.shape[1]), c).T
    e2 = np.sum((k @ a.T - v) ** 2, axis=1)
    return float(np.sqrt(ew @ e2 / (ew @ np.sum(v * v, axis=1))))


def main():
    train, valid, test = read_tokens("train"), read_tokens("validation"), read_tokens("test")
    count = collections.Counter(train)
    vocab = [word for word, _ in count.most_common(3000)]
    word_id = {word: i for i, word in enumerate(vocab)}
    key_context = {word: i for i, (word, _) in enumerate(count.most_common(256))}
    value_context = {word: i for i, (word, _) in enumerate(count.most_common(64))}
    ids = [word_id.get(word, -1) for word in train]
    key_ids = [key_context.get(word, -1) for word in train]
    value_ids = [value_context.get(word, -1) for word in train]
    keys = np.zeros((len(vocab), 256), dtype=np.float64)
    values = np.zeros((len(vocab), 64), dtype=np.float64)
    for position, item in enumerate(ids):
        if item < 0:
            continue
        for j in range(max(0, position - 2), min(len(ids), position + 3)):
            if j != position and key_ids[j] >= 0:
                keys[item, key_ids[j]] += 1
        for j in range(max(0, position - 10), min(len(ids), position + 11)):
            if j != position and value_ids[j] >= 0:
                values[item, value_ids[j]] += 1
    keys = np.log1p(keys)
    keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    keep = (np.linalg.norm(keys, axis=1) > 0) & (np.linalg.norm(values, axis=1) > 0)
    keys, values = keys[keep], values[keep]
    vocab = np.asarray(vocab)[keep]

    diagnostics = {"items": len(vocab), "train_tokens": len(train)}
    frequencies = {}
    uniform = np.full(len(vocab), 1 / len(vocab))
    unweighted = (keys * uniform[:, None]).T @ keys
    for name, tokens in (("validation", valid), ("test", test)):
        counts = collections.Counter(tokens)
        raw = np.asarray([counts[word] for word in vocab], dtype=np.float64)
        frequencies[name] = raw / raw.sum()
        ordered = np.sort(frequencies[name])[::-1]
        weighted = (keys * frequencies[name][:, None]).T @ keys
        plain = fit_error(keys, values, uniform, frequencies[name])
        oracle = fit_error(keys, values, frequencies[name], frequencies[name])
        diagnostics[name] = {
            "coverage": float(raw.sum() / len(tokens)),
            "top_10_percent": float(ordered[:max(1, len(vocab)//10)].sum()),
            "effective_rank_ratio": effective_rank(unweighted) / effective_rank(weighted),
            "oracle_gain": (plain - oracle) / plain,
        }

    paths = sorted(DATA.glob("*.parquet"))
    source_hashes = {}
    for path in paths:
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = ROOT / "wikitext2_qd3.npz"
    np.savez_compressed(
        output,
        keys=keys.astype(np.float32),
        values=values.astype(np.float32),
        frequency_val=frequencies["validation"],
        frequency_test=frequencies["test"],
        vocabulary=vocab,
        diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
        source_hashes_json=np.asarray(json.dumps(source_hashes, sort_keys=True)),
    )
    print(json.dumps(diagnostics, indent=2))
    print(f"prepared: {output}")


if __name__ == "__main__":
    main()
