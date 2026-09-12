import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_wikitext2 import DATA, ROOT, TOKEN


def sequence(split, word_id):
    path = DATA / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(path, columns=["text"])
    tokens = TOKEN.findall(" ".join(frame.text.fillna("")).lower())
    return np.asarray([word_id[token] for token in tokens if token in word_id], dtype=np.int32)


def main():
    source = ROOT / "wikitext2_qd3.npz"
    data = np.load(source, allow_pickle=False)
    vocabulary = np.asarray(data["vocabulary"])
    word_id = {word: index for index, word in enumerate(vocabulary.tolist())}
    validation_sequence = sequence("validation", word_id)
    test_sequence = sequence("test", word_id)
    output = ROOT / "wikitext2_ordered_qd3.npz"
    np.savez_compressed(
        output,
        keys=data["keys"],
        values=data["values"],
        frequency_val=data["frequency_val"],
        frequency_test=data["frequency_test"],
        vocabulary=vocabulary,
        validation_sequence=validation_sequence,
        test_sequence=test_sequence,
        base_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
        protocol_json=np.asarray(json.dumps({
            "query": "first part of a disjoint contiguous token window",
            "test": "immediately following part of the same window",
            "shuffled_control": "same query-token multiset in random order",
        }, sort_keys=True)),
    )
    print(f"validation ordered tokens: {len(validation_sequence):,}")
    print(f"test ordered tokens:       {len(test_sequence):,}")
    print(f"prepared: {output}")


if __name__ == "__main__":
    main()
