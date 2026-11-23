import numpy as np
import pandas as pd

from detector.data.cluster import cluster_near_duplicates
from detector.data.dedupe import collapse_exact
from detector.data.split import assert_no_leakage, assign_split

def _random_texts(n: int, n_words: int = 120, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(2000)]
    return [" ".join(rng.choice(vocab, n_words)) for _ in range(n)]

def test_conflicting_labels_are_dropped():
    raw = pd.DataFrame(
        {
            "text": ["Same essay.", "same   essay.", "Another essay."],
            "label": [0, 1, 1],
            "source": ["a", "b", "a"]
        }
    )
    unique, stats = collapse_exact(raw)
    assert list(unique["text"]) == ["Another essay."]
    assert stats["conflicting_texts_dropped"] == 1

def test_near_duplicates_never_straddle_splits():
    base = _random_texts(60)
    near = []
    for t in base:
        words = t.split()
        words[10], words[70] = "changed1", "changed2"
        near.append(" ".join(words))
    df = pd.DataFrame({"text": base + near, "label": ([0] * 30 + [1] * 30) * 2})

    df["cluster_id"] = cluster_near_duplicates(df["text"].tolist())
    assert df["cluster_id"].nunique() == 60

    df["split"] = assign_split(df, seed=1, n_folds=5, train_folds=3, val_folds=1)