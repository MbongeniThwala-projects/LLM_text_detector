import pandas as pd

from detector.features.augment import add_truncated_copies

def test_truncated_copies_keep_label_and_prefixes():
    base = pd.DataFrame(
        {
            "text": [" ".join(f"a{i}" for i in range(80)), " ".join(f"b{i}" for i in range(200))],
            "label": [0, 1]
        }
    )
    out = add_truncated_copies(base, n_copies=3, seed=0, min_words=20)

    assert len(out) == 2 + 2 * 3 #keep original, have 3 copies of each
    for i in range(2, len(out)):
        orig = base.iloc[(i - 2) % 2]
        copy = out.iloc[i]
        copy_words, orig_words = copy["text"].split(), orig["text"].split()
        assert copy["label"] == orig["label"]
        assert orig_words[: len(copy_words)] == copy_words  # prefix
        assert 20 <= len(copy_words) <= len(orig_words)