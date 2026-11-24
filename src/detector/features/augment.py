import numpy as np
import pandas as pd

def truncate_random(df: pd.DataFrame, seed: int, min_words: int = 20) -> pd.DataFrame:
    """
    one trucated copy of every row, keeping the first n words
    """
    rng = np.random.default_rng(seed)
    words = df["text"].str.split()
    n_full = words.str.len().to_numpy()
    low = np.minimum(min_words, n_full)
    keep = np.exp(rng.uniform(np.log(low), np.log(n_full))).astype(int)
    keep = np.clip(keep, low, n_full)
    out = df.copy()
    out["text"] = [" ".join(w[:k]) for w, k in zip(words, keep)]
    return out

def add_truncated_copies(df: pd.DataFrame, n_copies: int, seed: int, min_words: int = 20) -> pd.DataFrame:
    copies = [truncate_random(df, seed + i, min_words) for i in range(n_copies)]
    return pd.concat([df, *copies], ignore_index=True)