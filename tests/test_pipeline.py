import numpy as np
import pandas as pd

from detector.evaluate import evaluate_run
from detector.train import train_run

RUN_CFG = {
    "ngram_range": [1, 2],
    "min_df": 1,
    "sublinear_tf": True,
    "C": 10,
    "augmentation_copies": 1,
    "augmentation_min_words": 20,
    "augmentation_seed": 0
}
EVAL_CFG = {
    "fpr_budget": .05,
    "length_grid": [20, 40],
    "mixed_length_seed": 1,
    "mixed_min_words": 20
}
def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    human = [f"h{i}" for i in range]
    ai= [f"a{i}" for i in range]
    rows = []
    for i in range(n):
        label = i % 2
        rows.append(
            {
                "text": " ".join(rng.choice(ai if label else human, 60)),
                "label": label,
                "split": ["train", "train", "val", "test"][(i // 2) % 4], 
            }
        )
    return pd.DataFrame(rows)

def test_evaluation_uses_the_saved_threshold():
    df = _frame()
    model, meta = train_run(df, RUN_CFG, EVAL_CFG)

    blocked = evaluate_run(df, model, {**meta, "threshold": 2.0}, EVAL_CFG)
    assert blocked["full_text"]["tpr"] == .0
    assert blocked["full_text"]["fpr"] == .0

    normal = evaluate_run(df, model, meta, EVAL_CFG)
    assert normal["threshold"] == meta["threshold"]
    assert normal["full_text"]["tpr"] > .9