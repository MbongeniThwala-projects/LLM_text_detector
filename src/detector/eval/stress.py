import pandas as pd

from detector.eval.metrics import summarise, threshold_at_fpr
from detector.models.tfidf import fit, score

def truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])

def leave_one_topic_out(df: pd.DataFrame, topic_col: str = "topic", fpr: float = 0.01, augment=None) -> pd.DataFrame:
    """train on all topics but one and test on that topic
    threshold comes from validation rows of the other topics.
    that one topic rows come from train and val only
    """
    pool = df[df["split"].isin(["train", "val"])]
    rows = []
    for t in sorted(df[topic_col].unique()):
        tr = df[(df["split"] == "train") & (df[topic_col] != t)]
        cal = df[(df["split"] == "val") & (df[topic_col] != t)]
        held = pool[pool[topic_col] == t]
        if augment is not None:
            tr = augment(tr)        # applied to training rows only
        vec, clf = fit(tr["text"], tr["label"])
        thr = threshold_at_fpr(cal["label"], score(vec, clf, cal["text"]), fpr)
        r = summarise(held["label"], score(vec, clf, held["text"]), thr)
        rows.append({"held_out_topic": t, "n": len(held), "ai": int(held["label"].sum()), **r})
    return pd.DataFrame(rows)