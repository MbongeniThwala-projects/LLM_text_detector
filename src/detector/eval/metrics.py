import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

def threshold_at_fpr(y_true, scores, fpr: float) -> float:
    y = np.asarray(y_true)
    s = np.asarray(scores)
    human = np.sort(s[y == 0])
    allowed = int(np.floor(fpr * len(human) + 1e-9)) #human text that we might wrongfully flag
    return float(human[len(human) - allowed - 1])

def rates_at_threshold(y_true, scores, threshold: float) -> dict:
    """share of AI caught (tpr) and share of human texts wrongly flagged (fpr)"""
    y = np.asarray(y_true)
    flagged = np.asarray(scores) > threshold
    return {"tpr": float(flagged[y == 1].mean()), "fpr": float(flagged[y == 0].mean())}

def summarise(y_true, scores, threshold:float | None = None) -> dict:
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores))
    }
    if threshold is not None:
        out.update(rates_at_threshold(y_true, scores, threshold))
    return out