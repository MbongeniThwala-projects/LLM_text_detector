import pandas as pd

from detector.eval.metrics import summarise, threshold_at_fpr
from detector.models.tfidf import fit, score

def truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])

def leave_one_topic_out(df: pd.DataFrame, topic_col: str = "topic", fpr: float = 0.01)
    """train on all topics but one and test on the held-out topic
    threshold comes from validation rows of the other topics
    """