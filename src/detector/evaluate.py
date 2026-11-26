import argparse
import json
import pandas as pd
import joblib

from pathlib import Path

from detector.config import load_config
from detector.eval.metrics import summarise
from detector.eval.stress import leave_one_topic_out, truncate_words
from detector.features.augment import add_truncated_copies, truncate_random
from detector.models.tfidf import score
from detector.train import fit_kwargs, load_run_cfg

def evaluate_run(df: pd.DataFrame, model, meta: dict, eval_cfg: dict, split: str = "val") -> dict:
    vec, clf = model
    d = df[df["split"] == split]
    thr = meta["threshold"]

    report = {"split": split, "threshold": thr}
    report["full_text"] = summarise(d["label"], score(vec, clf, d["text"]), thr)

    mixed = truncate_random(d, eval_cfg["mixed_length_seed"], eval_cfg["mixed_min_word"])
    report["mixed_length"] = summarise(mixed["label"], score(vec, clf, mixed["text"]), thr)

    report["by_length"] = {}
    for n in eval_cfg["length_grid"]:
        texts = d["text"].map(lambda t: truncate_words(t, n))
        report["by_length"][str(n)] = summarise(d["label"], score(vec, clf, texts), thr)
    return report

def evaluate_topics(df: pd.DataFrame, run_cfg: dict, eval_cfg: dict) -> list[dict]:
    augment = None
    if run_cfg["augmentation_copies"] > 0:
        augment = lambda d: add_truncated_copies(
            d, 
            run_cfg["augmentation_copies"],
            run_cfg["augmentation_seed"],
            run_cfg["augmentation_min_words"]
        )
    table = leave_one_topic_out(df, fpr=eval_cfg["fpr_budget"], augment=augment, fit_kwargs=fit_kwargs(run_cfg))
    return table.round(4).to_dict("records")

def _to_json(o): return o.item() if hasattr(o, "item") else str(o)

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run")
    parser.add_argument("run")
    parser.add_argument("--config", default="config/data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--topics", default="store_code", help="also run; leave one topic out")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df = pd.read_parquet(cfg["data_path"])
    folder = Path(cfg["model_dir"]) / args.run
    model = joblib.load(folder / "model.joblib")
    meta = json.loads((folder / "meta.json").read_text())

    if args.split == "test":
        print("NB: the test spllit is meant to be used once, for the final comparison")

    report = evaluate_run(df, model, meta, cfg["evaluation"], args.split)
    if args.topics:
        report["leave_one_topic_out"] = evaluate_topics(df, load_run_cfg(cfg, args.run), cfg["evaluation"])

    out = Path(cfg["report_dir"])
    out.mkdir(parents=True, exist_ok=True)

    