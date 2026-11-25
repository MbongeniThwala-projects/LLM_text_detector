import argparse
import json
import joblib
import pandas as pd

from pathlib import Path

from detector.config import load_config
from detector.eval.metrics import threshold_at_fpr
from detector.features.augment import add_truncated_copies, truncate_random
from detector.models.tfidf import fit, score

def load_run_cfg(cfg: dict, name: str) -> dict:
    """"""
    if name not in cfg["runs"]:
        raise KeyError(f"Unknown run '{name}'. Known runs: {sorted(cfg['runs'])}")
    return {**cfg["default"], **cfg["runs"][name]}

def fit_kwargs(run_cfg: dict) -> dict:
    return {
        "C": run_cfg["C"],
        "ngram_range": tuple(run_cfg["ngram_range"]),
        "min_df": run_cfg["min_df"],
        "sublinear_tf": run_cfg["sublinear_tf"]
    }

def train_run(df: pd.DataFrame, run_cfg: dict, eval_cfg: dict):
    """
    Fit on the train split, then fix the decision threshold from the validation split

    the threshold is set once, in here, on a mixed length copy of val, and saved with the model. 
    Evaluation reads it back and never picks its own
    """
    trian = df[df["split"] == "train"]
    val = df[df["split"] == 'val']
    if run_cfg["augmentation_copies"] > 0:
        train = add_truncated_copies(
            train,
            run_cfg["augmentation_copies"],
            run_cfg["augmentation_seed"],
            run_cfg["augmentation_min_words"]
        )
    vec, clf = fit(train["text"], train["label"], **fit_kwargs(run_cfg))

    mixed = truncate_random(val, eval_cfg["mixed_length_seed"], eval_cfg["mixed_min_words"])
    threshold = threshold_at_fpr(mixed["label"], score(vec, clf, mixed["text"]), eval_cfg["fpr_budget"])
    meta = {
        "run_cfg": run_cfg,
        "fpr_budget": eval_cfg["fpr_budget"],
        "threshold": threshold,
        "n_train_rows": int(len(train))
    }
    return (vec, clf), meta

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train named runs from configs/model.yaml")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df = pd.read_parquet(cfg["data_path"])
    for name in args.runs:
        model, meta = train_run(df, load_run_cfg(cfg, name), cfg["evaluation"])
        out = Path(cfg["model_dir"]) / name
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: trained on {meta['n_train_rows']} rows, threshold {meta['threshold']:.4f}")

if __name__ == "__main__":
    main()