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