"""E9 learning curve for the TF-IDF baseline: clean-dev macro-F1 vs number of training tweets.

  python -m src.learning_curve

Uses E1c's selected configuration (word+char TF-IDF + logistic regression, C=3) without re-tuning, so only the
amount of data changes. Each seed draws a different random subset. The AfriBERTa side of E9 runs on Kaggle
(configs/e9_afriberta_*.yaml); the 100% points are E1c and E4.
"""
from __future__ import annotations

import pandas as pd

from src.baselines import build_model
from src.data import load_eval_split, load_split, preprocess_frame
from src.evaluate import append_records, compute_metrics

FRACTIONS = (0.10, 0.25, 0.50)
SEEDS = (42, 43, 44)
SPEC = {"model": "logreg", "features": "word+char"}
PARAMS = {"C": 3, "class_weight": None}           # E1c's grid-search winner


def main():
    train_all = preprocess_frame(load_split("yor", "train"))
    dev = preprocess_frame(load_eval_split("yor", "dev"))
    clean = dev[~dev["overlap_train"]]
    records = []
    for frac in FRACTIONS:
        n = round(frac * len(train_all))
        for seed in SEEDS:
            train = train_all.sample(n=n, random_state=seed)
            model = build_model(SPEC, PARAMS, seed).fit(train["text"], train["label_id"])
            for subset, df in (("all", dev), ("clean", clean)):
                m = compute_metrics(df["label_id"], model.predict(df["text"]))
                records.append({"exp_id": f"e9_tfidf_{round(frac * 100):03d}", "model": "logreg:word+char",
                                "seed": seed, "split": "dev", "subset": subset, "eval_diacritics": "original",
                                "n_train": n, "params": str(PARAMS), **m})
            print(f"{frac:4.0%} ({n:5d} tweets) seed {seed}: clean-dev macro-F1 {records[-1]['macro_f1']:.4f}", flush=True)
    append_records(records)


if __name__ == "__main__":
    main()
