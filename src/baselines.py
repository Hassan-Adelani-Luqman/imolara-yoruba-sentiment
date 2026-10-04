"""Classical baselines (E0–E1): majority class and TF-IDF + linear classifiers.

  python -m src.baselines                          # every experiment in configs/baselines.yaml
  python -m src.baselines --only e1c_wordchar_lr   # one experiment
  python -m src.baselines --final                  # Phase 6 only: also score the test split

Hyperparameters are chosen on clean-dev macro-F1. Outputs go to results/baselines/<exp_id>/ and
results/experiments.csv. These models are deterministic, so one run (seed 0) is reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.data import LABELS, load_eval_split, load_split, preprocess_frame, training_frame
from src.evaluate import append_records, compute_metrics, save_run, score_subsets

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "baselines"


def whitespace_tokens(text: str) -> list[str]:
    """Split on whitespace. sklearn's default token pattern drops one-letter words such as 'o' and
    breaks words at combining tone marks ('ọ̀rẹ́' -> 'rẹ'), which destroys Yoruba tokens."""
    return text.split()


def build_features(kind: str):
    word = TfidfVectorizer(tokenizer=whitespace_tokens, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True)
    return {"word": word, "char": char, "word+char": FeatureUnion([("word", word), ("char", char)])}[kind]


def build_model(spec: dict, params: dict, seed: int = 0):
    if spec["model"] == "majority":
        return DummyClassifier(strategy="most_frequent")
    if spec["model"] == "logreg":
        clf = LogisticRegression(max_iter=3000, random_state=seed, **params)
    elif spec["model"] == "linearsvc":
        clf = LinearSVC(random_state=seed, **params)
    else:
        raise ValueError(f"unknown model {spec['model']!r}")
    return Pipeline([("features", build_features(spec["features"])), ("clf", clf)])


def param_grid(spec: dict) -> list[dict]:
    grid = spec.get("grid") or {}
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())] or [{}]


def predict_with_probs(model, texts) -> tuple[np.ndarray, np.ndarray | None]:
    pred = model.predict(texts)
    probs = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
    return pred, probs


def top_features(model, k: int = 20) -> dict[str, list]:
    """Highest-weighted n-grams per class for a linear model (interpretability)."""
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    if clf is None or not hasattr(clf, "coef_"):
        return {}
    names = model.named_steps["features"].get_feature_names_out()
    return {label: [(names[j], round(float(clf.coef_[i, j]), 3)) for j in np.argsort(clf.coef_[i])[::-1][:k]]
            for i, label in enumerate(LABELS)}


def build_train(diacritics: str, style: str) -> pd.DataFrame:
    """Training frame in one diacritic form, or augmented with several forms ('mixed', 'mixed3'; E6)."""
    return training_frame(load_split("yor", "train"), diacritics, style)


def run_experiment(exp_id: str, spec: dict, eval_diacritics: list[str], final: bool = False) -> list[dict]:
    start = time.time()
    style = spec.get("train_style", "semeval")
    train = build_train(spec.get("train_diacritics", "original"), style)
    dev_raw = load_eval_split("yor", "dev")
    dev = preprocess_frame(dev_raw, "original", "semeval")          # always scored in the test format
    clean = ~dev["overlap_train"].to_numpy(dtype=bool)

    # grid search on clean dev
    grid_rows, best = [], None
    for params in param_grid(spec):
        model = build_model(spec, params).fit(train["text"], train["label_id"])
        score = compute_metrics(dev.loc[clean, "label_id"], model.predict(dev.loc[clean, "text"]))["macro_f1"]
        grid_rows.append({**{k: str(v) for k, v in params.items()}, "clean_dev_macro_f1": score})
        if best is None or score > best[0]:
            best = (score, params, model)
    _, best_params, model = best

    # score the selected model on all + clean dev (and test with --final) in every diacritic form
    base = {"exp_id": exp_id, "model": f"{spec['model']}:{spec.get('features', '-')}", "seed": 0,
            "train_style": style, "train_diacritics": spec.get("train_diacritics", "original"),
            "params": json.dumps(best_params), "notes": spec.get("notes", "")}
    records, predictions = [], {}
    for split in (["dev", "test"] if final else ["dev"]):
        raw = dev_raw if split == "dev" else load_eval_split("yor", "test")
        for mode in eval_diacritics:
            df = preprocess_frame(raw, mode, "semeval")
            df["pred_id"], probs = predict_with_probs(model, df["text"])
            recs, preds = score_subsets(df, probs, **base, split=split, eval_diacritics=mode)
            records.extend(recs)
            predictions[f"{split}_{mode}"] = preds
    runtime = round(time.time() - start, 1)
    for r in records:
        r["runtime_s"] = runtime

    info = {"spec": spec, "best_params": best_params, "grid": grid_rows, "n_train": len(train),
            "top_features": top_features(model)}
    save_run(OUT_DIR / exp_id, records, predictions, extra=info)
    append_records(records)
    headline = next(r for r in records if r["subset"] == "clean" and r["eval_diacritics"] == "original")
    print(f"{exp_id:22s} best {best_params}  clean-dev macro-F1 {headline['macro_f1']:.4f}  "
          f"(all-dev {records[0]['macro_f1']:.4f})  [{runtime}s]")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "baselines.yaml"))
    parser.add_argument("--only", nargs="+", help="experiment ids to run (default: all)")
    parser.add_argument("--final", action="store_true", help="also evaluate on the test split (Phase 6 only)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    for exp_id, spec in cfg["experiments"].items():
        if args.only and exp_id not in args.only:
            continue
        run_experiment(exp_id + ("_final" if args.final else ""), spec, cfg["eval_diacritics"], final=args.final)


if __name__ == "__main__":
    main()
