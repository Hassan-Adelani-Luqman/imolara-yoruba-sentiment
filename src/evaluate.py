"""Shared evaluation: metrics, bootstrap confidence intervals, run records and results collection.

Every training script (baselines, RNN, transformers) reports through `evaluate_predictions`
and `save_run`, so all experiments are scored identically.

CLI:
  python -m src.evaluate collect results/kaggle/<exp>   # append a downloaded run to experiments.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data import LABELS

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CSV = ROOT / "results" / "experiments.csv"
RUN_KEY = ["exp_id", "seed", "split", "subset", "eval_diacritics"]


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1 (primary), weighted-F1 (SemEval-2023 official) and per-class P/R/F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(LABELS)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
    }
    for i, name in enumerate(LABELS):
        metrics[f"precision_{name}"], metrics[f"recall_{name}"], metrics[f"f1_{name}"] = p[i], r[i], f[i]
    return {k: round(float(v), 4) for k, v in metrics.items()}


def _fast_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """macro_f1 / weighted_f1 / accuracy from a confusion matrix; equal to compute_metrics but ~100x faster,
    which matters inside bootstrap loops."""
    k = len(LABELS)
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    if metric == "accuracy":
        return float(np.trace(cm) / cm.sum())
    tp, gold, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    denom = gold + pred
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    if metric == "macro_f1":
        return float(f1.mean())
    if metric == "weighted_f1":
        return float((f1 * gold).sum() / gold.sum())
    raise ValueError(f"unsupported bootstrap metric {metric!r}")


def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1", n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for one metric (resampling test items with replacement)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    scores = [_fast_score(y_true[idx], y_pred[idx], metric)
              for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def paired_bootstrap(y_true, pred_a, pred_b, metric: str = "macro_f1", n: int = 1000, seed: int = 0) -> float:
    """Share of bootstrap samples in which system A does NOT beat system B (an approximate p-value)."""
    y_true, pred_a, pred_b = (np.asarray(a, dtype=int) for a in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    losses = 0
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        losses += _fast_score(y_true[idx], pred_a[idx], metric) <= _fast_score(y_true[idx], pred_b[idx], metric)
    return losses / n


def confusion(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return pd.DataFrame(cm, index=[f"gold_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])


def plot_confusion(y_true, y_pred, path: Path, title: str):
    """Row-normalised confusion matrix (recall per gold class), single-hue sequential blue, counts annotated."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    share = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{share[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(LABELS)), LABELS, fontsize=8, color="#52514e")
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8, color="#52514e")
    ax.set_xlabel("predicted", fontsize=8, color="#52514e")
    ax.set_ylabel("gold", fontsize=8, color="#52514e")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9, color="#0b0b0b")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_predictions(df: pd.DataFrame, probs: np.ndarray | None = None, **record) -> tuple[dict, pd.DataFrame]:
    """Score predictions for one split.

    df must contain id, text, label_id and pred_id. Extra keyword arguments (exp_id, model, seed,
    split, eval_diacritics, ...) are stored in the record. Returns (record, predictions table).
    """
    preds = df[["id", "text", "label_id", "pred_id"]].copy()
    preds["gold"] = preds["label_id"].map(dict(enumerate(LABELS)))
    preds["pred"] = preds["pred_id"].map(dict(enumerate(LABELS)))
    if probs is not None:
        for i, name in enumerate(LABELS):
            preds[f"p_{name}"] = np.round(probs[:, i], 4)
    metrics = compute_metrics(preds["label_id"], preds["pred_id"])
    lo, hi = bootstrap_ci(preds["label_id"], preds["pred_id"])
    record = {**record, "n": len(preds), **metrics, "macro_f1_ci_low": round(lo, 4), "macro_f1_ci_high": round(hi, 4)}
    return record, preds


def score_subsets(df: pd.DataFrame, probs: np.ndarray | None = None, **record) -> tuple[list[dict], pd.DataFrame]:
    """Score a split twice: all rows, and the 'clean' rows that do not duplicate a training tweet.

    df needs id, text, label_id, pred_id and overlap_train (see data.load_eval_split). About 13% of dev
    and 8.5% of test tweets duplicate a training tweet, which inflates scores on the full split.
    """
    rec_all, preds = evaluate_predictions(df, probs, **record, subset="all")
    clean = ~df["overlap_train"].to_numpy(dtype=bool)
    rec_clean, _ = evaluate_predictions(df[clean], None if probs is None else probs[clean], **record, subset="clean")
    preds["overlap_train"] = df["overlap_train"].to_numpy(dtype=bool)
    return [rec_all, rec_clean], preds


def save_run(output_dir: Path, records: list[dict], predictions: dict[str, pd.DataFrame], extra: dict | None = None):
    """Write metrics.json (list of records) and predictions_<name>.csv files into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    for name, preds in predictions.items():
        preds.to_csv(output_dir / f"predictions_{name}.csv", index=False)
    if extra:
        (output_dir / "run_info.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False))


def append_records(records: list[dict], csv_path: Path = EXPERIMENTS_CSV) -> pd.DataFrame:
    """Add records to experiments.csv, replacing earlier rows with the same (exp_id, seed, split, eval_diacritics)."""
    new = pd.DataFrame(records)
    for col in RUN_KEY:
        if col not in new:
            new[col] = {"eval_diacritics": "original", "subset": "all"}.get(col)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=RUN_KEY, keep="last")
    else:
        merged = new
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    return merged


def collect(run_dir: Path, csv_path: Path = EXPERIMENTS_CSV) -> int:
    """Append every metrics.json found under run_dir (e.g. a downloaded Kaggle run) to experiments.csv."""
    records = []
    for path in sorted(Path(run_dir).rglob("metrics.json")):
        records.extend(json.loads(path.read_text()))
    if records:
        append_records(records, csv_path)
    print(f"collected {len(records)} records from {run_dir} into {csv_path}")
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="append downloaded run metrics to results/experiments.csv")
    c.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "collect":
        return 0 if collect(args.run_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
