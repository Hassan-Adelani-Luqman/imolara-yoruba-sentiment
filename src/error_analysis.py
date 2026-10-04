"""Phase 6 error analysis on the test set: AfriBERTa-large (best on dev) vs the TF-IDF baseline.

  python -m src.error_analysis     # writes results/error_analysis/{slices.md, errors_to_tag.csv} + figures

1. Sliced performance by objective, automatically measured properties of each tweet (code-switching, diacritics,
   length, duplicate of a training tweet, model confidence).
2. Error overlap between the two models (which errors are shared, which are fixed by the transformer).
3. A stratified sample of 100 misclassified tweets for manual tagging. Objective flags are filled in; the
   `suggested_category` column is a heuristic hint ONLY. The meaning-based categories (sarcasm/irony, proverb,
   slang, likely label noise, ...) go in `category` and are assigned by a human reader of Yoruba.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LABELS, has_tone_marks, has_underdots, load_split, nfc, strip_all_diacritics
from src.eda import ENGLISH_WORDLIST, english_tokens, load_all
from src.evaluate import compute_metrics, plot_confusion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "error_analysis"
MODEL_DIR = ROOT / "results" / "kaggle" / "e4_afriberta_large_final" / "outputs"
BASELINE_DIR = ROOT / "results" / "baselines" / "e1c_wordchar_lr_final"
SEED = 42
CATEGORIES = ["code-switching", "missing diacritics / ambiguity", "sarcasm or irony", "proverb / idiom",
              "slang / new word", "religious or greeting formula", "news / factual (neutral vs other)",
              "neutral-negative boundary", "likely label noise", "too short / no context", "other"]


def load_predictions() -> pd.DataFrame:
    model = pd.read_csv(MODEL_DIR / f"seed{SEED}" / "predictions_test_original.csv")
    base = pd.read_csv(BASELINE_DIR / "predictions_test_original.csv").set_index("id")
    df = model.rename(columns={"pred_id": "pred_model", "pred": "pred_model_label"})
    df["pred_base"] = df["id"].map(base["pred_id"])
    df["confidence"] = df[[f"p_{l}" for l in LABELS]].max(axis=1)
    df["raw_text"] = df["id"].map(load_split("yor", "test").set_index("id")["text"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = load_all()
    english = {w.strip().lower() for w in ENGLISH_WORDLIST.read_text(errors="ignore").split() if w.strip().isalpha()}
    diacritised = {strip_all_diacritics(w) for d in data.values() for t in d["norm"] for w in t.split()
                   if has_tone_marks(w) or has_underdots(w)}
    df["n_tokens"] = df["text"].str.split().str.len()
    df["n_english"] = df["text"].map(lambda t: len(english_tokens(t, english, diacritised)))
    df["code_switched"] = df["n_english"] >= 2
    df["has_diacritics"] = df["raw_text"].map(lambda t: has_tone_marks(nfc(t)) or has_underdots(nfc(t)))
    df["length_bin"] = pd.cut(df["n_tokens"], [0, 5, 10, 20, 40, 1000], labels=["1-5", "6-10", "11-20", "21-40", "41+"])
    df["confidence_bin"] = pd.cut(df["confidence"], [0, 0.5, 0.7, 0.9, 1.0], labels=["<0.5", "0.5-0.7", "0.7-0.9", ">0.9"])
    df["model_correct"] = df["pred_model"] == df["label_id"]
    df["base_correct"] = df["pred_base"] == df["label_id"]
    return df


def slice_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, g in df.groupby(column, observed=True):
        rows.append({column: value, "n": len(g), "share": f"{len(g) / len(df):.1%}",
                     "AfriBERTa macro-F1": round(compute_metrics(g["label_id"], g["pred_model"])["macro_f1"], 3),
                     "TF-IDF macro-F1": round(compute_metrics(g["label_id"], g["pred_base"])["macro_f1"], 3),
                     "AfriBERTa accuracy": round(g["model_correct"].mean(), 3)})
    return pd.DataFrame(rows)


def suggest(row) -> str:
    """Heuristic hint only (not a judgement): the first matching objective pattern."""
    pair = {LABELS[row.label_id], LABELS[row.pred_model]}
    if row.n_tokens <= 5:
        return "too short / no context"
    if row.code_switched:
        return "code-switching"
    if not row.has_diacritics:
        return "missing diacritics / ambiguity"
    if not row.base_correct and row.pred_base == row.pred_model and row.confidence > 0.9:
        return "both models confidently disagree with gold: label noise OR idiom/proverb? (check)"
    if pair == {"neutral", "negative"}:
        return "neutral-negative boundary"
    return ""


def sample_errors(df: pd.DataFrame, n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Stratified by (gold, predicted) pair, proportional to how often each confusion occurs."""
    errors = df[~df["model_correct"] & ~df["overlap_train"]].copy()
    errors["confusion"] = errors["gold"] + " → " + errors["pred_model_label"]
    quota = (errors["confusion"].value_counts(normalize=True) * n).round().astype(int)
    parts = [errors[errors["confusion"] == c].sample(n=min(q, (errors["confusion"] == c).sum()), random_state=seed)
             for c, q in quota.items() if q > 0]
    out = pd.concat(parts).sort_values(["confusion", "confidence"], ascending=[True, False])
    out["pred_tfidf"] = out["pred_base"].map(dict(enumerate(LABELS)))
    out["suggested_category"] = out.apply(suggest, axis=1)
    out["category"] = ""          # to be filled in by a human reader (one or more of CATEGORIES, ';'-separated)
    out["notes"] = ""
    cols = ["id", "confusion", "raw_text", "text", "gold", "pred_model_label", "pred_tfidf", "confidence",
            "n_tokens", "n_english", "has_diacritics", "suggested_category", "category", "notes"]
    return out[cols].rename(columns={"pred_model_label": "pred_afriberta", "text": "normalised_text"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_features(load_predictions())
    clean = df[~df["overlap_train"]]

    agree = pd.crosstab(clean["model_correct"].map({True: "AfriBERTa right", False: "AfriBERTa wrong"}),
                        clean["base_correct"].map({True: "TF-IDF right", False: "TF-IDF wrong"}), margins=True)
    parts = ["# Error analysis (clean test, original text)", "",
             f"_AfriBERTa-large seed {SEED} (`e4_afriberta_large_final`) vs TF-IDF E1c. Generated by "
             "`python -m src.error_analysis`._", "",
             "## Error overlap", "", agree.to_markdown(), ""]
    for col, title in [("code_switched", "Code-switched (≥ 2 English tokens)"),
                       ("has_diacritics", "Tweet written with diacritics"),
                       ("length_bin", "Length (tokens)"), ("confidence_bin", "AfriBERTa confidence (max probability)")]:
        parts += [f"## {title}", "", slice_table(clean, col).to_markdown(index=False), ""]
    errors = clean[~clean["model_correct"]]
    pair_counts = (errors["gold"] + " → " + errors["pred_model_label"]).value_counts()
    parts += ["## Most frequent confusions (AfriBERTa errors)", "", pair_counts.rename("errors").to_markdown(), ""]
    (OUT / "slices.md").write_text("\n".join(parts))

    sample = sample_errors(df)
    sample.to_csv(OUT / "errors_to_tag.csv", index=False)
    (OUT / "categories.txt").write_text("\n".join(CATEGORIES) + "\n")
    plot_confusion(clean["label_id"], clean["pred_model"], ROOT / "results" / "figures" / "cm_afriberta_test.png",
                   f"AfriBERTa-large (seed {SEED}), clean test (n={len(clean):,})")
    plot_confusion(clean["label_id"], clean["pred_base"], ROOT / "results" / "figures" / "cm_tfidf_test.png",
                   f"TF-IDF word+char + LR, clean test (n={len(clean):,})")
    print((OUT / "slices.md").read_text())
    print(f"wrote {len(sample)} errors to {(OUT / 'errors_to_tag.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
