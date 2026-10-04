"""Export the deployed TF-IDF model (E6b: word+char TF-IDF + logistic regression, mixed3 training) with joblib.

  python -m src.export_tfidf        # writes models/tfidf_mixed3.joblib

The exported pipeline tokenises with the built-in `str.split` (identical to src.baselines.whitespace_tokens) so
the pickle does not depend on this repository's modules and loads inside the Hugging Face Space. The script checks
that its dev predictions equal those of the reported E6b run before saving.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.baselines import build_model
from src.data import LABELS, load_eval_split, load_split, preprocess_frame, training_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "tfidf_mixed3.joblib"
RUN = ROOT / "results" / "baselines" / "e6b_wordchar_lr_mixed3"


def main():
    params = json.loads(json.loads((RUN / "metrics.json").read_text())[0]["params"])   # selected on clean dev
    model = build_model({"model": "logreg", "features": "word+char"}, params)
    model.named_steps["features"].transformer_list[0][1].set_params(tokenizer=str.split)   # word vectorizer
    train = training_frame(load_split("yor", "train"), "mixed3")
    model.fit(train["text"], train["label_id"])

    dev = preprocess_frame(load_eval_split("yor", "dev"))
    reported = pd.read_csv(RUN / "predictions_dev_original.csv")
    assert (model.predict(dev["text"]) == reported["pred_id"].to_numpy()).all(), "export differs from reported run"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "labels": LABELS, "params": params,
                 "note": "word+char TF-IDF + LR, mixed3 training (E6b). Input: src.data.preprocess(text)."}, OUT)
    print(f"saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB), params {params}; "
          f"dev predictions identical to the reported E6b run")


if __name__ == "__main__":
    main()
