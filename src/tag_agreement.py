"""Agreement between model-assisted error tags and a native speaker's verification.

  python -m src.tag_agreement --make-sheet   # writes results/error_analysis/verification_sheet.csv (50 rows)
  python -m src.tag_agreement                # after the sheet is filled in: writes tag_agreement.md

Sheet = all low-confidence tags + a random sample of the rest (seed 0), up to 50 rows. Columns for the reviewer:
human_category (';'-separated, see CODEBOOK.md), gold_label_ok (Y/N), human_notes.

Agreement is reported per category as Cohen's kappa on presence/absence (multi-label tags), plus the mean Jaccard
overlap of the two tag sets per tweet and the share of tweets whose tag sets match exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "error_analysis"
TAGGED, SHEET, REPORT = DIR / "errors_tagged.csv", DIR / "verification_sheet.csv", DIR / "tag_agreement.md"
CATEGORIES = [c.strip().lower() for c in (DIR / "categories.txt").read_text().splitlines() if c.strip()]


def make_sheet(n: int = 50, seed: int = 0):
    tags = pd.read_csv(TAGGED)
    low = tags[tags["confidence_of_tag"] == "low"]
    rest = tags[tags["confidence_of_tag"] != "low"].sample(n=max(0, n - len(low)), random_state=seed)
    sheet = pd.concat([low, rest]).sample(frac=1, random_state=seed)        # shuffle: no confidence ordering cue
    sheet = sheet.rename(columns={"category": "claude_category", "notes": "claude_notes"})
    sheet["human_category"], sheet["gold_label_ok"], sheet["human_notes"] = "", "", ""
    cols = ["id", "normalised_text", "raw_text", "gold", "pred_afriberta", "human_category", "gold_label_ok",
            "human_notes", "claude_category", "claude_notes", "confidence_of_tag"]
    sheet[cols].to_csv(SHEET, index=False)
    print(f"wrote {len(sheet)} rows ({len(low)} low-confidence + {len(rest)} random) to {SHEET.relative_to(ROOT)}")


def as_set(value) -> set[str]:
    # normalise dashes so 'neutral–negative' (codebook) matches 'neutral-negative' (categories.txt)
    norm = lambda c: c.strip().replace("–", "-").replace("—", "-").lower()
    return {norm(c) for c in str(value).split(";") if c.strip()} if isinstance(value, str) else set()


def report():
    sheet = pd.read_csv(SHEET)
    done = sheet[sheet["human_category"].fillna("").str.strip() != ""]
    if done.empty:
        raise SystemExit(f"No human tags yet: fill in human_category in {SHEET.relative_to(ROOT)} (see CODEBOOK.md)")
    a, b = done["claude_category"].map(as_set), done["human_category"].map(as_set)
    unknown = sorted(set().union(*b) - set(CATEGORIES))
    rows = []
    for cat in CATEGORIES:
        x, y = a.map(lambda s: cat in s).astype(int), b.map(lambda s: cat in s).astype(int)
        if x.sum() + y.sum() == 0:
            continue
        kappa = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else np.nan
        rows.append({"category": cat, "model-assisted": int(x.sum()), "human": int(y.sum()),
                     "both": int((x & y).sum()), "Cohen's κ": round(kappa, 2) if not np.isnan(kappa) else "–"})
    jaccard = [len(s & t) / len(s | t) if s | t else 1.0 for s, t in zip(a, b)]
    gold_ok = done["gold_label_ok"].astype(str).str.upper().str.startswith("Y")
    parts = ["# Tag agreement: model-assisted vs native-speaker verification", "",
             f"_{len(done)} verified tweets (sheet: `verification_sheet.csv`; categories: `CODEBOOK.md`)._", "",
             pd.DataFrame(rows).to_markdown(index=False), "",
             f"- Mean Jaccard overlap of tag sets: **{np.mean(jaccard):.2f}**; identical tag sets: "
             f"**{np.mean([s == t for s, t in zip(a, b)]):.0%}**",
             f"- Reviewer agrees with the gold label for **{gold_ok.mean():.0%}** of these misclassified tweets "
             f"(i.e. {1 - gold_ok.mean():.0%} look mislabelled to a native speaker)"]
    if unknown:
        parts.append(f"- Categories used by the reviewer that are not in the codebook: {', '.join(unknown)}")
    REPORT.write_text("\n".join(parts) + "\n")
    print(REPORT.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--make-sheet", action="store_true")
    a = p.parse_args()
    make_sheet() if a.make_sheet else report()


if __name__ == "__main__":
    main()
