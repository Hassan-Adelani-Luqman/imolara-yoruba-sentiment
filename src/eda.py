"""Exploratory data analysis for AfriSenti Yoruba (Phase 1).

  python -m src.eda            # writes results/data_stats.json and results/figures/eda_*.png

The notebook notebooks/01_eda.ipynb calls the same functions and shows the outputs.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (LABELS, has_tone_marks, has_underdots, load_split, normalize_semeval, nfc,
                      overlaps_with, strip_all_diacritics)

ROOT = Path(__file__).resolve().parents[1]
STATS_JSON = ROOT / "results" / "data_stats.json"
FIG_DIR = ROOT / "results" / "figures"
SPLITS = ("train", "dev", "test")
AUX_LANGS = ("hau", "ibo", "pcm")
ENGLISH_WORDLIST = Path("/usr/share/dict/american-english")

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
SURFACE_FEATURES = {
    "uppercase": lambda t: any(c.isupper() for c in t),
    "punctuation": lambda t: bool(re.search(r"[.,!?;:'\"()]", t)),
    "mention": lambda t: "@" in t,
    "hashtag": lambda t: "#" in t,
    "url": lambda t: "http" in t,
    "retweet": lambda t: t.startswith("RT "),
    "emoji": lambda t: bool(EMOJI_RE.search(t)),
    "digit": lambda t: bool(re.search(r"\d", t)),
}

# palette (validated with the dataviz skill's validator; see docs): polarity uses a diverging pair
SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#8a8984", "positive": "#2a78d6"}
SPLIT_COLORS = {"train": "#2a78d6", "dev": "#eb6834", "test": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_all() -> dict[str, pd.DataFrame]:
    """Raw Yoruba splits plus a normalised text column (the format every model sees)."""
    data = {s: load_split("yor", s) for s in SPLITS}
    for df in data.values():
        df["norm"] = df["text"].map(lambda t: normalize_semeval(nfc(t)))
        df["n_tokens"] = df["norm"].str.split().str.len()
    return data


# --------------------------------------------------------------------------- statistics

def label_distribution(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    return {label: {"count": int(counts.get(label, 0)), "share": round(counts.get(label, 0) / len(df), 4)}
            for label in LABELS}


def length_stats(df: pd.DataFrame) -> dict:
    n = df["n_tokens"]
    return {"mean": round(float(n.mean()), 1), "median": float(n.median()), "p95": float(n.quantile(0.95)),
            "max": int(n.max()), "by_label_median": {l: float(n[df.label == l].median()) for l in LABELS}}


def surface_features(df: pd.DataFrame) -> dict:
    """Share of raw tweets containing each surface feature (shows the raw-train vs normalised-test gap)."""
    return {name: round(float(df["text"].map(f).mean()), 4) for name, f in SURFACE_FEATURES.items()}


def diacritic_stats(df: pd.DataFrame) -> dict:
    tones, dots = df["text"].map(has_tone_marks), df["text"].map(has_underdots)
    by_label = {l: round(float(tones[df.label == l].mean()), 4) for l in LABELS}
    return {"tone_marks": round(float(tones.mean()), 4), "underdots": round(float(dots.mean()), 4),
            "no_diacritics": round(float((~tones & ~dots).mean()), 4), "tone_marks_by_label": by_label}


def vocabulary_overlap(train: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Out-of-vocabulary rates of `other` w.r.t. train, with and without diacritics."""
    out = {}
    for name, f in {"with_diacritics": lambda t: t, "without_diacritics": strip_all_diacritics}.items():
        train_vocab = Counter(w for t in train["norm"] for w in f(t).split())
        tokens = [w for t in other["norm"] for w in f(t).split()]
        types = set(tokens)
        out[name] = {"train_vocab_size": len(train_vocab),
                     "oov_token_rate": round(sum(w not in train_vocab for w in tokens) / len(tokens), 4),
                     "oov_type_rate": round(sum(w not in train_vocab for w in types) / len(types), 4)}
    return out


def overlap_stats(data: dict[str, pd.DataFrame]) -> dict:
    """Tweets in dev/test that duplicate a training tweet (same text after normalisation, ignoring diacritics)."""
    train = data["train"]
    train_labels = train.assign(key=train["text"].map(_key)).drop_duplicates("key").set_index("key")["label"]
    out = {}
    for split in ("dev", "test"):
        df = data[split]
        mask = overlaps_with(df, train)
        same = (df.loc[mask, "text"].map(_key).map(train_labels) == df.loc[mask, "label"])
        out[split] = {"overlapping": int(mask.sum()), "share": round(float(mask.mean()), 4),
                      "same_label_as_train": int(same.sum()), "different_label": int((~same).sum())}
    out["test_vs_dev"] = int(overlaps_with(data["test"], data["dev"]).sum())
    out["train_internal_duplicates"] = int(train["text"].map(_key).duplicated().sum())
    return out


def _key(text: str) -> str:
    return strip_all_diacritics(normalize_semeval(nfc(text)))


def english_tokens(text: str, english: set[str], yoruba_vocab: set[str]) -> list[str]:
    """Heuristic English detector: undiacritised alphabetic tokens of length >= 4 found in an English
    wordlist and never written with Yoruba diacritics anywhere in the corpus. Approximate by design."""
    return [w for w in text.split()
            if len(w) >= 4 and w.isascii() and w.isalpha() and w in english and w not in yoruba_vocab]


def code_switching(data: dict[str, pd.DataFrame], wordlist: Path = ENGLISH_WORDLIST, min_tokens: int = 2) -> dict:
    if not wordlist.exists():
        return {"available": False, "reason": f"{wordlist} not found"}
    english = {w.strip().lower() for w in wordlist.read_text(errors="ignore").split() if w.strip().isalpha()}
    # words that appear diacritised anywhere are Yoruba even when written plain (e.g. "ise" ~ "iṣẹ́")
    diacritised = {strip_all_diacritics(w) for df in data.values() for t in df["norm"] for w in t.split()
                   if has_tone_marks(w) or has_underdots(w)}
    out = {"available": True, "min_english_tokens": min_tokens}
    for split, df in data.items():
        hits = df["norm"].map(lambda t: english_tokens(t, english, diacritised))
        flagged = hits.str.len() >= min_tokens
        out[split] = {"share_code_switched": round(float(flagged.mean()), 4),
                      "by_label": {l: round(float(flagged[df.label == l].mean()), 4) for l in LABELS}}
        if split == "train":
            out["top_english_tokens_train"] = Counter(w for h in hits for w in h).most_common(25)
            out["examples_train"] = df.loc[flagged, "norm"].head(8).tolist()
    return out


def distinctive_words(df: pd.DataFrame, top: int = 15, prior: float = 0.01) -> dict[str, list]:
    """Most class-distinctive words: weighted log-odds with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008), one class vs the rest, on normalised text."""
    counts = {l: Counter(w for t in df.loc[df.label == l, "norm"] for w in t.split()) for l in LABELS}
    total = sum(counts.values(), Counter())
    a0 = prior * sum(total.values())
    out = {}
    for label in LABELS:
        rest = total - counts[label]
        n1, n2 = sum(counts[label].values()), sum(rest.values())
        scores = {}
        for w, c in total.items():
            if c < 10:
                continue
            aw = prior * c
            y1, y2 = counts[label][w], rest[w]
            delta = math.log((y1 + aw) / (n1 + a0 - y1 - aw)) - math.log((y2 + aw) / (n2 + a0 - y2 - aw))
            var = 1 / (y1 + aw) + 1 / (y2 + aw)
            scores[w] = delta / math.sqrt(var)
        out[label] = [(w, round(s, 2)) for w, s in sorted(scores.items(), key=lambda x: -x[1])[:top]]
    return out


def aux_summary() -> dict:
    out = {}
    for lang in AUX_LANGS:
        df = load_split(lang, "train")
        out[lang] = {"train_size": len(df), "labels": label_distribution(df)}
    return out


def compute_all_stats(data: dict[str, pd.DataFrame] | None = None) -> dict:
    data = data or load_all()
    return {
        "source": "AfriSenti-SemEval 2023, yor, raw TSVs at commit 5aec3cf (see src/data.py)",
        "sizes": {s: len(df) for s, df in data.items()},
        "labels": {s: label_distribution(df) for s, df in data.items()},
        "length_tokens_normalised": {s: length_stats(df) for s, df in data.items()},
        "surface_features_raw": {s: surface_features(df) for s, df in data.items()},
        "diacritics": {s: diacritic_stats(df) for s, df in data.items()},
        "vocabulary": {s: vocabulary_overlap(data["train"], data[s]) for s in ("dev", "test")},
        "overlap": overlap_stats(data),
        "code_switching": code_switching(data),
        "distinctive_words_train": distinctive_words(data["train"]),
        "aux_languages": aux_summary(),
    }


# --------------------------------------------------------------------------- figures

def _style(ax, grid_axis="x"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_label_distribution(stats: dict, path: Path):
    """100% stacked horizontal bars per split, direct-labelled; legend above."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 2.6), dpi=200)
    splits = list(SPLITS)[::-1]
    for i, split in enumerate(splits):
        left = 0.0
        for label in LABELS:
            share = stats["labels"][split][label]["share"]
            ax.barh(i, share, left=left, height=0.62, color=SENTIMENT_COLORS[label],
                    edgecolor="white", linewidth=2)
            ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
            left += share
        ax.text(1.01, i, f"n = {stats['sizes'][split]:,}", va="center", color=INK_MUTED, fontsize=9)
    ax.set_yticks(range(len(splits)), splits, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SENTIMENT_COLORS[l]) for l in LABELS]
    ax.legend(handles, LABELS, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              labelcolor=INK_MUTED, fontsize=9)
    ax.set_title("Label distribution per split", loc="left", color=INK, fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_surface_features(stats: dict, path: Path):
    """Grouped bars: share of tweets with each surface feature, train/dev (raw) vs test (pre-normalised)."""
    import matplotlib.pyplot as plt
    feats = list(SURFACE_FEATURES)
    x = np.arange(len(feats))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=200)
    for j, split in enumerate(SPLITS):
        vals = [stats["surface_features_raw"][split][f] for f in feats]
        bars = ax.bar(x + (j - 1) * width, vals, width - 0.03, color=SPLIT_COLORS[split], label=split)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", va="bottom",
                    fontsize=6.5, color=INK_MUTED)
    ax.set_xticks(x, feats, color=INK, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style(ax, grid_axis="y")
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK_MUTED, fontsize=9)
    ax.set_title("Share of tweets with each surface feature: train/dev are raw, test was pre-normalised",
                 loc="left", color=INK, fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_length_by_label(data: dict[str, pd.DataFrame], path: Path):
    """Token-length distribution of normalised training tweets, one line per label."""
    import matplotlib.pyplot as plt
    df = data["train"]
    bins = np.arange(0, 61, 2)
    fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=200)
    for label in LABELS:
        hist, edges = np.histogram(df.loc[df.label == label, "n_tokens"].clip(upper=60), bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, color=SENTIMENT_COLORS[label], linewidth=2, label=label)
        med = df.loc[df.label == label, "n_tokens"].median()
        ax.axvline(med, color=SENTIMENT_COLORS[label], linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xlabel("tokens per tweet after normalisation (clipped at 60; dashed = median)", color=INK_MUTED, fontsize=9)
    ax.set_yticks([])
    _style(ax, grid_axis="x")
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK_MUTED, fontsize=9)
    ax.set_title("Tweet length by label (train)", loc="left", color=INK, fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(data: dict[str, pd.DataFrame], stats: dict, fig_dir: Path = FIG_DIR) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / "eda_label_distribution.png", fig_dir / "eda_surface_features.png",
             fig_dir / "eda_length_by_label.png"]
    plot_label_distribution(stats, paths[0])
    plot_surface_features(stats, paths[1])
    plot_length_by_label(data, paths[2])
    return paths


def main():
    data = load_all()
    stats = compute_all_stats(data)
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    for p in make_figures(data, stats):
        print("wrote", p.relative_to(ROOT))
    print("wrote", STATS_JSON.relative_to(ROOT))


if __name__ == "__main__":
    main()
