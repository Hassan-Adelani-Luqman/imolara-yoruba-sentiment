"""Report figures built from results/experiments.csv.

  python -m src.figures        # e6_diacritic_matrix.png, model_comparison.png, e9_learning_curve.png in results/figures/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.evaluate import EXPERIMENTS_CSV

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])

TRAIN_FORMS = {"original": "original", "no_tones": "no tones", "no_diacritics": "no diacritics",
               "mixed": "mixed (orig + no diac.)", "mixed3": "mixed3 (all three)"}
EVAL_FORMS = {"original": "original", "no_tones": "no tones", "no_diacritics": "no diacritics"}
E6_FAMILIES = {
    "TF-IDF word+char + LR": {"original": "e1c_wordchar_lr", "no_tones": "e6b_wordchar_lr_notones",
                              "no_diacritics": "e6b_wordchar_lr_nodiacritics", "mixed": "e6b_wordchar_lr_mixed",
                              "mixed3": "e6b_wordchar_lr_mixed3"},
    "AfriBERTa-large": {"original": "e4_afriberta_large", "no_tones": "e6a_afriberta_notones",
                        "no_diacritics": "e6a_afriberta_nodiacritics", "mixed": "e6a_afriberta_mixed",
                        "mixed3": "e6a_afriberta_mixed3"},
}


def load(split: str = "dev", subset: str = "clean") -> pd.DataFrame:
    df = pd.read_csv(EXPERIMENTS_CSV)
    return df[(df["split"] == split) & (df["subset"] == subset)]


def diacritic_matrix(df: pd.DataFrame, family: dict) -> pd.DataFrame:
    """Mean macro-F1 per (training form, evaluation form) for one model family; NaN where not run yet."""
    rows = {}
    for train_form, exp_id in family.items():
        g = df[df["exp_id"] == exp_id].groupby("eval_diacritics")["macro_f1"].mean()
        rows[TRAIN_FORMS[train_form]] = [g.get(e, np.nan) for e in EVAL_FORMS]
    return pd.DataFrame(rows, index=list(EVAL_FORMS.values())).T


def plot_diacritic_matrix(path: Path = FIG_DIR / "e6_diacritic_matrix.png"):
    df = load()
    mats = {name: diacritic_matrix(df, fam) for name, fam in E6_FAMILIES.items()}
    lo = np.nanmin([m.values for m in mats.values()])
    hi = np.nanmax([m.values for m in mats.values()])
    fig, axes = plt.subplots(1, len(mats), figsize=(10, 3.6), dpi=200, sharey=True)
    for ax, (name, m) in zip(axes, mats.items()):
        ax.imshow(m.values, cmap=BLUE, vmin=lo, vmax=hi, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                text = "–" if np.isnan(v) else f"{v:.3f}"
                dark = not np.isnan(v) and (v - lo) / max(hi - lo, 1e-9) > 0.6
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if dark else INK)
        ax.set_xticks(range(m.shape[1]), m.columns, fontsize=8, color=INK_MUTED)
        ax.set_yticks(range(m.shape[0]), m.index, fontsize=8, color=INK_MUTED)
        ax.set_xlabel("evaluated on", fontsize=8, color=INK_MUTED)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("trained on", fontsize=8, color=INK_MUTED)
    fig.suptitle("E6: clean-dev macro-F1 by training and evaluation diacritic form (mean over seeds)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return mats


MAIN_MODELS = {  # one representative per model family, in plot order
    "e0_majority": "Majority class", "e1c_wordchar_lr": "TF-IDF word+char + LR",
    "e2g_bilstm_word2vec_reg": "BiLSTM + attention (Word2Vec)", "e3b_xlmr_base": "XLM-R base",
    "e3a_mbert": "mBERT", "e5a_afroxlmr_base": "AfroXLMR base", "e5c_afroxlmr_large_lora": "AfroXLMR large (LoRA)",
    "e5b_afroxlmr_large": "AfroXLMR large (full FT)", "e4_afriberta_large": "AfriBERTa large",
}


def plot_model_comparison(path: Path = FIG_DIR / "model_comparison.png"):
    """Clean-dev macro-F1 per model (original text), with seed spread; one series, so no legend."""
    df = load()
    df = df[(df["eval_diacritics"] == "original") & df["exp_id"].isin(MAIN_MODELS)]
    stats = df.groupby("exp_id")["macro_f1"].agg(["mean", "std", "count"]).reindex(list(MAIN_MODELS)).dropna(how="all")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    y = np.arange(len(stats))
    ax.barh(y, stats["mean"], height=0.6, color="#2a78d6")
    ax.errorbar(stats["mean"], y, xerr=stats["std"].fillna(0), fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, m, sd in zip(y, stats["mean"], stats["std"].fillna(0)):   # inside the bar, left of the error bar
        ax.text(m - sd - 0.012, yi, f"{m:.3f}", va="center", ha="right", fontsize=8, color="white", fontweight="bold")
    baseline = stats.loc["e1c_wordchar_lr", "mean"]
    ax.axvline(baseline, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(baseline, -0.75, "TF-IDF baseline ", fontsize=7, color=INK_MUTED, va="bottom", ha="right")
    ax.set_yticks(y, [MAIN_MODELS[e] for e in stats.index], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("clean-dev macro-F1 (mean ± std over seeds)", fontsize=8, color=INK_MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=INK_MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Model comparison on clean dev (original text)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


CURVES = {  # series name -> (colour, {fraction: exp_id})
    "TF-IDF word+char + LR": ("#2a78d6", {0.10: "e9_tfidf_010", 0.25: "e9_tfidf_025", 0.50: "e9_tfidf_050",
                                          1.00: "e1c_wordchar_lr"}),
    "AfriBERTa-large": ("#eb6834", {0.10: "e9_afriberta_010", 0.25: "e9_afriberta_025", 0.50: "e9_afriberta_050",
                                    1.00: "e4_afriberta_large"}),
}
N_TRAIN = 8522


def plot_learning_curve(path: Path = FIG_DIR / "e9_learning_curve.png"):
    """E9: clean-dev macro-F1 vs training-set size (log x), mean with min-max band over seeds."""
    df = load()
    df = df[df["eval_diacritics"] == "original"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=200)
    ends = []                                   # (x, y, colour) of each series' last point, labelled below
    for name, (colour, points) in CURVES.items():
        xs, means, los, his = [], [], [], []
        for frac, exp_id in points.items():
            v = df.loc[df["exp_id"] == exp_id, "macro_f1"]
            if v.empty:
                continue
            xs.append(round(frac * N_TRAIN)); means.append(v.mean()); los.append(v.min()); his.append(v.max())
        if not xs:
            continue
        ax.fill_between(xs, los, his, color=colour, alpha=0.15, linewidth=0)
        ax.plot(xs, means, color=colour, linewidth=2, marker="o", markersize=5, label=name)
        ends.append([xs[-1], means[-1], colour, means[-1]])   # x, label y (may be nudged), colour, true value
    ends.sort(key=lambda e: e[1])
    for lower, upper in zip(ends, ends[1:]):    # keep end labels at least 0.012 apart vertically
        upper[1] = max(upper[1], lower[1] + 0.012)
    for x, y, colour, value in ends:
        ax.text(x * 1.08, y, f"{value:.3f}", va="center", fontsize=8, color=INK, fontweight="bold")
        ax.plot([x * 1.02, x * 1.07], [value, y], color=colour, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([852, 2130, 4261, 8522], ["852\n(10%)", "2,130\n(25%)", "4,261\n(50%)", "8,522\n(100%)"],
                  fontsize=8, color=INK_MUTED)
    ax.minorticks_off()
    ax.set_xlim(700, 8522 * 1.6)
    ax.set_xlabel("training tweets (log scale)", fontsize=8, color=INK_MUTED)
    ax.set_ylabel("clean-dev macro-F1", fontsize=8, color=INK_MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0, colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK_MUTED)
    ax.set_title("E9: learning curve (mean over 3 seeds, band = min-max)", loc="left", fontsize=10, color=INK, pad=22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, m in plot_diacritic_matrix().items():
        print(name); print(m.round(3).to_string()); print()
    plot_model_comparison()
    plot_learning_curve()
    print("wrote results/figures/e6_diacritic_matrix.png, model_comparison.png, e9_learning_curve.png")


if __name__ == "__main__":
    main()
