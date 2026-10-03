"""Load and preprocess AfriSenti tweets for Yoruba (and auxiliary Nigerian languages).

Data source: the official AfriSenti-SemEval 2023 repository (Muhammad et al., 2023),
pinned to a fixed commit so every run sees identical data. We use the raw TSVs rather
than the Hugging Face copy, which is lower-cased with punctuation removed, because
preprocessing is part of our own documented pipeline.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "afrisenti"

AFRISENTI_COMMIT = "5aec3cfcf87b59f885e98641f72e52182776429d"
AFRISENTI_URL = ("https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/"
                 "{commit}/data/{lang}/{split}.tsv")

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
SPLITS = ("train", "dev", "test")


# --------------------------------------------------------------------------- loading

def load_split(lang: str = "yor", split: str = "train", cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return one AfriSenti split as a DataFrame with columns id, lang, text, label, label_id.

    The TSV is downloaded once and cached under data/raw/afrisenti/<lang>/<split>.tsv.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = Path(cache_dir) / lang / f"{split}.tsv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = AFRISENTI_URL.format(commit=AFRISENTI_COMMIT, lang=lang, split=split)
        pd.read_csv(url, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False, dtype=str) \
          .to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    # keep_default_na=False: tweets such as "NA" or "null" must stay text, not become NaN
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\",
                     keep_default_na=False, dtype=str)
    df = df.rename(columns={"tweet": "text"})
    df.insert(0, "id", [f"{lang}_{split}_{i}" for i in range(len(df))])
    df.insert(1, "lang", lang)
    df["label_id"] = df["label"].map(LABEL2ID)
    if df["label_id"].isna().any():
        raise ValueError(f"unknown labels in {path}: {set(df.label) - set(LABELS)}")
    df["label_id"] = df["label_id"].astype(int)
    return df


def load_afrisenti(lang: str = "yor", splits: tuple[str, ...] = ("train", "dev")) -> dict[str, pd.DataFrame]:
    """Load several splits. The test split is deliberately excluded by default (used only in Phase 6)."""
    return {split: load_split(lang, split) for split in splits}


# --------------------------------------------------------------------------- diacritics

# Yoruba marks tone with combining grave (low), acute (high) and macron (mid, rare);
# the under-dot distinguishes the vowels ẹ/ọ and the consonant ṣ.
TONE_MARKS = {"̀", "́", "̄"}
UNDERDOT = "̣"


def nfc(text: str) -> str:
    """Canonical composition, so the same letter is always stored as the same code points."""
    return unicodedata.normalize("NFC", text)


def strip_tones(text: str) -> str:
    """Remove tone marks but keep under-dots: 'Yorùbá ọ̀rọ̀' -> 'Yoruba ọrọ'."""
    decomposed = unicodedata.normalize("NFD", text)
    return nfc("".join(ch for ch in decomposed if ch not in TONE_MARKS))


def strip_all_diacritics(text: str) -> str:
    """Remove every combining mark (tones and under-dots): 'Yorùbá ọ̀rọ̀ ṣe' -> 'Yoruba oro se'."""
    decomposed = unicodedata.normalize("NFD", text)
    return nfc("".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn"))


def has_tone_marks(text: str) -> bool:
    return any(ch in TONE_MARKS for ch in unicodedata.normalize("NFD", text))


def has_underdots(text: str) -> bool:
    return UNDERDOT in unicodedata.normalize("NFD", text)


DIACRITIC_MODES = {
    "original": lambda text: text,
    "no_tones": strip_tones,
    "no_diacritics": strip_all_diacritics,
}


# --------------------------------------------------------------------------- tweet cleaning

USER_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
RT_RE = re.compile(r"\bRT\b")
HASHTAG_RE = re.compile(r"#\w+")
APOSTROPHE_RE = re.compile(r"['’`]")
REPEAT_RE = re.compile(r"(.)\1{3,}")
SPACE_RE = re.compile(r"\s+")
VARIATION_SELECTORS = {"\ufe0e", "\ufe0f"}


def clean_tweet(text: str) -> str:
    """'raw' style: mask user handles and URLs, cap character repeats at 3, collapse whitespace.
    Case, punctuation, hashtags and emojis are kept."""
    text = USER_RE.sub("@user", text)
    text = URL_RE.sub("http", text)
    text = REPEAT_RE.sub(r"\1\1\1", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_semeval(text: str) -> str:
    """'semeval' style: reproduce the normalisation the AfriSenti organisers applied to the test split.

    The released Yoruba test set is lower-cased and has mentions, URLs, RT markers, hashtags,
    punctuation, digits and emojis removed, while train/dev are raw tweets. Applying this to every
    split removes the train/test format mismatch. Reconstructed from the 79 train/test near-duplicate
    pairs (76/79 reproduced exactly; the 3 misses differ in the underlying tweet) and the observation
    that hashtag words such as #tweetinyoruba (251 train tweets) never occur in test (docs/verification.md).
    """
    text = URL_RE.sub(" ", text)
    text = USER_RE.sub(" ", text)
    text = RT_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = APOSTROPHE_RE.sub("", text.lower())          # d'óró -> dóró
    # keep letters and combining marks (Yoruba tones/under-dots); everything else becomes a space
    text = "".join(ch if unicodedata.category(ch)[0] in "LM" and ch not in VARIATION_SELECTORS else " "
                   for ch in text)
    text = REPEAT_RE.sub(r"\1\1\1", text)
    return SPACE_RE.sub(" ", text).strip()


TEXT_STYLES = {"semeval": normalize_semeval, "raw": clean_tweet}


def preprocess(text: str, diacritics: str = "original", style: str = "semeval") -> str:
    """Full preprocessing used identically in training, evaluation and the web app."""
    if diacritics not in DIACRITIC_MODES:
        raise ValueError(f"diacritics must be one of {list(DIACRITIC_MODES)}, got {diacritics!r}")
    if style not in TEXT_STYLES:
        raise ValueError(f"style must be one of {list(TEXT_STYLES)}, got {style!r}")
    text = TEXT_STYLES[style](nfc(text))
    return DIACRITIC_MODES[diacritics](text)


def preprocess_frame(df: pd.DataFrame, diacritics: str = "original", style: str = "semeval") -> pd.DataFrame:
    """Return a copy of df with the text column preprocessed."""
    out = df.copy()
    out["text"] = [preprocess(t, diacritics, style) for t in out["text"]]
    return out


# --------------------------------------------------------------------------- overlap / leakage

def dedup_key(text: str) -> str:
    """Aggressive normal form for duplicate detection: semeval style without any diacritics."""
    return strip_all_diacritics(normalize_semeval(nfc(text)))


def overlaps_with(df: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows of df whose dedup_key also occurs in reference (e.g. dev/test tweets seen in train)."""
    ref_keys = set(reference["text"].map(dedup_key))
    return df["text"].map(dedup_key).isin(ref_keys)


def load_eval_split(lang: str = "yor", split: str = "dev") -> pd.DataFrame:
    """Load dev/test with an overlap_train column marking tweets that duplicate a training tweet."""
    df = load_split(lang, split)
    df["overlap_train"] = overlaps_with(df, load_split(lang, "train"))
    return df
