"""Static word embeddings for the RNN models (E2): pretrained fastText and our own Word2Vec.

  python -m src.embeddings fasttext     # download cc.yo.300.vec.gz (85 MB) and report coverage
  python -m src.embeddings word2vec     # train skip-gram Word2Vec on Yoruba Wikipedia + training tweets

Sources:
  fastText Common Crawl + Wikipedia Yoruba vectors (Grave et al., 2018), CC BY-SA 3.0
  Yoruba Wikipedia (wikimedia/wikipedia, 20231101.yo), CC BY-SA 4.0. Unlabelled text only; dev/test are never used.
All text goes through the same normalisation as the classifier input (normalize_semeval + NFC), so
embedding keys and model tokens match.
"""
from __future__ import annotations

import argparse
import gzip
import os
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_split, nfc, normalize_semeval, strip_all_diacritics, strip_tones

ROOT = Path(__file__).resolve().parents[1]
# On Kaggle the files come from the attached dataset (see scripts/kaggle_dataset.py); the kernel sets IMOLARA_EMB_DIR.
EMB_DIR = Path(os.environ.get("IMOLARA_EMB_DIR", ROOT / "data" / "embeddings"))
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yo.300.vec.gz"
WIKI_URL = ("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
            "20231101.yo/train-00000-of-00001.parquet")


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def normalise(text: str) -> str:
    return normalize_semeval(nfc(text))


# --------------------------------------------------------------------------- fastText

def load_fasttext(vocab: set[str] | None = None) -> dict[str, np.ndarray]:
    """Read cc.yo.300.vec.gz, normalising each key like our model input. When several raw keys map to
    the same normalised token (e.g. 'Ọba' and 'ọba'), the first (most frequent) one is kept."""
    plain = EMB_DIR / "cc.yo.300.vec"            # some uploads store the file decompressed
    path = plain if plain.exists() else download(FASTTEXT_URL, EMB_DIR / "cc.yo.300.vec.gz")
    opener = gzip.open if path.suffix == ".gz" else open
    vectors: dict[str, np.ndarray] = {}
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        next(f)  # header: "<n_words> <dim>"
        for line in f:
            word, _, rest = line.rstrip().partition(" ")
            key = normalise(word)
            if not key or " " in key or key in vectors or (vocab is not None and key not in vocab):
                continue
            vectors[key] = np.asarray(rest.split(), dtype=np.float32)
    return vectors


# --------------------------------------------------------------------------- Word2Vec

def wiki_sentences(min_tokens: int = 3) -> list[list[str]]:
    """Yoruba Wikipedia, one normalised paragraph per 'sentence'."""
    path = download(WIKI_URL, EMB_DIR / "yowiki-20231101.parquet")
    sentences = []
    for article in pd.read_parquet(path, columns=["text"])["text"]:
        for paragraph in article.split("\n"):
            tokens = normalise(paragraph).split()
            if len(tokens) >= min_tokens:
                sentences.append(tokens)
    return sentences


def train_word2vec(dim: int = 300, window: int = 5, min_count: int = 2, epochs: int = 10, seed: int = 0):
    """Skip-gram Word2Vec (Mikolov et al., 2013) on Yoruba Wikipedia + AfriSenti training tweets.
    workers=1 makes training deterministic for a given seed."""
    from gensim.models import Word2Vec
    tweets = [normalise(t).split() for t in load_split("yor", "train")["text"]]
    wiki = wiki_sentences()
    model = Word2Vec(sentences=wiki + tweets, vector_size=dim, window=window, min_count=min_count, sg=1,
                     negative=10, epochs=epochs, seed=seed, workers=1)
    path = EMB_DIR / f"w2v_yo_{dim}.kv"
    model.wv.save(str(path))
    print(f"trained on {len(wiki):,} wiki paragraphs + {len(tweets):,} tweets "
          f"({sum(map(len, wiki + tweets)):,} tokens); vocab {len(model.wv):,} -> {path.relative_to(ROOT)}")
    return model.wv


def load_word2vec(dim: int = 300) -> dict[str, np.ndarray]:
    from gensim.models import KeyedVectors
    path = EMB_DIR / f"w2v_yo_{dim}.kv"
    wv = KeyedVectors.load(str(path)) if path.exists() else train_word2vec(dim)
    return {w: wv[w] for w in wv.index_to_key}


# --------------------------------------------------------------------------- embedding matrix

LOOKUP_TIERS = ("exact", "no_tones", "no_diacritics")


def candidate_keys(token: str) -> list[tuple[str, str]]:
    """Lookup order for a token. The fastText Yoruba vocabulary contains no tokens with a tone mark on an
    under-dotted letter (it has 'wọn' and 'wón' but never 'wọ́n'), so we fall back to the tone-stripped
    and then the fully undiacritised spelling."""
    return [("exact", token), ("no_tones", strip_tones(token)), ("no_diacritics", strip_all_diacritics(token))]


def lookup(token: str, vectors: dict[str, np.ndarray]) -> tuple[str | None, np.ndarray | None]:
    for tier, key in candidate_keys(token):
        if key in vectors:
            return tier, vectors[key]
    return None, None


def query_keys(tokens) -> set[str]:
    return {key for t in tokens for _, key in candidate_keys(t)}


def embedding_matrix(itos: list[str], source: str, dim: int = 300, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Matrix aligned with the model vocabulary (row i = token itos[i]). Tokens without a pretrained
    vector get small random vectors; row 0 (<pad>) is zero. Returns (matrix, coverage by lookup tier)."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(itos), dim)).astype(np.float32)
    matrix[0] = 0.0
    if source == "random":
        return matrix, {}
    vectors = load_fasttext(query_keys(itos)) if source == "fasttext" else load_word2vec(dim)
    tiers = Counter()
    for i, token in enumerate(itos[2:], start=2):          # 0 = <pad>, 1 = <unk>
        tier, vec = lookup(token, vectors)
        tiers[tier or "missing"] += 1
        if vec is not None:
            matrix[i] = vec
    n = max(1, len(itos) - 2)
    return matrix, {t: round(tiers[t] / n, 4) for t in (*LOOKUP_TIERS, "missing")}


def coverage_report(source: str):
    """How many training-vocabulary types and tokens have a pretrained vector."""
    counts = Counter(w for t in load_split("yor", "train")["text"] for w in normalise(t).split())
    vectors = load_fasttext(query_keys(counts)) if source == "fasttext" else load_word2vec()
    types, tokens = Counter(), Counter()
    for w, c in counts.items():
        tier = lookup(w, vectors)[0] or "missing"
        types[tier] += 1
        tokens[tier] += c
    print(f"{source}: {len(counts):,} training word types")
    for tier in (*LOOKUP_TIERS, "missing"):
        print(f"  {tier:14s} {types[tier] / len(counts):6.1%} of types  {tokens[tier] / sum(counts.values()):6.1%} of tokens")
    print("most frequent missing:", [w for w, _ in counts.most_common() if lookup(w, vectors)[0] is None][:15])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["fasttext", "word2vec"])
    args = parser.parse_args()
    if args.source == "word2vec":
        train_word2vec()
    coverage_report(args.source)


if __name__ == "__main__":
    main()
