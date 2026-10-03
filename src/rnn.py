"""BiLSTM / BiGRU sentiment classifier with additive attention over static word embeddings (E2).

  python -m src.rnn --config configs/e2b_bilstm_fasttext.yaml --seed 42 --output_dir results/rnn/e2b_bilstm_fasttext/seed42
  python -m src.evaluate collect results/rnn/e2b_bilstm_fasttext        # add the runs to experiments.csv

Architecture (input -> output):
  tokens (normalised tweet, whitespace-split, <= max_tokens)
  -> Embedding (300-d; fastText / own Word2Vec / random; frozen or fine-tuned)
  -> dropout -> bidirectional LSTM or GRU (hidden h per direction) -> H in R^{T x 2h}
  -> additive attention (Bahdanau et al., 2015): a_t = softmax_t(v^T tanh(W H_t)), c = sum_t a_t H_t
     (ablation: no attention -> concatenated final forward/backward states)
  -> dropout -> Linear(2h, 3) -> softmax over {negative, neutral, positive}
Trained with Adam + cross-entropy; the best epoch is chosen on clean-dev macro-F1 (early stopping).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data import LABELS, load_eval_split, load_split, preprocess_frame
from src.embeddings import embedding_matrix
from src.evaluate import compute_metrics, save_run, score_subsets

DEFAULTS = {
    "lang": "yor",
    "embeddings": "fasttext",        # fasttext | word2vec | random
    "emb_dim": 300,
    "freeze_embeddings": False,
    "rnn": "lstm",                   # lstm | gru
    "hidden": 128,                   # per direction
    "layers": 1,
    "attention": True,
    "dropout": 0.3,
    "max_tokens": 64,                # 95th percentile is 45 tokens
    "min_freq": 1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "epochs": 15,
    "patience": 3,
    "grad_clip": 1.0,
    "class_weights": False,
    "train_style": "semeval",
    "train_diacritics": "original",
    "eval_style": "semeval",
    "eval_diacritics": ["original", "no_tones", "no_diacritics"],
    "select_on_clean_dev": True,
    "notes": "",
}
PAD, UNK = 0, 1


def load_config(path: str) -> dict:
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text())}
    cfg.setdefault("exp_id", Path(path).stem)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- vocabulary / batching

def build_vocab(texts, min_freq: int) -> list[str]:
    counts = Counter(w for t in texts for w in t.split())
    return ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= min_freq)


def encode(texts, stoi: dict[str, int], max_tokens: int) -> list[list[int]]:
    return [[stoi.get(w, UNK) for w in t.split()[:max_tokens]] or [UNK] for t in texts]


def batches(ids: list[list[int]], labels: np.ndarray | None, batch_size: int, shuffle: bool, rng=None):
    order = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
    for start in range(0, len(ids), batch_size):
        idx = order[start:start + batch_size]
        lengths = torch.tensor([len(ids[i]) for i in idx])
        x = torch.full((len(idx), int(lengths.max())), PAD, dtype=torch.long)
        for row, i in enumerate(idx):
            x[row, :len(ids[i])] = torch.tensor(ids[i])
        y = torch.tensor(labels[idx]) if labels is not None else None
        yield x, lengths, y, idx


# --------------------------------------------------------------------------- model

class AdditiveAttention(nn.Module):
    """a_t = softmax_t(v^T tanh(W h_t)); context = sum_t a_t h_t. Padding positions are masked out."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)            # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)                          # (B, T)
        return (alpha.unsqueeze(-1) * H).sum(dim=1), alpha             # (B, 2h), (B, T)


class RNNClassifier(nn.Module):
    def __init__(self, embeddings: np.ndarray, cfg: dict):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=cfg["freeze_embeddings"],
                                                      padding_idx=PAD)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cfg["rnn"]]
        self.rnn = rnn_cls(embeddings.shape[1], cfg["hidden"], num_layers=cfg["layers"], batch_first=True,
                           bidirectional=True, dropout=cfg["dropout"] if cfg["layers"] > 1 else 0.0)
        self.attention = AdditiveAttention(2 * cfg["hidden"]) if cfg["attention"] else None
        self.dropout = nn.Dropout(cfg["dropout"])
        self.out = nn.Linear(2 * cfg["hidden"], len(LABELS))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        H, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))   # (B, T, 2h)
        if self.attention is not None:
            features, alpha = self.attention(H, x != PAD)
        else:
            h_n = hidden[0] if isinstance(hidden, tuple) else hidden                     # LSTM returns (h, c)
            features, alpha = torch.cat([h_n[-2], h_n[-1]], dim=-1), None               # last fwd + bwd states
        return self.out(self.dropout(features)), alpha


@torch.no_grad()
def predict(model, ids, cfg, device) -> tuple[np.ndarray, list]:
    model.eval()
    probs, alphas = np.zeros((len(ids), len(LABELS)), dtype=np.float32), [None] * len(ids)
    for x, lengths, _, idx in batches(ids, None, 256, shuffle=False):
        logits, alpha = model(x.to(device), lengths.to(device))
        probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        if alpha is not None:
            for row, i in enumerate(idx):
                alphas[i] = alpha[row, :lengths[row]].cpu().numpy().round(4).tolist()
    return probs, alphas


# --------------------------------------------------------------------------- training

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--final", action="store_true", help="also evaluate on the test split (Phase 6 only)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    train = preprocess_frame(load_split(cfg["lang"], "train"), cfg["train_diacritics"], cfg["train_style"])
    dev_raw = load_eval_split(cfg["lang"], "dev")
    select = ~dev_raw["overlap_train"] if cfg["select_on_clean_dev"] else slice(None)
    dev_select = preprocess_frame(dev_raw[select], "original", cfg["eval_style"])

    itos = build_vocab(train["text"], cfg["min_freq"])
    stoi = {w: i for i, w in enumerate(itos)}
    matrix, coverage = embedding_matrix(itos, cfg["embeddings"], cfg["emb_dim"], seed=args.seed)
    model = RNNClassifier(matrix, cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train_ids = encode(train["text"], stoi, cfg["max_tokens"])
    train_y = train["label_id"].to_numpy()
    dev_ids = encode(dev_select["text"], stoi, cfg["max_tokens"])
    weights = None
    if cfg["class_weights"]:
        counts = np.bincount(train_y, minlength=len(LABELS))
        weights = torch.tensor(len(train_y) / (len(LABELS) * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    rng = np.random.default_rng(args.seed)
    best_f1, best_state, best_epoch, history, bad_epochs = -1.0, None, 0, [], 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, lengths, y, _ in batches(train_ids, train_y, cfg["batch_size"], shuffle=True, rng=rng):
            optimizer.zero_grad()
            logits, _ = model(x.to(device), lengths.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item() * len(y)
        dev_probs, _ = predict(model, dev_ids, cfg, device)
        dev_f1 = compute_metrics(dev_select["label_id"], dev_probs.argmax(-1))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(total_loss / len(train_ids), 4), "dev_macro_f1": dev_f1})
        print(f"epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  clean-dev macro-F1 {dev_f1:.4f}", flush=True)
        if dev_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = dev_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break
    model.load_state_dict(best_state)

    base = {"exp_id": cfg["exp_id"], "model": f"bi{cfg['rnn']}{'+attn' if cfg['attention'] else ''}:{cfg['embeddings']}"
            f"{'(frozen)' if cfg['freeze_embeddings'] else ''}", "seed": args.seed,
            "train_style": cfg["train_style"], "train_diacritics": cfg["train_diacritics"],
            "class_weights": cfg["class_weights"], "trainable_params": trainable, "total_params": total,
            "git_commit": os.environ.get("IMOLARA_GIT_COMMIT", ""), "notes": cfg["notes"]}
    records, predictions, attention_examples = [], {}, []
    for split in (["dev", "test"] if args.final else ["dev"]):
        raw = dev_raw if split == "dev" else load_eval_split(cfg["lang"], "test")
        for mode in cfg["eval_diacritics"]:
            df = preprocess_frame(raw, mode, cfg["eval_style"])
            ids = encode(df["text"], stoi, cfg["max_tokens"])
            probs, alphas = predict(model, ids, cfg, device)
            df["pred_id"] = probs.argmax(-1)
            recs, preds = score_subsets(df, probs, **base, split=split, eval_diacritics=mode)
            records.extend(recs)
            predictions[f"{split}_{mode}"] = preds
            if split == "dev" and mode == "original" and cfg["attention"]:
                for i in np.flatnonzero(~df["overlap_train"].to_numpy(dtype=bool))[:25]:
                    tokens = df["text"].iloc[i].split()[:cfg["max_tokens"]]
                    attention_examples.append({"id": df["id"].iloc[i], "gold": LABELS[df["label_id"].iloc[i]],
                                               "pred": LABELS[df["pred_id"].iloc[i]],
                                               "tokens": tokens, "attention": alphas[i]})
            for r in recs:
                print(f"[{split}/{mode}/{r['subset']}] macro-F1 {r['macro_f1']:.4f}", flush=True)

    runtime = round(time.time() - start, 1)
    for r in records:
        r["runtime_s"] = runtime
    info = {"config": cfg, "seed": args.seed, "vocab_size": len(itos), "embedding_coverage": coverage,
            "best_epoch": best_epoch, "best_clean_dev_macro_f1": best_f1, "history": history,
            "device": str(device), "runtime_s": runtime}
    output_dir = Path(args.output_dir)
    save_run(output_dir, records, predictions, extra=info)
    if attention_examples:
        (output_dir / "attention_dev.json").write_text(json.dumps(attention_examples, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
