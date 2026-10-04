"""Export the deployment AfriBERTa model to ONNX, quantise it to int8, and validate both on clean dev.

  python -m src.export_onnx        # writes models/onnx/{model.onnx, model_int8.onnx} and results/onnx_validation.md

The Streamlit app runs the int8 model with onnxruntime + the `tokenizers` library (no PyTorch), so it fits the
~1 GB memory limit of Streamlit Community Cloud. This script checks, on clean dev in all three diacritic forms:
  * tokenizers.Tokenizer (tokenizer.json) produces exactly the same ids as the transformers tokenizer;
  * ONNX fp32 reproduces PyTorch (max |Δ logit|, prediction agreement);
  * int8 dynamic quantisation (per-channel): prediction agreement with PyTorch and the change in macro-F1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_eval_split, preprocess_frame
from src.evaluate import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deploy_afriberta_mixed3"
OUT_DIR = ROOT / "models" / "onnx"
REPORT = ROOT / "results" / "onnx_validation.md"
MAX_LEN = 128


def fast_encode(tok: Tokenizer, texts: list[str]):
    """Batch-encode with the standalone tokenizers library, exactly as the app does."""
    tok.enable_truncation(MAX_LEN)
    tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
    enc = tok.encode_batch(texts)
    return (np.array([e.ids for e in enc], dtype=np.int64), np.array([e.attention_mask for e in enc], dtype=np.int64))


def export():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    model.config.return_dict = False
    dummy = (torch.ones(2, 16, dtype=torch.long), torch.ones(2, 16, dtype=torch.long))
    torch.onnx.export(model, dummy, OUT_DIR / "model.onnx", input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch"}})
    # per-channel scales: 96.3-96.6% agreement with fp32 and macro-F1 within 0.3 points (per-tensor: 94-95%, -1.1 points)
    quantize_dynamic(OUT_DIR / "model.onnx", OUT_DIR / "model_int8.onnx", weight_type=QuantType.QInt8, per_channel=True)
    for f in ("model.onnx", "model_int8.onnx"):
        print(f"{f}: {(OUT_DIR / f).stat().st_size / 1e6:.0f} MB")


def logits_torch(model, ids, mask, bs=64):
    with torch.no_grad():
        return np.concatenate([model(input_ids=torch.tensor(ids[i:i + bs]), attention_mask=torch.tensor(mask[i:i + bs]))
                               .logits.numpy() for i in range(0, len(ids), bs)])


def logits_onnx(session, ids, mask, bs=64):
    return np.concatenate([session.run(None, {"input_ids": ids[i:i + bs], "attention_mask": mask[i:i + bs]})[0]
                           for i in range(0, len(ids), bs)])


def validate():
    hf_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    fast = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    fp32 = ort.InferenceSession(str(OUT_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(OUT_DIR / "model_int8.onnx"), providers=["CPUExecutionProvider"])
    dev = load_eval_split("yor", "dev")
    rows = []
    for form in ("original", "no_tones", "no_diacritics"):
        df = preprocess_frame(dev[~dev["overlap_train"]], form)
        texts = df["text"].tolist()
        ids, mask = fast_encode(fast, texts)
        ref = hf_tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
        same_ids = ids.shape == ref["input_ids"].shape and (ids == ref["input_ids"]).all()
        lt, lf, lq = logits_torch(model, ids, mask), logits_onnx(fp32, ids, mask), logits_onnx(int8, ids, mask)
        f1 = lambda lg: compute_metrics(df["label_id"], lg.argmax(-1))["macro_f1"]
        rows.append({"dev form (clean)": form, "tokenizer ids identical": bool(same_ids),
                     "ONNX fp32 max |Δlogit|": f"{np.abs(lt - lf).max():.2e}",
                     "ONNX fp32 agreement": f"{(lt.argmax(-1) == lf.argmax(-1)).mean():.2%}",
                     "int8 agreement": f"{(lt.argmax(-1) == lq.argmax(-1)).mean():.2%}",
                     "macro-F1 PyTorch": round(f1(lt), 4), "macro-F1 int8": round(f1(lq), 4),
                     "Δ int8": f"{f1(lq) - f1(lt):+.4f}"})
    table = pd.DataFrame(rows)
    sizes = {f: f"{(OUT_DIR / f).stat().st_size / 1e6:.0f} MB" for f in ("model.onnx", "model_int8.onnx")}
    REPORT.write_text("# ONNX export and int8 quantisation: validation on clean dev\n\n"
                      f"_Generated by `python -m src.export_onnx`. Sizes: {sizes}. The deployed app uses the int8 model._\n\n"
                      + table.to_markdown(index=False) + "\n")
    print(REPORT.read_text())


if __name__ == "__main__":
    export()
    validate()
