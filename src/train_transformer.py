"""Fine-tune a pre-trained Transformer encoder for Yoruba tweet sentiment (experiments E3–E8).

Runs on any CUDA machine (we use Kaggle T4s via scripts/kaggle_run.py) or, slowly, on CPU.

  python -m src.train_transformer --config configs/e5a_afroxlmr_base.yaml --seed 42 --output_dir out/e5a/seed42
  python -m src.train_transformer --config ... --final     # Phase 6 only: also score the test split

The model is selected on dev macro-F1. The test split is never loaded unless --final is given.

One process = one GPU: Kaggle's T4 machines have two GPUs, and with both visible the HF Trainer silently
wraps the model in DataParallel, doubling the effective batch (and halving the number of updates). We pin
the process to a single GPU before torch initialises CUDA, so `batch_size` in the config is the real batch.
To use both GPUs, run two seeds in parallel with CUDA_VISIBLE_DEVICES=0 / 1 (the Kaggle notebooks do this).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # must happen before torch touches CUDA

import argparse
import inspect
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, disable_progress_bars
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed)

from src.data import ID2LABEL, LABEL2ID, LABELS, load_eval_split, load_split, preprocess_frame
from src.evaluate import compute_metrics, save_run, score_subsets

DEFAULTS = {
    "lang": "yor",
    "aux_langs": [],                 # E7: extra AfriSenti languages added to training, e.g. [hau, ibo, pcm]
    "train_diacritics": "original",  # original | no_tones | no_diacritics | mixed (original + no_diacritics)
    "eval_diacritics": ["original"],  # dev/test are scored once per listed form (E6)
    "train_style": "semeval",        # semeval (organisers' test format) | raw: text style used for training
    "eval_style": "semeval",         # dev/test are always scored in the test format unless overridden
    "select_on_clean_dev": True,     # pick the best epoch on dev tweets that do not duplicate train
    "max_length": 128,
    "learning_rate": 2e-5,
    "epochs": 5,
    "batch_size": 32,
    "grad_accum": 1,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "class_weights": False,          # E8: inverse-frequency weighted cross-entropy
    "lora": None,                    # E5c: {r, alpha, dropout, target_modules}
    "train_subset": None,            # int: first N training tweets (smoke test / learning curve)
    "save_model": False,
    "notes": "",
}


def load_config(path: str) -> dict:
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text())}
    cfg.setdefault("exp_id", Path(path).stem)
    return cfg


def build_train_frame(cfg: dict, seed: int) -> pd.DataFrame:
    frames = [load_split(lang, "train") for lang in [cfg["lang"], *cfg["aux_langs"]]]
    train = pd.concat(frames, ignore_index=True)
    if cfg["train_subset"]:
        train = train.sample(n=min(cfg["train_subset"], len(train)), random_state=seed)
    if cfg["train_diacritics"] == "mixed":
        train = pd.concat([preprocess_frame(train, "original", cfg["train_style"]),
                           preprocess_frame(train, "no_diacritics", cfg["train_style"])], ignore_index=True)
        return train.drop_duplicates(subset=["text", "label_id"]).reset_index(drop=True)
    return preprocess_frame(train, cfg["train_diacritics"], cfg["train_style"])


class KeepBestTrainable(TrainerCallback):
    """Keep an in-memory copy of the trainable weights from the best evaluation so far.

    With PEFT/LoRA models the Trainer's load_best_model_at_end does not restore the classification head
    (observed: best clean-dev macro-F1 0.707 during training, 0.342 after the 'best' model was reloaded),
    so after training we load this copy instead. For LoRA it holds only the adapters + head (~2.6M values).
    """

    def __init__(self, metric: str = "eval_macro_f1"):
        self.metric, self.best, self.state = metric, float("-inf"), None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics and model is not None and metrics.get(self.metric, float("-inf")) > self.best:
            self.best = metrics[self.metric]
            self.state = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}

    def restore(self, model):
        if self.state is not None:
            missing = set(self.state) - {n for n, _ in model.named_parameters()}
            assert not missing, f"cannot restore {len(missing)} parameters"
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self.state:
                        p.copy_(self.state[n].to(p.device))


class WeightedTrainer(Trainer):
    """Trainer with optional class-weighted cross-entropy (E8)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_model(cfg: dict):
    # dtype=float32: transformers 5 loads weights in their stored dtype by default ("auto"), and
    # Davlan/afro-xlmr-large is stored in float16. Master weights must be fp32; AMP (fp16=True) handles speed.
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, dtype=torch.float32)
    if cfg["lora"]:
        from peft import LoraConfig, TaskType, get_peft_model
        lora = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lora["r"], lora_alpha=lora["alpha"],
            lora_dropout=lora.get("dropout", 0.1), target_modules=lora.get("target_modules", ["query", "value"])))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def predict(trainer: Trainer, ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    logits = trainer.predict(ds).predictions
    logits = logits[0] if isinstance(logits, tuple) else logits
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    return probs.argmax(-1), probs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--final", action="store_true", help="also evaluate on the test split (Phase 6 only)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    disable_progress_bars()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    start = time.time()

    train_df = build_train_frame(cfg, args.seed)
    dev_raw = load_eval_split(cfg["lang"], "dev")
    select_mode = "original" if cfg["train_diacritics"] == "mixed" else cfg["train_diacritics"]
    select_rows = ~dev_raw["overlap_train"] if cfg["select_on_clean_dev"] else slice(None)
    dev_select = preprocess_frame(dev_raw[select_rows], select_mode, cfg["eval_style"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
                                 preserve_index=False)
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    model, trainable, total = build_model(cfg)
    class_weights = None
    if cfg["class_weights"]:
        counts = np.bincount(train_df["label_id"], minlength=len(LABELS))
        class_weights = torch.tensor(len(train_df) / (len(LABELS) * counts), dtype=torch.float32)

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        return compute_metrics(labels, np.argmax(logits, axis=-1))

    # Warmup as an explicit step count: works on transformers 4.x and 5.x (5.x removed warmup_ratio).
    steps_per_epoch = math.ceil(len(train_df) / (cfg["batch_size"] * cfg["grad_accum"]))
    warmup_steps = int(cfg["warmup_ratio"] * steps_per_epoch * cfg["epochs"])

    # Batch tweets of similar length together to cut padding (~2x faster). The option was renamed in
    # transformers 5 (group_by_length -> train_sampling_strategy), so use whichever this version has.
    params = inspect.signature(TrainingArguments.__init__).parameters
    length_grouping = ({"train_sampling_strategy": "group_by_length"} if "train_sampling_strategy" in params
                       else {"group_by_length": True})

    # Checkpoints live in a temp dir so they never end up in the (downloaded) output folder.
    ckpt_dir = Path(tempfile.mkdtemp(prefix="imolara_ckpt_"))
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_steps=warmup_steps,
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        disable_tqdm=True,  # keeps Kaggle logs readable; loss/metrics are still logged
        seed=args.seed,
        **length_grouping,
    )
    keep_best = KeepBestTrainable()
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=to_dataset(train_df),
        eval_dataset=to_dataset(dev_select),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"]), keep_best],
        class_weights=class_weights,
    )
    train_result = trainer.train()
    if cfg["lora"]:          # see KeepBestTrainable: restore adapters + head from the best epoch ourselves
        keep_best.restore(trainer.model)

    # Score the selected model on dev (and test with --final) in every requested diacritic form.
    base = {"exp_id": cfg["exp_id"], "model": cfg["model_name"], "seed": args.seed,
            "train_diacritics": cfg["train_diacritics"], "train_style": cfg["train_style"], "aux_langs": "+".join(cfg["aux_langs"]),
            "lora": bool(cfg["lora"]), "class_weights": cfg["class_weights"],
            "trainable_params": trainable, "total_params": total,
            "git_commit": os.environ.get("IMOLARA_GIT_COMMIT", ""), "notes": cfg["notes"]}
    records, predictions = [], {}
    splits = ["dev", "test"] if args.final else ["dev"]
    for split in splits:
        raw = dev_raw if split == "dev" else load_eval_split(cfg["lang"], "test")
        for mode in cfg["eval_diacritics"]:
            df = preprocess_frame(raw, mode, cfg["eval_style"])
            df["pred_id"], probs = predict(trainer, to_dataset(df))
            recs, preds = score_subsets(df, probs, **base, split=split, eval_diacritics=mode)
            records.extend(recs)
            predictions[f"{split}_{mode}"] = preds
            for r in recs:
                print(f"[{split}/{mode}/{r['subset']}] macro-F1 {r['macro_f1']:.4f}  weighted-F1 {r['weighted_f1']:.4f}")

    # Sanity check: the selected model, re-scored on clean dev, must reproduce the best score seen in training.
    best_seen = keep_best.best if cfg["lora"] else trainer.state.best_metric
    final_select = next(r["macro_f1"] for r in records
                        if r["split"] == "dev" and r["subset"] == ("clean" if cfg["select_on_clean_dev"] else "all")
                        and r["eval_diacritics"] == select_mode)
    selection_consistent = best_seen is None or abs(final_select - best_seen) < 0.005
    if not selection_consistent:
        print(f"WARNING: best model not restored correctly (best during training {best_seen:.4f}, "
              f"final {final_select:.4f})", flush=True)

    runtime = round(time.time() - start, 1)
    for record in records:
        record["runtime_s"] = runtime
    info = {"config": cfg, "seed": args.seed, "n_train": len(train_df), "runtime_s": runtime,
            "train_runtime_s": train_result.metrics.get("train_runtime"),
            "best_checkpoint_dev_macro_f1": trainer.state.best_metric,
            "best_seen_dev_macro_f1": best_seen, "selection_consistent": selection_consistent,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "n_gpu_used": training_args.n_gpu,
            "effective_batch": cfg["batch_size"] * cfg["grad_accum"] * max(1, training_args.n_gpu),
            "log_history": trainer.state.log_history}
    save_run(output_dir, records, predictions, extra=info)

    if cfg["save_model"]:
        model_to_save = trainer.model.merge_and_unload() if cfg["lora"] else trainer.model
        model_to_save.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")
    shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
