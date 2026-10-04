"""Regression test: with length-grouped training, predictions must still come back in dataset order.

transformers 5 applies train_sampling_strategy="group_by_length" to evaluation too, which silently shuffled
our final predictions (scores fell to chance). WeightedTrainer forces a sequential eval sampler.
Downloads a tiny random model (~1 MB) from the HF Hub.
"""
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments

from src.train_transformer import WeightedTrainer

TINY = "optimum-intel-internal-testing/tiny-random-xlm-roberta"


def test_predictions_follow_dataset_order(tmp_path):
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForSequenceClassification.from_pretrained(TINY, num_labels=3).eval()
    texts = [" ".join(["ẹ kú àbọ̀"] * n) for n in np.random.default_rng(0).integers(1, 30, 200)]
    ds = Dataset.from_dict({"text": texts, "labels": [0] * len(texts)}).map(
        lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"])
    args = TrainingArguments(output_dir=str(tmp_path), per_device_eval_batch_size=32, report_to="none",
                             train_sampling_strategy="group_by_length")
    trainer = WeightedTrainer(model=model, args=args, processing_class=tok, data_collator=DataCollatorWithPadding(tok))
    predicted = trainer.predict(ds).predictions
    with torch.no_grad():
        manual = np.concatenate([model(**tok(texts[i:i + 32], truncation=True, max_length=128, padding=True,
                                              return_tensors="pt")).logits.numpy() for i in range(0, len(texts), 32)])
    assert np.allclose(predicted, manual, atol=1e-4)
