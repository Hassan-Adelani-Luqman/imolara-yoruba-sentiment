---
language: yo
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: castorini/afriberta_large
datasets:
  - shmuhammad/AfriSenti-twitter-sentiment
tags:
  - sentiment-analysis
  - yoruba
  - african-languages
  - afrisenti
metrics:
  - f1
---

# Ìmọ̀lára: AfriBERTa-large for Yoruba tweet sentiment (diacritic-robust)

Fine-tuned [`castorini/afriberta_large`](https://huggingface.co/castorini/afriberta_large) for 3-class sentiment (negative / neutral / positive)
on the Yoruba part of **AfriSenti-SemEval 2023** (Muhammad et al., 2023). The model is trained on every tweet in three written forms
(original, tone marks removed, all diacritics removed: "mixed3"), which makes it robust to Yoruba typed without diacritics.

**Files:** `model.safetensors` + tokenizer (PyTorch / transformers); `onnx/model_int8.onnx`, a per-channel int8 dynamic-quantised ONNX export (127 MB, used by the web app with onnxruntime; on clean dev it agrees with the PyTorch model on 96.3–96.6% of tweets, and macro-F1 is within ±0.3 points in all three diacritic forms).

This repository also contains `tfidf_mixed3.joblib`: the word + character TF-IDF + logistic regression model trained the same way
(scikit-learn 1.9.1). Both models power the demo web app: https://imolara-yoruba-sentiment-oafn6nuzsde8db6a9fdahs.streamlit.app/

## Usage

Inputs must be normalised exactly as in training (lower-case; mentions, URLs, hashtags, punctuation, digits and emojis removed; NFC).
See `src/data.py::preprocess` in the [GitHub repository](https://github.com/Hassan-Adelani-Luqman/imolara-yoruba-sentiment).

```python
from transformers import pipeline
clf = pipeline("text-classification", model="Hassanadelani1/imolara-afriberta-mixed3", top_k=None)
clf("ẹ kú àbọ̀ o inú mi dùn púpọ̀ láti rí yín")
```

## Results (AfriSenti Yoruba test set)

| Model | clean-test macro-F1 | test weighted-F1 (×100) | Δ without tone marks | Δ without any diacritics |
|---|---|---|---|---|
| AfriBERTa-large mixed3 (3 seeds; **this model is seed 42**) | 0.741 ± 0.007 | 77.5 | −0.002 | −0.012 |
| TF-IDF mixed3 (`tfidf_mixed3.joblib`) | 0.745 | 77.8 | +0.001 | −0.006 |
| AfriBERTa-large, no augmentation | 0.732 ± 0.005 | 76.7 | −0.038 | −0.070 |

"Clean" test = the 4,129 test tweets that do not duplicate a training tweet. The weighted-F1 is the official SemEval-2023 metric
(published Yoruba results: AfriSenti paper AfriBERTa-large 72.9, AfroXLMR-large 74.1; best SemEval-2023 system 80.2).

## Training

Learning rate 3e-5, batch 32, up to 10 epochs with early stopping (patience 3) on clean-dev macro-F1, max length 128, fp16, on one
Kaggle T4. Training data: the 8,522 AfriSenti Yoruba training tweets × 3 diacritic forms (de-duplicated). The model was selected
on the development set only; the test set was used once.

## Limitations and bias

- Twitter domain only; about 10% of AfriSenti Yoruba labels look unreliable (confident-learning estimate), and annotator agreement is κ = 0.65.
- Common errors: proverbs and idioms, negation (*kò*, *kì í*), religious or greeting words in non-positive tweets, code-switching.
- Probabilities are over-confident (94% of test predictions exceed 0.9; about 77% of those are correct).
- Not suitable for decisions about individuals.

## Citation and credits

Data: Muhammad et al. (2023), *AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages* (EMNLP), CC BY 4.0.
Base model: Ogueji, Zhu & Lin (2021), *Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for
Low-resourced Languages* (MRL), MIT licence.
