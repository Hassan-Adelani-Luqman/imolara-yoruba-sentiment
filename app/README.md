---
title: Ìmọ̀lára Yoruba Sentiment
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: mit
short_description: Yoruba tweet sentiment, robust to missing diacritics
models:
  - Hassanadelani1/imolara-afriberta-mixed3
---

# Ìmọ̀lára: Yoruba tweet sentiment

Compare a fine-tuned **AfriBERTa-large** model and a **TF-IDF + logistic regression** baseline on Yoruba tweets.
Both were trained on AfriSenti-SemEval 2023 Yoruba with every tweet written in three forms (with tone marks, without tone marks,
without any diacritics), so they stay accurate when Yoruba is typed without diacritics.

Code, experiments and report: https://github.com/Hassan-Adelani-Luqman/imolara-yoruba-sentiment
