# Ìmọ̀lára — Project Plan

**Ìmọ̀lára** (Yoruba: "feelings, emotions") — sentiment classification for Yoruba tweets.
Repo / Space / Kaggle slug: `imolara-yoruba-sentiment` (no diacritics in URLs). Native-speaker check of the name's spelling and tone marks: [ ] done.

**Report title (working):** *Ìmọ̀lára: Tone-Mark-Robust Sentiment Classification for Yoruba Tweets with Africa-Centric Language Models*

**Course:** Summative Project (NLP) — Option 3: Text Classification for an African Language
**Due:** 18 Oct 2026, 23:59 (hard cutoff 20 Oct)
**Deliverables:** PDF report · GitHub repo · deployed web app · demo video (the brief says 7–10 min in one place and 10–15 min in another, so confirm with the instructor)

---

## 1. Project summary

| Item | Decision |
|---|---|
| Task | 3-class sentiment classification (positive / negative / neutral) |
| Language | Yoruba (`yor`), Nigeria / Benin |
| Primary dataset | AfriSenti, Yoruba subset: 8,523 train / 2,091 dev / 4,516 test tweets, CC-BY-4.0 |
| Baseline | TF-IDF (word + char n-grams) + Logistic Regression / Linear SVM |
| Main models | BiLSTM/BiGRU + attention on fastText embeddings; mBERT, XLM-R, AfriBERTa, AfroXLMR (full fine-tune and LoRA) |
| Headline research questions | RQ1: Do Africa-centric pre-trained models beat general multilingual ones and classical baselines on Yoruba sentiment? RQ2: How robust are models to tweets written with vs. without tone marks? RQ3: Does training data from related Nigerian languages (Hausa, Igbo, Pidgin) help? |
| Primary metric | Macro-F1 (plus weighted-F1, accuracy, per-class P/R/F1) |
| Deployment | Gradio app on Hugging Face Spaces (fallback: Streamlit Community Cloud) |
| Compute | Local CPU (16 GB RAM, no GPU) for data and baselines; Kaggle GPU (T4) for all GPU training, launched from the CLI with the Kaggle API (see [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md)) |

**Why Yoruba sentiment (for the report's motivation):**
- Yoruba has tens of millions of speakers (cite Ethnologue) but few NLP resources (Joshi et al., 2020).
- Social-media Yoruba often leaves out or misuses tone marks, which makes it a good test of robustness.
- AfriSenti results show a large gap between models (XLM-R-base about 63 vs AfroXLMR-large about 74 F1), so the model comparison is meaningful.
- The dataset is large enough (8.5k train, 4.5k test) for stable conclusions.

---

## 2. Phase overview

| Phase | Dates | Goal | Key output |
|---|---|---|---|
| 0 | 3–4 Oct | Setup and fact verification | Repo skeleton, env, verified facts log |
| 1 | 4 Oct | Data acquisition, EDA and preprocessing | `src/data.py`, EDA notebook, data stats table |
| 2 | 4–5 Oct | Baselines (E0–E1) | First rows of `results/experiments.csv` |
| 3 | 5–6 Oct | Embeddings + RNN (E2) | `src/rnn.py`, BiLSTM/BiGRU results |
| 4 | 7–9 Oct | Transformer fine-tuning (E3–E5) | `src/train_transformer.py`, best model chosen on dev |
| 5 | 10 Oct | Ablations (E6–E8) | Tone-mark, cross-lingual and class-weight results |
| 6 | 11 Oct | Final test evaluation and error analysis | Test scores, confidence intervals, error taxonomy |
| 7 | 12 Oct | Deployment | Live app URL, model on the HF Hub |
| 8 | 13–15 Oct | Research report | PDF report |
| 9 | 16 Oct | Repo polish and reproducibility check | Final README, clean-run check |
| 10 | 17 Oct | Demo video | Recorded and uploaded video |
| 11 | 18 Oct | Submission and viva prep | Submitted PDF, Q&A notes |

The schedule keeps 2 buffer days (19–20 Oct) after the deadline, for emergencies only.

---

## Phase 0 — Setup and verification (3–4 Oct)

**Tasks**
1. `git init`; create a GitHub repo (public at submission time); add a `.gitignore` covering `.venv/`, `data/raw/`, `models/`, `*.ckpt` and `wandb/`.
2. Local Python 3.12 venv for CPU work: `datasets`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `gensim`, `torch` (CPU), `transformers`, `peft`, `evaluate`, `gradio`.
3. Kaggle GPU pipeline, following [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md): verify phone number, set up CLI auth, add `kaggle/run_template.py` + `scripts/kaggle_run.py` + `requirements-kaggle.txt`, and pass a **smoke-test run** (`configs/smoke.yaml`) end to end.
4. **Verify these facts and record them in `docs/verification.md` with sources:**
   - [ ] AfriSenti `yor` split sizes and label distribution (from the data itself).
   - [ ] Whether XLM-R (CC-100) and mBERT (Wikipedia) cover Yoruba in pre-training.
   - [ ] AfroXLMR and AfriBERTa language lists (both expected to include Yoruba).
   - [ ] SemEval-2023 Task 12 Yoruba leaderboard: top weighted-F1 and the winning approach.
   - [ ] fastText Yoruba vectors (`cc.yo.300.bin` / `wiki.yo.vec`) can be downloaded.
   - [ ] HF Spaces free-tier rules for Gradio apps (CPU vs ZeroGPU).
   - [ ] Video length (7–10 vs 10–15 min): ask the instructor.

**Done when:** the repo skeleton is committed, the env installs cleanly, and `docs/verification.md` is filled in.

---

## Phase 1 — Data acquisition, EDA and preprocessing (4 Oct)

**Tasks**
1. Load `shmuhammad/AfriSenti-twitter-sentiment` (config `yor`), keeping the **official splits**. The test set is used only once, in Phase 6.
2. **Integrity checks:**
   - Exact and near-duplicates (after normalisation) within each split and across splits.
   - Empty or very short tweets.
   - Label consistency for duplicated texts.
3. **EDA** (`notebooks/01_eda.ipynb`):
   - Label distribution per split.
   - Tweet length (tokens and characters).
   - Most frequent tokens per class.
   - % of tweets containing **tone marks** (combining U+0300 grave, U+0301 acute, U+0304 macron) and **under-dots** (ẹ, ọ, ṣ / U+0323).
   - % with English or Pidgin words (estimated with an English wordlist), emojis, hashtags, URLs and @mentions.
   - Vocabulary overlap between train and test (OOV rate).
4. **Preprocessing** (`src/data.py`), all as configurable functions:
   - Unicode **NFC** normalisation (essential: the same Yoruba character can be stored precomposed or decomposed).
   - Replace @mentions with `@user` and URLs with `http`; collapse repeated characters (more than 3) and extra whitespace.
   - Keep emojis (they carry sentiment) and lowercase only for classical models.
   - `strip_tones(text)`: NFD, remove U+0300/0301/0304, NFC. Under-dots are kept.
   - `strip_all_diacritics(text)`: also removes under-dots (ẹ→e, ọ→o, ṣ→s).
5. **Auxiliary data:** load AfriSenti `hau`, `ibo` and `pcm` train splits for E7 (no test usage).

**Outputs:** `src/data.py`, `notebooks/01_eda.ipynb`, `results/figures/label_dist.png`, `results/data_stats.json`, and a dataset table for report section 4.

**Done when:** the data loads in one call, the stats are reproducible and the preprocessing has unit tests (`tests/test_data.py`: NFC, tone stripping and masking).

---

## Phase 2 — Baselines (4–5 Oct)

Run locally on CPU.

| ID | Model | Details |
|---|---|---|
| E0 | Majority class | Lower bound |
| E1a | TF-IDF word 1–2-grams + Logistic Regression | `class_weight` ∈ {None, balanced}; C ∈ {0.1, 1, 10}; tuned on dev |
| E1b | TF-IDF char 2–5-grams (`char_wb`) + Linear SVM | Char n-grams cope with spelling variation and missing tone marks |
| E1c | Word + char TF-IDF union + LR | Main baseline reported in the report |

- **Evaluation module** (`src/evaluate.py`), reused by every phase:
  - Accuracy, macro-F1, weighted-F1, per-class P/R/F1.
  - Confusion matrix plot.
  - Bootstrap 95% CI (1,000 resamples).
  - Export misclassified examples to CSV.
  - Append one row to `results/experiments.csv`: `id, model, config, seed, split, acc, macro_f1, weighted_f1, notes`.
- Save the top features per class (interpretability material for the report and video).

**Done when:** E0–E1 dev scores are logged and the best baseline is chosen on dev macro-F1.

---

## Phase 3 — Word embeddings + RNN (5–6 Oct)

Run on CPU if feasible, otherwise on Kaggle (`python scripts/kaggle_run.py configs/e2a_bilstm_fasttext.yaml --module src.rnn --wait`).

**Embeddings (compare two)**
- (a) Pretrained fastText Yoruba vectors (300-d; subword-aware, so they handle out-of-vocabulary words).
- (b) Word2Vec skip-gram trained with `gensim` on unlabelled Yoruba text: AfriSenti training tweets plus Yoruba Wikipedia dump (if time allows). Settings: dim 100–300, window 5, min_count 2.

**Model** (`src/rnn.py`, PyTorch):
- Embedding (init from (a) or (b); frozen vs fine-tuned)
- BiLSTM (or BiGRU), 1–2 layers, hidden size 128
- Additive attention pooling
- Dropout 0.3, then a Linear layer to 3 classes

**Training:** Adam (lr 1e-3), batch 64, max length 64 tokens, up to 15 epochs, early stopping on dev macro-F1 (patience 3), gradient clipping 1.0, 3 seeds.

**Experiments**

| ID | Variant |
|---|---|
| E2a | BiLSTM + attention, fastText embeddings, frozen |
| E2b | BiLSTM + attention, fastText embeddings, fine-tuned |
| E2c | BiGRU + attention, best embedding setup |
| E2d | BiLSTM + attention, own Word2Vec embeddings |

- Save attention weights for 10 dev examples (for the video and error analysis).

**Done when:** E2 rows are logged (mean ± std over 3 seeds), with notes on whether the RNNs beat TF-IDF and why.

---

## Phase 4 — Transformer fine-tuning (7–9 Oct)

Run on a Kaggle T4: one Kaggle kernel per experiment, launched with `python scripts/kaggle_run.py configs/<exp>.yaml --wait`.

**Script:** `src/train_transformer.py --config configs/<exp>.yaml` (HF `Trainer` + `peft`).
- Best checkpoint chosen on dev macro-F1.
- Outputs (metrics, dev predictions, logs) are written to `/kaggle/working`, downloaded to `results/kaggle/<exp>/` and appended to `experiments.csv`. Model weights are downloaded only for the final model (`--with-model`).

**Default hyperparameters**
- Max length 128.
- Learning rate 2e-5 for base models, 1e-5 for large ones.
- Batch 32 for base, 16 for large (with gradient accumulation if needed).
- 5 epochs, warmup ratio 0.1, weight decay 0.01, fp16, early stopping (patience 2).
- 3 seeds (42, 43, 44).

| ID | Model (HF id) | Yoruba in pre-training? | Notes |
|---|---|---|---|
| E3a | `google-bert/bert-base-multilingual-cased` | Verify (Wikipedia) | General multilingual |
| E3b | `FacebookAI/xlm-roberta-base` | Verify (CC-100) | General multilingual |
| E4 | `castorini/afriberta_large` | Yes | Trained only on African languages, small data |
| E5a | `Davlan/afro-xlmr-base` | Yes | XLM-R adapted to African languages |
| E5b | `Davlan/afro-xlmr-large` | Yes | Full fine-tune if it fits on a T4; otherwise LoRA only |
| E5c | AfroXLMR-large + **LoRA** | Yes | r ∈ {8, 16}, α = 2r, dropout 0.1, target `query`/`value`, lr 2e-4; report % trainable params, memory and time vs full fine-tune |

**Time budget:** a base model takes about 5–10 min per run and a large one about 20–40 min, so about 25 runs in total, about 12–15 GPU-hours, within one week of Kaggle's free GPU quota (about 30 h/week; check the quota on your Kaggle profile).

**Done when:** E3–E5 are logged with mean ± std, and the **best model is selected on dev only**.

---

## Phase 5 — Ablations (10 Oct)

All ablations use the best model from Phase 4. If GPU time is short, run them on AfroXLMR-base so the comparison stays like-for-like.

| ID | Question | Setup |
|---|---|---|
| E6 | Tone-mark robustness (RQ2) | Train on {original, tones stripped, all diacritics stripped, mixed-augmentation (each training tweet seen in both forms)} × evaluate on dev in each form. Present as a heatmap matrix |
| E6′ | Same question for the baseline | Repeat on TF-IDF + LR to compare: are char n-grams naturally more robust? |
| E7 | Cross-lingual transfer (RQ3) | Train on yor + {hau, ibo, pcm} (all together and one at a time if time allows); evaluate on yor dev |
| E8 | Class imbalance | Weighted cross-entropy (inverse frequency) vs standard |
| E9 (optional) | Learning curve | Train on 10/25/50/100% of yor data; plot macro-F1 vs size for TF-IDF vs the best transformer |

**Done when:** all ablation rows are logged, and each one has a one-sentence answer for the report.

---

## Phase 6 — Final test evaluation and error analysis (11 Oct)

1. **Freeze choices.** Run the selected configs (best baseline, best RNN and best transformer, plus any ablation winners) on the **test set once**.
2. **Quantitative evaluation**
   - Full metrics table with 95% bootstrap CIs.
   - Paired bootstrap or McNemar test between the best model and the baseline, to check the gap is not chance.
   - Compare with the published AfriSenti and SemEval Yoruba scores, using the matching metric (weighted-F1).
3. **Error analysis**
   - Sample 100 test errors from the best model.
   - Hand-tag each with one or more categories:
     - Missing or wrong tone marks / lexical ambiguity
     - Code-switching (Yoruba–English–Pidgin)
     - Sarcasm, irony or proverbs
     - Slang or new words (out-of-vocabulary)
     - Sentiment carried by emoji or hashtags
     - Neutral vs negative boundary
     - Likely label noise (annotator disagreement)
   - Report counts per category, the confusion matrix and **6–8 annotated examples** (with an English gloss).
   - **Model comparison:** errors the baseline makes but the transformer fixes, and errors both make.
   - **Interpretability:** LIME (or attention) for 3–4 cases.
4. **Successful examples:** pick 5 correct predictions that show what each model gets right.

**Outputs:** `notebooks/03_error_analysis.ipynb`, `results/errors_tagged.csv`, `results/figures/*`.

---

## Phase 7 — Deployment (12 Oct)

1. Push the best model, tokenizer and model card to the HF Hub (`<user>/yoruba-sentiment-afroxlmr`). The model card covers data, metrics, limitations and the license.
2. **`app/app.py` (Gradio)**
   - Text box for Yoruba input. The same preprocessing as training (NFC, masking) is applied automatically.
   - Output: predicted label with class probabilities as a bar chart.
   - Example buttons: positive, negative and neutral tweets, the same tweet with and without tone marks, a code-switched tweet, and a known failure case.
   - "About" panel: model, data, test macro-F1 and limitations.
3. **Hosting:** HF Space (CPU, or ZeroGPU if CPU Gradio requires PRO). If the large model is too slow on CPU, use dynamic int8 quantisation or fall back to the AfroXLMR-base checkpoint, and state this in the report.
4. Check latency (under 2 s per query is the target) and the cold start, and confirm the app works from an incognito window and on a phone.

**Done when:** a public URL works and the app's predictions match the offline predictions on 10 test tweets.

---

## Phase 8 — Research report (13–15 Oct)

**Format:** ACL-style LaTeX (Overleaf) or Markdown→PDF via pandoc, 8–12 pages plus references, using **one citation style throughout** (ACL / author–year).
**Project links box** on the first page: GitHub, demo video, live system.

| § | Section | Content and sources |
|---|---|---|
| — | Abstract | Problem, language, data, approach, key numbers, conclusion (about 200 words) |
| 1 | Introduction | Yoruba and NLP under-resourcing, use cases, RQ1–RQ3, contributions |
| 2 | Related work | African sentiment analysis (NaijaSenti, AfriSenti, SemEval-2023 Task 12 and its top systems); African pre-trained LMs (AfriBERTa, AfroXLMR, SERENGETI); Yoruba diacritics (Orife 2018; Adelani et al. 2021); multilingual LMs (mBERT, XLM-R); LoRA. **Gap:** no systematic tone-mark robustness study plus a deployed tool |
| 3 | Dataset | Source, collection and annotation (from the AfriSenti paper), sizes, label distribution, EDA findings, preprocessing, quality issues, license |
| 4 | Methodology | Baseline, embeddings + RNN architecture, transformer fine-tuning, LoRA maths (W + BA), hyperparameters, model-selection protocol |
| 5 | Experiments & results | Main table (E0–E5), ablation tables and figures (E6–E9), interpretation of each result |
| 6 | Error analysis & discussion | Error taxonomy, examples, links to the literature |
| 7 | Deployment | Architecture diagram (user → Gradio → tokenizer → model → softmax), hosting, link |
| 8 | Limitations & future work | Twitter domain only, label noise, 3 classes, no dialect coverage, compute limits, possible diacritic restoration step |
| 9 | Conclusion | Answers to RQ1–RQ3, lessons learned |
| 10 | References | Papers, datasets, models, libraries (HF Transformers, PEFT, scikit-learn, gensim, Gradio) |
| — | Statement of own work | What was reused (data, pre-trained models, libraries) vs implemented by me; AI-assistance disclosure |

**Done when:** every factual claim is cited and every number matches `experiments.csv`.

---

## Phase 9 — Repository polish and reproducibility (16 Oct)

**Final layout**
```
├── README.md            problem, data, method, results, links, how to run
├── PLAN.md              this file
├── requirements.txt     pinned versions
├── configs/             one YAML per experiment
├── src/
│   ├── data.py          loading, normalisation, tone stripping
│   ├── baselines.py     E0–E1
│   ├── embeddings.py    fastText / Word2Vec
│   ├── rnn.py           BiLSTM/BiGRU + attention (E2)
│   ├── train_transformer.py   E3–E8 (full FT + LoRA)
│   ├── evaluate.py      metrics, CIs, confusion matrix, error export
│   └── predict.py       CLI inference
├── notebooks/           01_eda, 03_error_analysis
├── kaggle/              run_template.py (Kaggle entry point), _build/ (generated, git-ignored)
├── scripts/             kaggle_run.py (bundle → push → wait → download)
├── requirements-kaggle.txt   extra packages installed on Kaggle (no torch/transformers pins)
├── results/             experiments.csv, figures/, errors_tagged.csv
├── app/                 app.py, requirements.txt (Space)
├── tests/               test_data.py
├── docs/                verification.md, KAGGLE_GUIDE.md
└── report/              report source + final PDF
```
- **README:** overview, links (app, video, model, report), quick start, how to reproduce each phase (exact commands), results table, credits and licenses.
- **Reproducibility check:** fresh clone, then run the baseline locally and one transformer config on Kaggle (and once on any CUDA machine with `python -m src.train_transformer`), following only the README.

---

## Phase 10 — Demo video (17 Oct)

Plan for about 10 min; adjust once the instructor confirms the length. Each segment maps to a rubric criterion.

| Time | Segment | Rubric criterion |
|---|---|---|
| 0:00–1:00 | Problem, Yoruba, motivation, users | Problem definition |
| 1:00–2:15 | Dataset, EDA, preprocessing, tone-mark issue | Problem definition |
| 2:15–4:15 | Models: baseline → BiLSTM + attention → transformers + LoRA; architecture, inputs/outputs, hyperparameters, *why*; show key code | Model development |
| 4:15–6:00 | Experiments table: why each one was run, what changed, what was learned | Baseline & experimentation |
| 6:00–7:45 | Metrics (why macro-F1), results with CIs, error examples and categories | Evaluation & error analysis |
| 7:45–9:30 | Live app: how it connects to the model, good / tone-stripped / code-switched / failure inputs | Deployment |
| 9:30–10:00 | Limitations, lessons, future work | Technical understanding |

**Format:** screen recording with face cam (OBS), uploaded unlisted to YouTube or Drive. Check the link opens without signing in.

---

## Phase 11 — Submission and viva prep (18 Oct)

- **Final PDF check:** links work, links are on the first page, no placeholders remain.
- Make the GitHub repo public; tag the release `v1.0-submission`.
- Upload the PDF to the course portal.
- **Viva notes** (1 page): why macro-F1; how attention and self-attention work; what LoRA changes; why a given model won or lost; how tone-mark stripping is implemented; what the test-set protocol was.

---

## 3. Rubric coverage

| Criterion (pts) | Where it's earned |
|---|---|
| Problem, language & dataset (8) | Phase 1 + report §1, §3; gap evidence from SemEval/AfriSenti plus robustness |
| Model development (10) | Phases 3–4 + video 2:15–4:15 |
| Baseline & experiments (8) | Phases 2, 5 + `experiments.csv` + video 4:15–6:00 |
| Evaluation & error analysis (7) | Phase 6 + video 6:00–7:45 |
| Report (10) | Phase 8 |
| Deployment (7) | Phase 7 + video 7:45–9:30 |
| Code & GitHub (5) | Phase 9 |
| Individual understanding (5) | Writing the code myself, viva notes, explaining *why* throughout the video |

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle GPU quota runs out | Smoke-test first; base models before large; LoRA for large; run seeds in one session; track hours after each run. The code is platform-agnostic, so Colab is an emergency fallback |
| AfroXLMR-large runs out of memory on a T4 | fp16 + gradient accumulation + max length 128; otherwise LoRA only |
| HF Space free tier blocked or slow | ZeroGPU Space or Streamlit Cloud; quantised base model |
| Label noise limits scores | Measure and discuss it in error analysis; it is a finding, not a failure |
| Complex model doesn't beat baseline | Explain it (data size, domain, noise); the rubric explicitly allows this |
| Schedule slips | E9 and the extra E7 variants are optional; core path is E0, E1, E2a, E3b, E5a, E5c, E6 |
| Accidental test-set tuning | Test is used only in Phase 6, enforced by the evaluation script flag `--split test` |
