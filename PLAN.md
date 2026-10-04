# Ìmọ̀lára — Project Plan

**Ìmọ̀lára** (Yoruba: "feelings, emotions") — sentiment classification for Yoruba tweets.
Repo / Space / Kaggle slug: `imolara-yoruba-sentiment` (no diacritics in URLs). Native-speaker check of the name's spelling and tone marks: [ ] done.

**Report title (working):** *Ìmọ̀lára: Tone-Mark-Robust Sentiment Classification for Yoruba Tweets with Africa-Centric Language Models*

**Course:** Summative Project (NLP) — Option 3: Text Classification for an African Language
**Due:** 18 Oct 2026, 23:59 (hard cutoff 20 Oct)
**Deliverables:** PDF report · GitHub repo · deployed web app · demo video, **7–10 min** (confirmed with the instructor)

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
| Deployment | Gradio app on a Hugging Face **ZeroGPU** Space (free; the account is >30 days old). Fallback: Streamlit Community Cloud |
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
   - [x] HF Spaces free-tier rules: CPU Gradio needs PRO; free accounts get up to 2 ZeroGPU Spaces → use ZeroGPU.
   - [x] Video length: 7–10 min.

**Done when:** the repo skeleton is committed, the env installs cleanly, and `docs/verification.md` is filled in.

---

## Phase 1 — Data acquisition, EDA and preprocessing (4 Oct) ✅ done

**What was done**
1. Loaded the **raw AfriSenti TSVs pinned to commit `5aec3cf`** (`src/data.py`). The HF copy relies on a loading script that current `datasets` can't run. Official splits kept: **8,522 / 2,090 / 4,515**.
2. **Integrity:** no empty tweets, 15 near-duplicates inside train, no conflicting duplicate labels. **13% of dev and 8.5% of test duplicate a training tweet** (about 96% with the same label).
3. **EDA:** `src/eda.py` (CLI, writes `results/data_stats.json` and `results/figures/eda_*.png`) and `notebooks/01_eda.ipynb` (executed, with narrative).
4. **Preprocessing** (`src/data.py`, 8 unit tests):
   - NFC.
   - **`normalize_semeval`**: the organisers' test normalisation, reconstructed and validated (76/79 pairs exact). It is applied to all splits and is the default style.
   - `style="raw"` (mask mentions and URLs, keep emojis) is kept for an ablation.
   - `strip_tones` and `strip_all_diacritics`.
   - `dedup_key`, `overlaps_with` and `load_eval_split` (adds an `overlap_train` flag).
5. Auxiliary hau/ibo/pcm train splits load; sizes and labels are in the stats file.

**Key findings that change the plan**
| Finding | Consequence |
|---|---|
| **Test was pre-normalised by the organisers** (0% uppercase, punctuation, hashtags, emojis); train/dev are raw | All splits get `normalize_semeval`; **emojis are no longer kept** (they never appear in test). New ablation **E1-style**: train raw vs normalised, scored on normalised dev |
| Dev/test duplicate train (13% / 8.5%) | Every result is reported on **all** and **clean** subsets (`evaluate.score_subsets`); model selection uses **clean dev** (1,817 tweets) |
| 77% of tweets carry tone marks, but **presence correlates with label** (neutral 85%, negative 71%) | E6 also tests whether models rely on "has diacritics" as a shortcut |
| Test OOV much higher than dev (41% vs 29% of word types) | Favours subword/char models; check the dev→test drop in Phase 6 |
| About 14% English code-switching (heuristic) | Error-analysis category; mBERT/XLM-R English knowledge may help |
| Pidgin is 1.4% neutral | E7 runs with and without pcm |

---

## Phase 2 — Baselines (4–5 Oct) ✅ done

**Results** (`results/summary.md`; selection on clean dev, n = 1,817)

| ID | Model | macro-F1 clean dev | macro-F1 all dev | F1 neg / neu / pos |
|---|---|---|---|---|
| E0 | Majority class | 0.198 | 0.198 | 0.00 / 0.00 / 0.59 |
| E1a | Word 1–2-gram TF-IDF + LR (C=30, balanced) | 0.706 | 0.738 | 0.63 / 0.72 / 0.77 |
| E1b | Char 2–5-gram TF-IDF + LinearSVC (C=0.1, balanced) | 0.714 | 0.733 | 0.62 / 0.74 / 0.78 |
| **E1c** | **Word + char TF-IDF + LR (C=3)** (main baseline) | **0.722** [0.701, 0.744] | 0.753 | 0.61 / 0.76 / 0.80 |
| E1d | E1c trained on **raw** text | 0.684 | 0.706 | 0.60 / 0.68 / 0.78 |

**What we learned**
1. **Word + character features are complementary:** E1c beats words alone (+1.7) and characters alone (+0.8).
2. **Matching the test format matters:** training on raw tweets costs **3.9 macro-F1 points** (E1d vs E1c). This confirms the Phase 1 decision.
3. **Duplicates inflate dev by about 3 points** (all dev 0.753 vs clean 0.722), so selecting on clean dev was the right call.
4. **Tone-mark sensitivity is large:** with tone marks stripped at test time, every baseline loses **8–13 points**. Character n-grams lose least on tone stripping (−8.3), but no less when under-dots are removed too. The models trained on diacritised text treat diacritics as part of word identity, which motivates the E6 training-side variants.
5. **Negative is the hard class** (recall 56%; confused with both neutral and positive), as expected for the 22% minority.
6. **Tokenisation pitfall:** scikit-learn's default token pattern drops one-letter words (*o*, *ẹ*) and splits words at combining tone marks (*ọ̀rẹ́* → *rẹ*). We tokenise on whitespace.
7. The top features are linguistically sensible (negative: *ò* 'not', *pa* 'kill', *ikú* 'death', *olè* 'thief'; positive: *ire*, *ẹ kú*, *rere*, *ìfẹ́*; neutral: *kí ni*, *ǹjẹ́*, *Ifá*), which is useful material for the video.

**Bar for the neural models:** clean-dev macro-F1 **0.722**.

<details><summary>Original Phase 2 plan</summary>


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

- **E1-style ablation:** the best E1 configuration trained on `raw` vs `semeval` text, both scored on normalised dev (measures the cost of the train/test format mismatch).
- All baselines are tuned on **clean dev** and reported on all + clean dev.

**Done when:** E0–E1 dev scores are logged and the best baseline is chosen on clean-dev macro-F1.

</details>

---

## Phase 3 — Word embeddings + RNN (5–6 Oct) ✅ done

**Results** (clean-dev macro-F1, mean ± std over 3 seeds; Kaggle T4, about 20–40 s per seed)

| ID | Variant | macro-F1 | Δ no tone marks |
|---|---|---|---|
| E2a | BiLSTM+attn, fastText **frozen** | 0.590 ± 0.004 | −0.067 |
| E2h | BiLSTM+attn, own Word2Vec **frozen** | 0.657 ± 0.010 | −0.090 |
| E2e | BiLSTM+attn, **random** init | 0.676 ± 0.010 | −0.110 |
| E2f | BiLSTM, fastText, **no attention** | 0.679 ± 0.008 | −0.103 |
| E2c | **BiGRU**+attn, fastText | 0.684 ± 0.008 | −0.094 |
| E2b | BiLSTM+attn, fastText fine-tuned | 0.687 ± 0.013 | −0.115 |
| E2d | BiLSTM+attn, own **Word2Vec** fine-tuned | 0.694 ± 0.007 | −0.074 |
| **E2g** | E2d + stronger regularisation (dropout 0.5, h=64, wd 1e-5) | **0.699 ± 0.012** | −0.080 |
| E1c | TF-IDF word+char + LR (baseline) | **0.722** | −0.103 |

**What we learned**
1. **No RNN beats the TF-IDF baseline** (best 0.699 vs 0.722). Likely reasons: small data (8.5k tweets); a high rate of unseen words that word-level RNNs map to `<unk>` while character n-grams still match them; short texts where bag-of-n-grams already captures most of the signal; and fast overfitting (every run peaks at epoch 2–3). Stronger regularisation (E2g) helps only within noise.
2. **Pretrained embeddings help modestly** (+1.1 to +1.8 over random), and **orthography-matched embeddings help most**. Our Word2Vec, trained on text normalised like ours, beats fastText and is the most robust to missing tone marks.
3. **Frozen fastText fails (0.590). About 6.7 points of that come from orthography mismatch** (frozen Word2Vec scores 0.657), and freezing itself costs about 3.7 points relative to fine-tuning. The fastText vocabulary has no tone marks on under-dotted vowels (*wọ́n*, *jẹ́*); tiered lookup (exact → no tones → no diacritics) raises its token coverage from 80.8% to 92.2%.
4. **Attention (+0.8) and LSTM vs GRU make differences within seed noise.** The attention weights are still interpretable: they land on *rere*, *ire*, *ẹ kú*, *ìrànlọ́wọ́* (`attention_dev.json`). A failure example: a news tweet about someone's *release* from prison gets attention on names and prison words, and is predicted negative.

<details><summary>Original Phase 3 plan</summary>


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

</details>

---

## Phase 4 — Transformer fine-tuning (7–9 Oct) ✅ done (4 Oct)

**Results** (clean-dev macro-F1, mean ± std over 3 seeds; executed Kaggle notebooks in `notebooks/kaggle/`)

| ID | Model | Yoruba in pre-training | macro-F1 | Δ no tones | Δ no diacritics |
|---|---|---|---|---|---|
| E3b | XLM-R-base (278M) | **no** | 0.650 ± 0.006 | −0.067 | −0.101 |
| E3a | mBERT (178M) | yes (Wikipedia) | 0.674 ± 0.006 | −0.078 | −0.092 |
| E5a | AfroXLMR-base (278M) | yes | 0.687 ± 0.004 | −0.013 | −0.062 |
| E5c | AfroXLMR-large + **LoRA** r=16 (2.6M trainable, 0.47%) | yes | 0.708 ± 0.004 | **−0.006** | −0.041 |
| E5b | AfroXLMR-large full FT (560M) | yes | 0.714 ± **0.029** | −0.014 | −0.041 |
| **E4** | **AfriBERTa-large (126M)** | yes | **0.731 ± 0.006** | −0.062 | −0.073 |
| E1c | TF-IDF word+char + LR (baseline) | – | 0.722 | −0.103 | −0.102 |

**What we learned**
1. **RQ1 (pre-training coverage):** the clean comparison is XLM-R-base vs AfroXLMR-base: identical architecture and tokenizer, with AfroXLMR additionally adapted on African languages incl. Yoruba. That adaptation is worth **+3.7 points** (0.650 → 0.687). mBERT (saw Yoruba Wikipedia) also beats XLM-R (no Yoruba).
2. **Smaller, Africa-only AfriBERTa (126M) is the best model**, ahead of AfroXLMR-base/large. This matches the AfriSenti paper's ranking for Yoruba. Africa-specific tokenisation and pre-training data matter more than size.
3. **Transformers vs TF-IDF: no significant difference.** Paired bootstrap on clean dev, AfriBERTa vs E1c, gives p = 0.10–0.36 per seed. With 8.5k noisy tweets (κ = 0.65), a strong n-gram model is hard to beat.
4. **AfroXLMR models are far more robust to missing tone marks** (−0.6 to −1.4 points) than TF-IDF, the RNNs, mBERT, XLM-R or AfriBERTa (−6 to −13). Removing under-dots as well still costs them 4–6 points.
5. **LoRA vs full fine-tuning (AfroXLMR-large):** similar mean (0.708 vs 0.714), but LoRA is about **7× more stable** (std 0.004 vs 0.029; one full-FT seed nearly collapsed, 0.198 at epoch 2), trains 0.47% of the parameters and is about twice as fast per epoch.
6. **Engineering lessons (see `docs/verification.md`):** Kaggle T4 machines have 2 GPUs (silent DataParallel); transformers 5 loads checkpoints in their stored dtype (afro-xlmr-large is fp16); length-grouped sampling also reorders evaluation, which scrambled predictions; LoRA at lr 3e-4 collapsed. All were fixed, regression-tested, and guarded by the per-run `selection_consistent` check.

**Phase 5 base model:** AfriBERTa-large (best and fast, about 4 min/seed, but the least diacritic-robust transformer, so it has the most to gain from E6).

<details><summary>Original Phase 4 plan</summary>


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

</details>

---

## Phase 5 — Ablations (10 Oct) ✅ done (4 Oct)

**E6: tone-mark robustness** (clean-dev macro-F1 by training form × evaluation form; figure `results/figures/e6_diacritic_matrix.png`)

| Trained on → evaluated on | TF-IDF: original | no tones | no diacritics | AfriBERTa: original | no tones | no diacritics |
|---|---|---|---|---|---|---|
| original (E1c / E4) | 0.722 | 0.620 | 0.620 | 0.731 | 0.669 | 0.658 |
| no tones | 0.643 | 0.716 | 0.678 | 0.667 | 0.721 | 0.691 |
| no diacritics | 0.597 | 0.680 | 0.717 | 0.626 | 0.681 | 0.713 |
| mixed (original + no diacritics) | **0.733** | 0.686 | 0.718 | **0.732** | 0.696 | 0.708 |
| **mixed3** (all three forms) | 0.732 | **0.720** | 0.717 | 0.723 | 0.716 | 0.713 |

**E7: cross-lingual** (AfriBERTa, Yoruba clean dev): + hau/ibo/pcm **0.716 ± 0.003**; + hau/ibo **0.715 ± 0.009**; Yoruba only 0.731 ± 0.006.
**E8: class weights** (AfriBERTa): 0.729 ± 0.012 vs 0.731 ± 0.006; negative recall 0.686 vs 0.684.

**What we learned**
1. **RQ2: models trained on diacritised text are brittle to how users actually type** (−6 to −13 points). Each "specialist" (trained on one form) is best only on that form. **Augmenting training with every written form (mixed3) gives near-uniform performance** (worst-case drop −1.5 for TF-IDF, −1.0 for AfriBERTa) at no meaningful cost on original text. This is the project's main practical finding, and it drives the deployed model.
2. With mixed3, **TF-IDF matches or edges AfriBERTa in every form** (0.732/0.720/0.717 vs 0.723/0.716/0.713). Character n-grams over all spellings are a very strong, cheap solution.
3. **RQ3: adding Hausa/Igbo/Pidgin hurts Yoruba** (−1.5). Removing Pidgin (skewed labels) makes no difference, so the cause is diluting Yoruba (about 30% of the training mix) with other languages and domains, not the label prior.
4. **Class weighting does nothing:** the imbalance is mild, and negative-class errors come from ambiguity (see Phase 6 error analysis).
5. **E9: learning curve** (`results/figures/e9_learning_curve.png`; clean-dev macro-F1 at 10/25/50/100% of training data, 3 seeds): TF-IDF 0.594 / 0.644 / 0.676 / 0.722; AfriBERTa 0.642 / 0.666 / 0.688 / 0.731. **Pre-training helps most when labelled data is scarce** (+4.8 at 852 tweets, +0.9 at 8,522). As data grows, character n-grams catch up, which explains why transformers do not clearly beat TF-IDF on the full set. Neither curve has plateaued, so more labelled Yoruba data would still help.

**Candidates for final test evaluation (Phase 6, chosen on dev only):** E1c (main baseline), E6b mixed3 TF-IDF, E2g (best RNN), E4 AfriBERTa (best on original text), E6a mixed3 AfriBERTa (most robust transformer), E5c AfroXLMR-large LoRA.

<details><summary>Original Phase 5 plan</summary>


All ablations use the best model from Phase 4. If GPU time is short, run them on AfroXLMR-base so the comparison stays like-for-like.

| ID | Question | Setup |
|---|---|---|
| E6 | Tone-mark robustness (RQ2) | Train on {original, tones stripped, all diacritics stripped, mixed-augmentation (each training tweet seen in both forms)} × evaluate on dev in each form. Present as a heatmap matrix |
| E6′ | Same question for the baseline | Repeat on TF-IDF + LR to compare: are char n-grams naturally more robust? |
| E7 | Cross-lingual transfer (RQ3) | Train on yor + {hau, ibo, pcm} and yor + {hau, ibo} (pcm is only 1.4% neutral); evaluate on yor dev |
| E8 | Class imbalance | Weighted cross-entropy (inverse frequency) vs standard |
| E9 (optional) | Learning curve | Train on 10/25/50/100% of yor data; plot macro-F1 vs size for TF-IDF vs the best transformer |

**Done when:** all ablation rows are logged, and each one has a one-sentence answer for the report.

</details>

---

## Phase 6 — Final test evaluation and error analysis (11 Oct) ✅ done (4 Oct); optional native-speaker verification open

**Test results** (`results/test_results.md`; each model re-trained with its dev-selected configuration and scored once on test)

| Model | clean-test macro-F1 | test weighted-F1 (×100) | Δ no tones | Δ no diacritics |
|---|---|---|---|---|
| **TF-IDF mixed3 (E6b)** | **0.745** | **77.8** | +0.001 | −0.006 |
| **AfriBERTa mixed3 (E6a)** | **0.741 ± 0.007** | 77.5 | −0.002 | −0.012 |
| TF-IDF (E1c) | 0.737 | 77.2 | −0.081 | −0.084 |
| AfriBERTa (E4) | 0.732 ± 0.005 | 76.7 | −0.038 | −0.070 |
| BiLSTM + attention, Word2Vec (E2g) | 0.710 ± 0.005 | 74.7 | −0.075 | −0.085 |
| AfroXLMR-large + LoRA (E5c) | 0.702 ± 0.001 | 73.2 | −0.001 | −0.034 |
| *AfriSenti paper: AfroXLMR-large / AfriBERTa-large* | – | *74.1 / 72.9* | | |
| *SemEval-2023 best Yoruba (king001 / NLNDE)* | – | *80.2 / 80.0* | | |

**What we learned**
1. **Dev-based selection generalised:** model ranking on test matches dev, and dev → test changes are small (−0.5 to +1.5 points).
2. **No model significantly beats the TF-IDF baseline on test** (paired bootstrap: best p = 0.055 for TF-IDF mixed3; AfriBERTa seeds p = 0.13–0.87).
3. **mixed3 augmentation transfers to test:** both mixed3 models lose ≤ 1.2 points on undiacritised input (vs 7–8 without augmentation) and are the two best models overall.
4. **Our single models beat the AfriSenti paper's single-model baselines by 3–5 weighted-F1 points** (AfriBERTa 76.7 vs 72.9; TF-IDF 77.2 vs AfroXLMR-large 74.1). A plausible contributor is matching the organisers' test normalisation in training (E1d shows raw-text training costs 3.9 points on dev). The SemEval winners (about 80) used extra pre-training and/or ensembles.
5. **Error analysis** (`results/error_analysis/slices.md`; AfriBERTa vs TF-IDF on clean test):
   - Code-switched tweets cost about 5 points; undiacritised tweets about 5 (AfriBERTa) to 6 (TF-IDF).
   - **64% of AfriBERTa's errors are shared with TF-IDF** (a hard core of 651 tweets); each model fixes about 360 of the other's errors, so the two are complementary.
   - AfriBERTa is **over-confident**: 94% of its predictions have p > 0.9, but those are 77% accurate.
   - Confusions are spread out; positive → neutral is the most common (219).
   - **Meaning-based tags** (`errors_tagged.csv`, 100 errors): model-assisted by Claude, with confidence and gloss per row; **unverified**. Top categories: proverb/idiom 28, news/factual 24, missing diacritics 22, too short 20, neutral-negative boundary 19, likely label noise 16, religious/greeting 11, code-switching 8, sarcasm 7.
   - **Reliability plan:** (a) native-speaker verification of 50 rows (`verification_sheet.csv` + `CODEBOOK.md`; `python -m src.tag_agreement` gives per-category Cohen's κ); (b) **statistical label-noise estimate by confident learning** (Northcutt et al., 2021; `src/label_noise.py`): **10.8% of training tweets** (negative 14.0%, neutral 10.6%, positive 9.3%), 10.0% of dev and 9.4% of test are flagged as likely label issues. That fits κ = 0.65 and implies an accuracy ceiling of roughly 90%. Reading-based "label noise" tags did **not** match the statistical flags (31% vs 31%), so noise claims rest on the statistical estimate.
   - **Reviewer-free evidence** (no Yoruba reader available): **duplicate-label conflicts**: 15 of 386 test tweets that duplicate a training tweet have a different gold label, and AfriBERTa predicts the training copy's label in all 15. **Keyword traps**: religious/greeting words → 49% of slice errors are false positives; negative-content words → 77% of slice errors follow the negative word; negation (*kò/kì í*) costs 6–9 accuracy points.
   - **Notebook:** `notebooks/03_error_analysis.ipynb` (executed). Quantitative claims rest only on objective evidence; meaning-based tags appear as 8 clearly labelled illustrative examples. Native-speaker verification (`verification_sheet.csv`) stays available as an optional improvement.

<details><summary>Original Phase 6 plan</summary>


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

</details>

---

## Phase 7 — Deployment (12 Oct) 🔄 built and tested (4 Oct); Streamlit Cloud deploy pending (user sign-in)

**What was built**
- **Model repo:** [`Hassanadelani1/imolara-afriberta-mixed3`](https://huggingface.co/Hassanadelani1/imolara-afriberta-mixed3). Contains AfriBERTa-large mixed3 (seed 42; clean dev 0.723/0.716/0.708, consistent with E6a), the per-channel **int8 ONNX** export (127 MB), the TF-IDF mixed3 model (7.5 MB) and a model card.
- **Hosting change:** Hugging Face refused free Gradio Spaces (ZeroGPU and CPU both need PRO), so the app is deployed on **Streamlit Community Cloud** (`streamlit_app/`). The Gradio version (`app/`) is kept for PRO or local use.
- **ONNX validation** (`results/onnx_validation.md`): tokenizer ids identical; ONNX fp32 equals PyTorch (max |Δlogit| 3e-5); per-channel int8 agrees on 96.3–96.6% of clean-dev tweets, with macro-F1 within ±0.3 points (per-tensor int8: 94–95%, −1.1 points, rejected).
- **App tests** (Streamlit `AppTest`, models fetched from the Hub): all examples work, no exceptions, cold start 54 s, **peak memory about 550 MB** (limit about 1 GB), about 0.05 s per prediction.
- **Deploy steps:** `docs/DEPLOYMENT.md`. Sign in at share.streamlit.io with GitHub, set main file `streamlit_app/streamlit_app.py`, choose Python 3.12, deploy.

<details><summary>Original Phase 7 plan</summary>


1. Push the best model, tokenizer and model card to the HF Hub (`<user>/yoruba-sentiment-afroxlmr`). The model card covers data, metrics, limitations and the license.
2. **`app/app.py` (Gradio)**
   - Text box for Yoruba input. The same preprocessing as training (NFC, masking) is applied automatically.
   - Output: predicted label with class probabilities as a bar chart.
   - Example buttons: positive, negative and neutral tweets, the same tweet with and without tone marks, a code-switched tweet, and a known failure case.
   - "About" panel: model, data, test macro-F1 and limitations.
3. **Hosting:** Gradio on a HF **ZeroGPU** Space. Wrap inference in `@spaces.GPU` (short duration; one prediction takes well under 1 s). Free visitors get about 5 min of GPU a day, which is plenty for a demo. Load the model once at start-up. If ZeroGPU misbehaves, fall back to Streamlit Community Cloud on CPU with the AfroXLMR-base checkpoint (int8-quantised if needed), and state this in the report.
4. Check latency (under 2 s per query is the target) and the cold start, and confirm the app works from an incognito window and on a phone.

**Done when:** a public URL works and the app's predictions match the offline predictions on 10 test tweets.

</details>

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

Target **9:15** (hard limit 7–10 min). Each segment maps to a rubric criterion. Rehearse with a timer, because the model and experiments segments overrun most easily.

| Time | Segment | Rubric criterion |
|---|---|---|
| 0:00–0:45 | Problem, Yoruba, motivation, users | Problem definition |
| 0:45–1:45 | Dataset, EDA, preprocessing, tone-mark issue | Problem definition |
| 1:45–3:30 | Models: baseline → BiLSTM + attention → transformers + LoRA; architecture, inputs/outputs, hyperparameters, *why*; show key code | Model development |
| 3:30–5:00 | Experiments table: why each one was run, what changed, what was learned | Baseline & experimentation |
| 5:00–6:30 | Metrics (why macro-F1), results with CIs, error examples and categories | Evaluation & error analysis |
| 6:30–8:30 | Live app: how it connects to the model, good / tone-stripped / code-switched / failure inputs | Deployment |
| 8:30–9:15 | Limitations, lessons, future work | Technical understanding |

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
| Model development (10) | Phases 3–4 + video 1:45–3:30 |
| Baseline & experiments (8) | Phases 2, 5 + `experiments.csv` + video 3:30–5:00 |
| Evaluation & error analysis (7) | Phase 6 + video 5:00–6:30 |
| Report (10) | Phase 8 |
| Deployment (7) | Phase 7 + video 6:30–8:30 |
| Code & GitHub (5) | Phase 9 |
| Individual understanding (5) | Writing the code myself, viva notes, explaining *why* throughout the video |

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle GPU quota runs out | Smoke-test first; base models before large; LoRA for large; run seeds in one session; track hours after each run. The code is platform-agnostic, so Colab is an emergency fallback |
| AfroXLMR-large runs out of memory on a T4 | fp16 + gradient accumulation + max length 128; otherwise LoRA only |
| ZeroGPU Space fails or its quota is exhausted during marking | Streamlit Community Cloud (CPU) with a quantised base model; keep a recorded demo in the video as evidence |
| Label noise limits scores | Measure and discuss it in error analysis; it is a finding, not a failure |
| Complex model doesn't beat baseline | Explain it (data size, domain, noise); the rubric explicitly allows this |
| Schedule slips | E9 and the extra E7 variants are optional; core path is E0, E1, E2a, E3b, E5a, E5c, E6 |
| Accidental test-set tuning | Test is loaded only with `--final` (Phase 6); selection uses clean dev |
