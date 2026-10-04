# Ìmọ̀lára: Yoruba Tweet Sentiment Classification

*Ìmọ̀lára* (Yoruba: "feelings, emotions") classifies Yoruba tweets as **negative**, **neutral** or **positive**. It compares classical,
recurrent and Transformer models on AfriSenti-SemEval 2023, and studies a problem that matters in practice: **Yoruba is often typed
without its tone marks and under-dots**, and models trained on diacritised text break when that happens.

| | |
|---|---|
| **Live app** | **https://imolara-yoruba-sentiment-oafn6nuzsde8db6a9fdahs.streamlit.app/** (Streamlit; may take ~1 min to wake up) |
| **Model** | [Hassanadelani1/imolara-afriberta-mixed3](https://huggingface.co/Hassanadelani1/imolara-afriberta-mixed3) (Hugging Face Hub) |
| **Report** | [`report/main.pdf`](report/main.pdf) (ACL-style LaTeX source in [`report/`](report/)) |
| **Demo video** | https://youtu.be/bsjjJIeKi5s |
| **Course** | NLP Summative Project: Option 3, Text Classification for an African Language |

---

## Key results

**Test set** (AfriSenti Yoruba, scored once after model selection on dev; *clean* = the 4,129 test tweets that do not duplicate a
training tweet; weighted-F1 is the official SemEval-2023 metric). Full tables: [`results/test_results.md`](results/test_results.md).

| Model | clean-test macro-F1 | test weighted-F1 ×100 | Δ without tone marks | Δ without any diacritics |
|---|---|---|---|---|
| **TF-IDF word+char + LR, mixed3** | **0.745** | **77.8** | +0.001 | −0.006 |
| **AfriBERTa-large, mixed3** | **0.741 ± 0.007** | 77.5 | −0.002 | −0.012 |
| TF-IDF word+char + LR (baseline) | 0.737 | 77.2 | −0.081 | −0.084 |
| AfriBERTa-large | 0.732 ± 0.005 | 76.7 | −0.038 | −0.070 |
| BiLSTM + attention (Word2Vec) | 0.710 ± 0.005 | 74.7 | −0.075 | −0.085 |
| AfroXLMR-large + LoRA | 0.702 ± 0.001 | 73.2 | −0.001 | −0.034 |
| *AfriSenti paper: AfroXLMR-large / AfriBERTa-large* | | *74.1 / 72.9* | | |
| *SemEval-2023 best Yoruba system* | | *80.2* | | |

*mixed3* = every training tweet included three times: as written, without tone marks, and without any diacritics.

**Main findings**
1. **Diacritic robustness (RQ2).** Models trained on normally written Yoruba lose 6–13 macro-F1 points when users type without diacritics.
   Training on all three written forms (*mixed3*) removes almost all of that loss, with no cost on normal text
   ([E6 figure](results/figures/e6_diacritic_matrix.png)).
2. **Pre-training coverage (RQ1).** XLM-R (no Yoruba in pre-training) → AfroXLMR (the same model adapted to African languages incl. Yoruba): **+3.7 points**.
   The small Africa-only **AfriBERTa (126M) beats AfroXLMR-large (560M)**.
3. **Transformers vs n-grams.** No model significantly beats the tuned TF-IDF baseline (paired bootstrap, p ≥ 0.055). The learning curve
   shows why: pre-training helps most with little data (+4.8 points at 10% of the data), and the gap closes to +0.9 at the full 8.5k tweets
   ([E9 figure](results/figures/e9_learning_curve.png)).
4. **Cross-lingual data (RQ3).** Adding Hausa/Igbo/Pidgin training data *hurts* Yoruba (−1.5 points).
5. **Label noise.** About 10% of AfriSenti Yoruba labels look unreliable (confident learning). 15 test tweets duplicate a training tweet
   with a different label. This caps the achievable accuracy ([error analysis](notebooks/03_error_analysis.ipynb)).
6. **Data-format finding.** The released test split is pre-normalised (lower-cased, no punctuation/hashtags/emojis) while train/dev are
   raw. We reconstruct that normalisation and apply it everywhere; training on raw text costs 3.9 points.

---

## Problem, data and method (summary)

- **Data.** [AfriSenti-SemEval 2023](https://github.com/afrisenti-semeval/afrisent-semeval-2023) Yoruba (Muhammad et al., 2023; CC BY 4.0):
  8,522 / 2,090 / 4,515 tweets (train/dev/test). Raw TSVs are pinned to commit `5aec3cf` and downloaded automatically (`src/data.py`).
- **Preprocessing.** NFC; the organisers' test normalisation, reconstructed and validated on train/test duplicate pairs; optional tone-mark
  or full diacritic removal. Duplicates between splits (13% of dev, 8.5% of test) are reported separately (*all* vs *clean*), and model
  selection uses **clean dev**.
- **Models.** Majority; TF-IDF (word / char / word+char) + logistic regression or linear SVM; BiLSTM/BiGRU + additive attention over
  fastText or our own Word2Vec (trained on Yoruba Wikipedia); fine-tuned mBERT, XLM-R, AfriBERTa, AfroXLMR-base/large (full and LoRA).
- **Evaluation.** Macro-F1 (primary), weighted-F1 (official), per-class F1, 95% bootstrap CIs, paired bootstrap tests, 3 seeds for every
  neural model. The test set was used once, in Phase 6.

Details, decisions and every experiment: [`PLAN.md`](PLAN.md) · data checks and sources: [`docs/verification.md`](docs/verification.md) ·
EDA: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) · all dev results: [`results/summary.md`](results/summary.md).

### Experiments (clean-dev macro-F1, mean ± std over 3 seeds where applicable)

| ID | Experiment | macro-F1 | Notes |
|---|---|---|---|
| E0 | Majority class | 0.198 | lower bound |
| E1c | **TF-IDF word+char + LR (baseline)** | **0.722** | E1a word 0.706, E1b char SVM 0.714 |
| E1d | E1c trained on raw tweets | 0.684 | cost of ignoring the test-set normalisation |
| E2 | BiLSTM/BiGRU + attention | 0.590 – 0.699 | best: own Word2Vec + regularisation (E2g); frozen fastText worst |
| E3a / E3b | mBERT / XLM-R-base | 0.674 / 0.650 | XLM-R has no Yoruba in pre-training |
| E4 | **AfriBERTa-large** | **0.731 ± 0.006** | best on dev |
| E5a / E5b / E5c | AfroXLMR-base / large / large+LoRA | 0.687 / 0.714 ± 0.029 / 0.708 ± 0.004 | LoRA: 0.47% of parameters, 7× more stable |
| E6 | Diacritic augmentation (TF-IDF and AfriBERTa) | 0.732 / 0.723 (mixed3) | worst-case drop −10 → −1.5 points |
| E7 | AfriBERTa + Hausa/Igbo(/Pidgin) | 0.715 / 0.716 | negative transfer |
| E8 | AfriBERTa + class weights | 0.729 | no effect |
| E9 | Learning curve (10/25/50/100%) | see figure | gap closes with more data |

---

## Repository layout

```
src/                     all code (run as modules: python -m src.<name>)
  data.py                AfriSenti loading, normalisation, diacritic modes, overlap detection
  eda.py                 EDA statistics and figures
  baselines.py           E0–E1, E6′ (TF-IDF models), configs/baselines.yaml
  embeddings.py, rnn.py  fastText / Word2Vec and BiLSTM/BiGRU + attention (E2)
  train_transformer.py   Transformer fine-tuning incl. LoRA (E3–E9)
  evaluate.py            metrics, bootstrap CIs, run records, results collection
  summarize.py, figures.py, final_eval.py, learning_curve.py
  error_analysis.py, label_noise.py, tag_agreement.py
  export_tfidf.py, export_onnx.py      deployment exports
configs/                 one YAML per neural experiment; baselines.yaml for E0/E1/E6′
scripts/                 kaggle_run.py (launch/fetch Kaggle GPU runs), kaggle_notebook.py, kaggle_dataset.py, deploy_hf.py
notebooks/               01_eda, 03_error_analysis; kaggle/ = executed Kaggle notebooks (one per GPU experiment, real outputs)
results/                 experiments.csv, summary.md, test_results.md, figures/, baselines/, kaggle/ (per-run outputs), error_analysis/
streamlit_app/           deployed web app (ONNX int8 AfriBERTa + TF-IDF)
app/                     Gradio version of the app (for a Hugging Face PRO Space or local use) and the model card
docs/                    verification.md (facts and pipeline issues), KAGGLE_GUIDE.md, DEPLOYMENT.md
tests/                   unit and regression tests (pytest)
```

---

## Reproducing the results

**Setup** (Python 3.12; CPU is enough for everything except neural training):
```bash
git clone https://github.com/Hassan-Adelani-Luqman/imolara-yoruba-sentiment.git && cd imolara-yoruba-sentiment
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU build (see the comment in requirements.txt)
pip install -r requirements.txt
pytest -q                                                              # 12 tests
```

**CPU stages** (data is downloaded on first use):
```bash
python -m src.eda                      # Phase 1: data statistics + figures
python -m src.baselines                # Phase 2 (+ E6′): TF-IDF baselines → results/experiments.csv
python -m src.learning_curve           # E9 (TF-IDF side)
python -m src.embeddings word2vec      # Phase 3: train our Word2Vec (≈ 10 min); fastText is downloaded on demand
python -m src.summarize                # results/summary.md
python -m src.figures                  # report figures
```

**GPU stages.** Every neural experiment is one config, and runs the same way anywhere:
```bash
python -m src.train_transformer --config configs/e4_afriberta_large.yaml --seed 42 --output_dir out/e4/seed42
python -m src.rnn --config configs/e2g_bilstm_word2vec_reg.yaml --seed 42 --output_dir out/e2g/seed42
python -m src.evaluate collect out/e4  # add the runs to results/experiments.csv
```
We ran them on **Kaggle T4 GPUs** with one command per experiment. It builds a notebook that clones this repository at the current commit, trains
the 3 seeds two at a time (one per GPU) and downloads the results and the executed notebook ([`docs/KAGGLE_GUIDE.md`](docs/KAGGLE_GUIDE.md)):
```bash
python scripts/kaggle_run.py configs/e4_afriberta_large.yaml --wait            # dev
python scripts/kaggle_run.py configs/e4_afriberta_large.yaml --final --wait    # Phase 6: + test
```

**Phase 6 (test) and error analysis** (needs the `*_final` runs, which are included in `results/`):
```bash
python -m src.baselines --final --only e0_majority e1c_wordchar_lr e6b_wordchar_lr_mixed3
python -m src.final_eval               # results/test_results.md
python -m src.error_analysis           # results/error_analysis/
python -m src.label_noise              # confident-learning label-noise estimate
```

**Deployment** ([`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)):
```bash
python -m src.export_tfidf && python -m src.export_onnx     # models/ (+ validation: results/onnx_validation.md)
streamlit run streamlit_app/streamlit_app.py                 # local app
```

**Reproducibility notes.** Seeds 42/43/44 for neural models. TF-IDF models are deterministic. Neural re-runs on GPU are not bit-identical:
the Phase 6 re-trainings reproduce the original dev scores to within ±0.003 macro-F1 (e.g. AfriBERTa 0.731 → 0.732).
Package versions used on Kaggle are recorded in each run's `env.txt`.

---

## Engineering notes

Several silent pipeline problems were found and fixed during the project. Each fix is documented in [`docs/verification.md`](docs/verification.md),
and most are guarded by a test or an automatic check:
- Kaggle's T4 machines have 2 GPUs → the HF Trainer silently used DataParallel (doubled batch).
- transformers 5 loads checkpoints in their stored dtype, and AfroXLMR-large is fp16.
- `group_by_length` also reorders *evaluation* batches in transformers 5 → scrambled predictions (regression test `tests/test_prediction_order.py`).
- scikit-learn's default tokeniser breaks Yoruba words at combining tone marks.
- The fastText Yoruba vocabulary has no tone-marked under-dotted vowels (tiered lookup added).
- Every run records `selection_consistent`: the final model must reproduce its best dev score.

---

## Credits and licences

- **Data:** AfriSenti-SemEval 2023 (Muhammad et al., 2023), CC BY 4.0. Yoruba Wikipedia (Wikimedia), CC BY-SA 4.0 (Word2Vec training text).
- **Pre-trained models:** AfriBERTa (Ogueji et al., 2021, MIT); AfroXLMR (Alabi et al., 2022, MIT); XLM-R (Conneau et al., 2020);
  mBERT (Devlin et al., 2019); fastText Yoruba vectors (Grave et al., 2018, CC BY-SA 3.0).
- **Methods:** LoRA (Hu et al., 2022); confident learning (Northcutt, Jiang & Chuang, 2021); additive attention (Bahdanau et al., 2015).
- **Libraries:** PyTorch, Hugging Face Transformers / Datasets / PEFT / Hub, scikit-learn, gensim, ONNX Runtime, Streamlit, Gradio, pandas, matplotlib.
- **Compute:** Kaggle Notebooks (NVIDIA T4).

Full references are in the report.

**Licence:** code in this repository is released under the [MIT licence](LICENSE). Data and pre-trained models keep their own licences (above).
