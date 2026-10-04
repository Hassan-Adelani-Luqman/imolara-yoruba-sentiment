# Ìmọ̀lára — Demo video script (target 9:15, limit 7–10 min)

**How to use this.** Each segment has a time window, **SHOW** (what is on screen) and **SAY** (word for word, about 140 words/min).
Rehearse once with a timer. The model and experiment segments are where you will overrun, so trim the *optional* sentences (marked ⟨…⟩) if you are behind.
Speak in your own voice: it is fine to paraphrase, as long as every point and number stays.

**Pronunciation (approximate):** Ìmọ̀lára ≈ *ee-maw-LAH-rah* (low–mid–high–mid tones) · Yorùbá ≈ *YOH-roo-bah* · AfriBERTa ≈ *afri-BER-ta*.

---

## Before recording (checklist)

1. **Wake the app** 5 minutes before: open https://imolara-yoruba-sentiment-oafn6nuzsde8db6a9fdahs.streamlit.app/ and wait until the models have loaded (first load ≈ 1 min).
2. Open these **tabs/windows in this order** (left to right), each zoomed so text is readable (editor font ≥ 16 pt, browser zoom 125%):
   1. `report/main.pdf` — page 1 (title + abstract)
   2. VS Code: `notebooks/01_eda.ipynb` scrolled to **§2 The train/test format mismatch** (the surface-feature chart)
   3. VS Code: `src/data.py` at `def normalize_semeval`
   4. VS Code: `src/baselines.py` at `def build_features`
   5. VS Code: `src/rnn.py` at `class AdditiveAttention`
   6. VS Code: `src/train_transformer.py` at `def build_model`, and `configs/e4_afriberta_large.yaml` + `configs/e5c_afroxlmr_large_lora.yaml`
   7. VS Code: `notebooks/kaggle/e4_afriberta_large.ipynb` scrolled to the training-log cell
   8. `report/main.pdf` — Table 3 (dev results, page 4) and Figure 2 (E6 heatmap, page 5)
   9. `results/figures/e9_learning_curve.png`
   10. `report/main.pdf` — Table 4 (test results, page 5)
   11. VS Code: `notebooks/03_error_analysis.ipynb` scrolled to **§4 Keyword traps** and **§5 duplicate-label conflicts**
   12. The live app
   13. VS Code: `streamlit_app/streamlit_app.py` at `def load_models` / `def afriberta_probs`
3. Turn off notifications; close unrelated tabs. Record with OBS (screen + webcam corner), 1080p.
4. Upload as **unlisted** (YouTube) or shared-link (Drive) and check the link opens in a private window. Paste it into `README.md` and `report/main.tex` (the red "[link to be added]"), recompile the PDF.

---

## 1 · Introduction and problem — 0:00–0:40

**SHOW:** `report/main.pdf`, title page (abstract visible).

**SAY:**
> Hello, I'm Hassan Adelani Luqman. My project is **Ìmọ̀lára** — Yoruba for "feelings" — a sentiment classifier for Yorùbá tweets: it labels a tweet as negative, neutral or positive.
> Yorùbá has around fifty million speakers but very few NLP tools. And it has a practical problem: written Yorùbá uses tone marks and under-dots, but people very often type it *without* them on their phones. So my questions were: do African-centred language models beat general ones and a classical baseline; how robust are the models to missing diacritics, and can we fix that; and does data from other Nigerian languages help?

*(100 words ≈ 40 s)*

---

## 2 · Dataset and preparation — 0:40–1:42

**SHOW:** `notebooks/01_eda.ipynb`, the surface-feature bar chart (§2). Then switch to `src/data.py`, `normalize_semeval`.

**SAY:**
> I used the Yorùbá part of AfriSenti, the SemEval-2023 benchmark: eight and a half thousand training tweets, about two thousand dev and four and a half thousand test, annotated by three native speakers each. Negative is the minority class, about twenty-two percent, so my main metric is **macro-F1**, which weights all three classes equally.
> Exploring the data, I found something important. *[point at the chart]* The training and dev tweets are raw — uppercase, punctuation, hashtags — but the released **test set had already been normalised** by the organisers: lower-cased, no punctuation, no hashtags. A model trained on raw tweets would be tested on a different format.
> *[switch to code]* So I reconstructed their normalisation here, in `normalize_semeval`, verified it reproduces 76 of 79 tweets that appear in both train and test, and applied it to every split. I also found that thirteen percent of dev tweets duplicate a training tweet, so I select models on a **clean** dev subset without those duplicates.

*(156 words ≈ 1:02 — drop nothing here; it carries the "Problem & Dataset" criterion)*

---

## 3 · Models — 1:42–3:33

**SHOW:** `src/baselines.py` → `build_features`.

**SAY:**
> My **baseline** is TF-IDF with logistic regression. Each tweet becomes a sparse vector of word unigrams and bigrams **plus character 2-to-5-grams**. I chose character n-grams because they still match words that are spelled differently or have missing tone marks. One detail: scikit-learn's default tokenizer breaks Yorùbá words at combining tone marks and drops one-letter words like "o", so I split on whitespace instead.

**SHOW:** `src/rnn.py` → `AdditiveAttention` and `RNNClassifier.forward`.

**SAY:**
> Next, recurrent models. The input tokens go through a 300-dimensional embedding layer — either fastText vectors or a Word2Vec model I trained myself on Yorùbá Wikipedia — then a bidirectional LSTM or GRU. Additive attention scores every hidden state, a softmax turns the scores into weights, and the weighted sum is classified by a linear layer into the three classes. Trained with Adam, learning rate 0.001, and early stopping on clean-dev macro-F1.

**SHOW:** `src/train_transformer.py` → `build_model`; then `configs/e4_afriberta_large.yaml` and `configs/e5c_afroxlmr_large_lora.yaml`.

**SAY:**
> Then Transformers, using Hugging Face: mBERT, XLM-R, AfriBERTa and AfroXLMR. Each tweet is tokenized into subwords, up to 128 tokens, passed through the pre-trained encoder, and a classification head on top outputs the three class probabilities. I fine-tune the whole model with a learning rate around two to three times ten to the minus five, batch size 32, ten percent warm-up, and early stopping.
> For the 560-million-parameter AfroXLMR-large I also used **LoRA**: the original weights stay frozen and small rank-16 matrices are added to the attention query and value projections, so only 2.6 million parameters — under half a percent — are trained.

**SHOW:** `notebooks/kaggle/e4_afriberta_large.ipynb`, the training-log cell.

**SAY:**
> ⟨optional⟩ All GPU training ran on Kaggle. Each experiment is a notebook that clones my repository at the exact commit, trains three seeds — one per GPU — and saves this executed notebook with its real outputs back into the repository.

*(278 words ≈ 1:51 — the longest segment; the ⟨optional⟩ Kaggle paragraph is the first thing to cut)*

---

## 4 · Baseline and experiments — 3:33–5:12

**SHOW:** `report/main.pdf`, Table 3 (dev results).

**SAY:**
> Here are the dev results. The TF-IDF baseline reaches 0.722 macro-F1 — I chose it as the baseline because n-gram models are strong on short texts, and it tells us whether deep models are worth it. Training it on raw tweets instead drops it to 0.684, which confirms the format issue mattered.
> The recurrent models never beat it: the best is 0.699. They overfit after two or three epochs, and words never seen in training become unknown tokens, while character n-grams still match them.
> For the Transformers, the cleanest comparison is XLM-R versus AfroXLMR: same architecture, but AfroXLMR was further pre-trained on African languages including Yorùbá — that alone adds 3.7 points. The best model is AfriBERTa, 0.731, a small model trained only on African languages. LoRA matched full fine-tuning while being seven times more stable across seeds.

**SHOW:** Figure 2 (E6 heatmap).

**SAY:**
> My main experiment is robustness. Rows are how a model was trained, columns how it is tested. Trained on normal text, both models lose six to thirteen points when tested without diacritics. So I trained on every tweet in all three forms — with tones, without tones, without any diacritics — called mixed3, the bottom row: performance is almost the same in every column, at no cost.

**SHOW:** `results/figures/e9_learning_curve.png`.

**SAY:**
> ⟨optional⟩ This learning curve explains why Transformers don't pull ahead: their advantage is almost five points with ten percent of the data, but under one point with all of it. Adding Hausa, Igbo and Pidgin data actually hurt, and class weights did nothing.

*(248 words ≈ 1:39, of which 43 optional)*

---

## 5 · Evaluation and error analysis — 5:12–6:22

**SHOW:** `report/main.pdf`, Table 4 (test results).

**SAY:**
> On the test set, which I used only once after choosing models on dev, the two mixed3 models are best: TF-IDF 0.745 and AfriBERTa 0.741 macro-F1, and they lose at most about one point on text without diacritics. I report macro-F1 because it treats the small negative class fairly, and weighted-F1 because it is the official SemEval metric — on that scale we get 77.8, above the 74.1 of the strongest single model published with the dataset. I also ran a paired bootstrap test: no model is *significantly* better than the tuned baseline, which is itself a finding.

**SHOW:** `notebooks/03_error_analysis.ipynb`, §4 Keyword traps, then §5 duplicate-label conflicts.

**SAY:**
> For errors, two-thirds of AfriBERTa's mistakes are also made by TF-IDF — a hard core. Some are keyword traps: tweets with religious words like "Ọlọ́run" are mostly positive, so the model calls a neutral Bible verse positive. Negation with "kò" lowers accuracy by about six points. And some errors are in the labels themselves: fifteen test tweets are copies of training tweets with a *different* label, and a confident-learning analysis suggests about ten percent of labels are unreliable.

*(176 words ≈ 1:10)*

---

## 6 · Live demo — 6:22–8:28

**SHOW:** the live app.

**SAY + DO** (do each action as you say it):
> This is the deployed app on Streamlit Cloud. *[select "Positive greeting"]* "Ẹ kú àbọ̀ o! Inú mi dùn púpọ̀" — "welcome, I'm very happy to see you". Both models say positive, with the probabilities shown here for AfriBERTa on the left and TF-IDF on the right.
> *[select "Same, typed without diacritics"]* Now the same sentence typed without any diacritics — still positive. That is the mixed3 training at work.
> *[select "Negative complaint"]* "This government has ruined us, no electricity, no water" — negative. *[select "Neutral news"]* A news sentence about a governors' meeting — neutral.
> *[select "Positive greeting" again, then click "Remove all diacritics"]* I can also strip the diacritics myself with this switch — and this box shows exactly what the models see after normalisation.
> Now failure cases. *[select "Hard case: Bible verse"]* "In the beginning God created the heavens and the earth" is neutral in the dataset, but both models say positive — the keyword trap from the error analysis. *[select "Hard case: code-switched Pidgin complaint"]* A complaint about traffic mixed with Pidgin English — the models disagree, and the app warns the user. *[select "Hard case: idiom"]* And the idiom "the bean cake has dissolved in the oil", meaning things fell apart: AfriBERTa gets it, TF-IDF does not.

**SHOW:** `streamlit_app/streamlit_app.py` → `load_models`, `afriberta_probs`.

**SAY:**
> How it connects: the app downloads my fine-tuned AfriBERTa and the TF-IDF model from the Hugging Face Hub, runs the same preprocessing function as training, and computes the probabilities here. To fit the free server's memory, I exported AfriBERTa to ONNX and quantised it to 8-bit; I checked that it agrees with the original model on about 96 percent of dev tweets with the same macro-F1.

*(239 words ≈ 1:36 + about 30 s of clicking ≈ 2:06 — speak while the page updates)*

---

## 7 · Limitations, lessons and close — 8:28–9:10

**SHOW:** `report/main.pdf`, §8 Limitations (or the README "Main findings").

**SAY:**
> Limitations: the data is only Twitter, about ten percent of labels look noisy, the models are over-confident, and idioms and negation remain hard. Next I would re-annotate flagged tweets, try adaptive pre-training, and combine both models, since their errors are partly different.
> My main lessons: audit the data before modelling — the format mismatch and duplicates changed my results more than any model choice; use several seeds and significance tests before claiming a winner; and test a model the way people will actually use it. Thank you — the code, notebooks, report and the live app are all linked in the repository.

*(102 words ≈ 41 s)*

---

### Timing summary

| Segment | Window | Spoken words | Rubric evidence |
|---|---|---|---|
| 1 Problem | 0:00–0:40 | 100 | Problem definition |
| 2 Data | 0:40–1:42 | 156 | Problem definition, dataset |
| 3 Models | 1:42–3:33 | 278 | Model development (architecture, input/output, embeddings, training, hyperparameters, decisions, code) |
| 4 Experiments | 3:33–5:12 | 248 | Baseline & experimentation (why baseline, what changed, why, what was learned) |
| 5 Evaluation | 5:12–6:22 | 176 | Evaluation & error analysis (why metrics, interpretation, success/failure examples) |
| 6 Demo | 6:22–8:28 | 239 + clicks | Deployment (workflow, model connection, multiple inputs, failures) |
| 7 Close | 8:28–9:10 | 102 | Technical understanding, limitations |
| **Total** | **≈ 9:10** | **1,299** | |

Times assume about 150 spoken words per minute plus about 30 s of clicking in the demo. If your rehearsal runs over **9:40**, cut the two
⟨optional⟩ paragraphs (Kaggle in segment 3, learning curve in segment 4): together they save about 35 s. If it runs under 7:30, slow down —
pausing on each figure helps the viewer.
