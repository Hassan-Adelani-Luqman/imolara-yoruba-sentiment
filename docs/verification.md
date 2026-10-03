# Verified Facts Log (Phase 0)

Checked 4 Oct 2026. "Own check" means computed from the data in this repo; otherwise the source is linked.

## Data: AfriSenti, Yoruba (`yor`)

| Fact | Value | Source |
|---|---|---|
| Source used | Raw TSVs from the official repo, pinned to commit `5aec3cf` (`src/data.py`) | https://github.com/afrisenti-semeval/afrisent-semeval-2023 |
| Why not the HF copy | `shmuhammad/AfriSenti-twitter-sentiment` relies on a loading script that `datasets` ≥ 4 no longer runs. Its parquet export is lowercased, has punctuation removed and uses integer labels | Own check |
| Split sizes | **8,522 / 2,090 / 4,515** (train/dev/test). The paper's 8,523 / 2,091 / 4,516 includes the TSV header line | Own check; Muhammad et al. 2023, Table 6 |
| Labels (train) | positive 3,542 (41.6%), neutral 3,108 (36.5%), negative 1,872 (22.0%); dev and test have similar proportions | Own check |
| Tweets with tone marks | 77.2% train / 77.8% dev / 78.4% test | Own check (`has_tone_marks`) |
| Tweets with under-dots | 72.9% / 73.0% / 73.9% | Own check |
| Annotation | 3 native speakers per tweet, majority vote. Free-marginal κ for yor = **0.65** | Muhammad et al. 2023 (AfriSenti), Table 5 |
| License | CC BY 4.0 (stated on the HF dataset card; the GitHub repo has no LICENSE file) | https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment |
| Paper on diacritics | Tone "is rarely fully rendered in written form" (èdè 'language' vs edé 'crayfish' → "ede"). The data contains some code-mixed tweets. No percentages given | arXiv 2302.08956 |

**Finding to discuss in the report:** most AfriSenti Yoruba tweets *do* carry diacritics (about 77%), which goes against the usual claim that diacritics are dropped online (Orife 2018). The likely reason is that the tweets were collected with keyword lists that included diacritised words. This makes the tone-mark ablation (E6) a robustness test for users who type *without* diacritics, as most phone keyboards do.

## Auxiliary languages (E7), train sizes

hau 14,172 · ibo 10,192 · pcm 5,121. Pidgin has only 72 neutral examples (1.4%), so watch the class balance in E7.

## Models: which ones saw Yoruba in pre-training

| Model | HF id | Params | Yoruba? | Source |
|---|---|---|---|---|
| mBERT (cased) | `google-bert/bert-base-multilingual-cased` | ~178M | **Yes** (Wikipedia, 104 languages) | google-research/bert `multilingual.md` |
| XLM-R base | `FacebookAI/xlm-roberta-base` | ~278M | **No**. CC-100 Table 6 goes from xh to yi; ig is also absent | Conneau et al. 2020, App. A |
| AfriBERTa large | `castorini/afriberta_large` | ~126M | Yes (also pcm, hau, ibo) | Ogueji et al. 2021; model card |
| AfroXLMR base / large | `Davlan/afro-xlmr-base` / `-large` | ~278M / ~560M | Yes (also hau, ibo, pcm) | Alabi et al. 2022 (COLING); model card |

This gives RQ1 a clean contrast: **XLM-R (no Yoruba) vs AfroXLMR (XLM-R adapted to Yoruba and other African languages)**, which share the same architecture and tokenizer.

AfriBERTa ships only `pytorch_model.bin`. Recent `transformers` needs torch ≥ 2.6 to load `.bin` files. Check against `env.txt` from the Kaggle smoke run.

## Published reference scores (Yoruba)

| System | Score | Metric | Source |
|---|---|---|---|
| AfroXLMR-large (AfriSenti baseline) | 74.1 | weighted-F1* | AfriSenti Table 7 |
| AfriBERTa-large | 72.9 | weighted-F1* | same |
| AfroXLMR-base | 70.0 | weighted-F1* | same |
| XLM-R-base | 62.7 | weighted-F1* | same |
| SemEval-2023 Task A best (king001) | **80.16** | weighted-F1 | Muhammad et al. 2023 (SemEval), Table 3 |
| NLNDE (AfroXLMR-large + language-adaptive and task-adaptive pre-training; 1st overall) | 79.95 | weighted-F1 | same |

\*The AfriSenti Table 7 caption says "accuracy", but the SemEval paper reports the same numbers as weighted-F1. **The official SemEval metric is weighted-F1**, so we report it alongside macro-F1.

## fastText Yoruba vectors (E2), all live (HTTP 200)

| File | Size | URL |
|---|---|---|
| `cc.yo.300.vec.gz` | 85 MB | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yo.300.vec.gz |
| `wiki.yo.vec` | 58 MB | https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.yo.vec |
| `cc.yo.300.bin.gz` (subword model) | 2.4 GB | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.yo.300.bin.gz |

License: CC BY-SA 3.0. Plan: use `cc.yo.300.vec.gz` (small). The `.bin` version is needed only for subword embeddings of out-of-vocabulary words.

## Deployment: Hugging Face Spaces (affects Phase 7)

- **Free accounts can no longer create Gradio Spaces on CPU Basic.** Gradio and Docker Spaces now need PRO, except that free accounts in good standing (verified email, **account older than 30 days**) may host **up to 2 ZeroGPU Spaces**. Source: https://huggingface.co/docs/hub/spaces-overview
- ZeroGPU: Gradio only; functions decorated with `@spaces.GPU`; free visitors get about 5 min/day of GPU (anonymous visitors about 2 min). A sentiment model needs well under 1 s per query.
- **Action:** check the HF account age. Fallbacks: Streamlit Community Cloud (CPU), or a Static Space running an ONNX model in the browser.

## Language facts (for the report)

- Speakers: about 48M L1, about 50M total. Source: Ethnologue 28th ed. (2025), cited via Wikipedia; I could not open Ethnologue itself (paywall).
- Diacritics: Orife, I. (2018). Attentive Sequence-to-Sequence Learning for Diacritic Restoration of Yorùbá Language Text. *Interspeech 2018*, 2848–2852. doi:10.21437/Interspeech.2018-42
- Adelani, D. I., et al. (2021). The Effect of Domain and Diacritics in Yoruba–English Neural Machine Translation. *MT Summit XVIII*, 61–75.

## Still open

- [x] Video length: **7–10 min** (instructor).
- [x] HF account is older than 30 days → deploy on a ZeroGPU Space.
- [x] Kaggle environment (smoke run, 4 Oct): Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, datasets 4.8.5, peft 0.20.0, scikit-learn 1.6.1. torch ≥ 2.6, so AfriBERTa's `.bin` weights load. The full pipeline (install → train → 3-way diacritic eval → download → collect) took 88 s. Smoke macro-F1 0.198 = always predicting the majority class after 13 steps on 200 tweets: expected, not a bug.
- [ ] Confirm the SemEval-2023 Task 12 author list and page numbers on the ACL Anthology page.
