# Running Ìmọ̀lára GPU experiments on Kaggle

All neural experiments (E2–E9, the Phase 6 test runs and the deployment model) were trained on **Kaggle T4 GPUs**. One local command
launches a run. It waits for a free GPU, trains, and downloads both the results and the **executed notebook** into the repository. Nothing
is edited in the Kaggle web editor, and the training code itself is platform-agnostic: `python -m src.train_transformer --config …` runs
on any CUDA machine. Kaggle is only the launcher.

```
git commit + push ──▶ scripts/kaggle_run.py configs/<exp>.yaml --wait
                        │ builds a small launcher notebook (scripts/kaggle_notebook.py) and pushes it as kernel imolara-<exp>-nb
                        ▼
  Kaggle (T4 ×2, internet on)
    launcher:  git clone <this repo> @ <commit>  →  build experiment notebook  →  jupyter nbconvert --execute
    experiment notebook:  setup (clone, pip, GPU check) → config → seeds trained in parallel, one GPU each → results table
                        │ /kaggle/working/outputs/seed*/{metrics.json, predictions_*.csv, run_info.json}, <exp>.ipynb (executed)
                        ▼
  results/kaggle/<exp>/ (outputs)    notebooks/kaggle/<exp>.ipynb (executed copy)    results/experiments.csv (collected)
```

---

## 1. One-time account setup (about 10 min)

1. **Verify your phone number:** kaggle.com → *Settings* → *Phone verification*. Without it, kernels cannot use a **GPU** or the **internet**.
2. **GPU quota:** about 30 GPU-hours per week on the free tier (see the quota panel in any notebook editor). A session can run for up to about 12 h.
3. **API token:** kaggle.com/settings/api → *Generate New Token*.

## 2. Local setup

```bash
pip install kaggle                       # in the project venv (pinned in requirements.txt)
mkdir -p ~/.kaggle && echo "<TOKEN>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
echo 'export KAGGLE_USERNAME="<your-kaggle-username>"' >> ~/.bashrc && source ~/.bashrc
kaggle kernels list -m                   # should list your kernels, not show a 401
```
- Other ways to log in: `kaggle auth login` (browser), the `KAGGLE_API_TOKEN` environment variable, or the legacy `~/.kaggle/kaggle.json`.
- **GitHub push access (SSH key):** Kaggle clones the public repo at the commit you launch from, so `scripts/kaggle_run.py` pushes HEAD
  first and refuses to launch with uncommitted changes in `src/`, `configs/` or `scripts/`.
- **RNN runs** need the embeddings dataset once: `python scripts/kaggle_dataset.py` uploads `cc.yo.300.vec.gz` and our Word2Vec as the
  private dataset `<user>/imolara-embeddings`. It is attached automatically when `--module src.rnn` is used.

## 3. Commands

```bash
python scripts/kaggle_run.py configs/e4_afriberta_large.yaml --wait                    # 3 seeds (42 43 44), dev scores
python scripts/kaggle_run.py configs/e2g_bilstm_word2vec_reg.yaml --module src.rnn --wait
python scripts/kaggle_run.py configs/e4_afriberta_large.yaml --final --wait            # Phase 6: re-train + score test
python scripts/kaggle_run.py configs/deploy_afriberta_mixed3.yaml --seeds 42 --wait --with-model   # keep the weights
python scripts/kaggle_run.py configs/e4_afriberta_large.yaml --fetch-only              # re-download a finished run
```
- Without `--wait`, the command returns after the push; use `--fetch-only` later.
- `--final` runs record under `<exp>_final` so they never overwrite the original dev rows.
- Runs whose names start with `smoke` or `deploy` are not collected into `experiments.csv`.
- **Contract for any training module:** `python -m <module> --config <yaml> --seed <int> --output_dir <dir> [--final]`, writing `metrics.json`
  (a list of records), `predictions_<split>_<form>.csv`, `run_info.json` and, if `save_model: true`, `model/`.

## 4. Behaviour you can rely on

| What | How |
|---|---|
| Free tier allows **2 concurrent GPU sessions** | A refused push still exits 0, so the runner checks the push output and retries every 2 min (up to 6 h). Launches simply queue |
| Each Kaggle T4 machine has **2 GPUs** | Each seed is a separate process pinned to one GPU (`CUDA_VISIBLE_DEVICES`); two seeds run at once. Batch size in the config is the real batch (no DataParallel) |
| Code provenance | The notebook clones the repo at the full commit hash and prints it; every record stores `git_commit` |
| Downloads time out occasionally | `kernels output` is retried 4 times |
| Failed runs | `nbconvert --allow-errors`: the executed notebook (with the traceback) is still saved, and the runner downloads the log |
| Model selection sanity | Each run records `selection_consistent` (the final model re-scores to its best dev epoch) |

## 5. Measured training time (mean per seed; seeds run two at a time)

| Experiment | min/seed | | Experiment | min/seed |
|---|---|---|---|---|
| BiLSTM/BiGRU (E2) | 0.4–0.9 | | AfroXLMR-base (E5a) | 8.0 |
| AfriBERTa-large (E4) | 3.8 | | AfroXLMR-large + LoRA (E5c) | 12.5 |
| AfriBERTa mixed3 (E6a, 3× data) | 7.3 | | AfroXLMR-large full FT (E5b) | 23.6 |
| mBERT / XLM-R-base (E3) | 5.5 / 8.4 | | AfriBERTa + hau/ibo(/pcm) (E7) | 7.4–7.8 |

Add about 3–5 min per run for kernel start, package installation and model download. The whole project used roughly 10 GPU-hours,
including the failed and re-run attempts listed in `docs/verification.md`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `401 Unauthorized` | Token missing or expired: recreate it and check `chmod 600` |
| "Commit your changes first" | Uncommitted changes in `src/`, `configs/` or `scripts/`: commit (the runner pushes) |
| Run fails with "No GPU" | Phone not verified, or the weekly quota is used up |
| `ImportError: … torchao` (LoRA) | Kaggle ships an old torchao that peft rejects; the notebook uninstalls it (already handled) |
| `CUDA out of memory` | Lower `batch_size`, raise `grad_accum`, or use LoRA |
| Final scores near chance but per-epoch scores fine | Evaluation order scrambled; `WeightedTrainer` uses a sequential eval sampler (regression test `tests/test_prediction_order.py`) |
| `selection_consistent: false` | The best checkpoint was not restored or scored correctly. Investigate before using the run |

## 7. Legacy script mode

`--script` builds a packed Python kernel (code embedded as base64; template `kaggle/run_template.py`) instead of a notebook. It was used for
the Phase 3 RNN runs and the first Phase 4 attempt. Kept for reference; notebook mode is the default.
