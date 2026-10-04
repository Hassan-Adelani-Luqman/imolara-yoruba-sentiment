# Deploying Ìmọ̀lára

**Live app:** https://imolara-yoruba-sentiment-oafn6nuzsde8db6a9fdahs.streamlit.app/

## What is deployed

| Piece | Where | Built by |
|---|---|---|
| AfriBERTa-large mixed3 (PyTorch weights + tokenizer) | HF Hub model repo [`Hassanadelani1/imolara-afriberta-mixed3`](https://huggingface.co/Hassanadelani1/imolara-afriberta-mixed3) | Kaggle run `configs/deploy_afriberta_mixed3.yaml` (seed 42, `save_model: true`) |
| `onnx/model_int8.onnx` (per-channel int8, 127 MB) | same repo | `python -m src.export_onnx` (validation: `results/onnx_validation.md`) |
| `tfidf_mixed3.joblib` (7.5 MB) | same repo | `python -m src.export_tfidf` (checks dev predictions = reported E6b run) |
| Web app (Streamlit) | Streamlit Community Cloud, from this GitHub repo | `streamlit_app/streamlit_app.py` + `streamlit_app/requirements.txt` |

Upload / refresh the model repo: `python scripts/deploy_hf.py model` (needs `hf auth login` with a write token), then
upload the ONNX file with `HfApi().upload_file(..., path_in_repo="onnx/model_int8.onnx")`.

**Why Streamlit and ONNX?** Hugging Face no longer allows free accounts to host Gradio or Docker Spaces (CPU or ZeroGPU both require PRO;
see `docs/verification.md`). Streamlit Community Cloud is free but has about 1 GB of RAM, too little for PyTorch plus a 500 MB model.
The int8 ONNX model running in `onnxruntime` with the standalone `tokenizers` library needs no PyTorch: peak memory is about 550 MB, and it
agrees with the PyTorch model on 96–97% of dev tweets (macro-F1 within ±0.3 points). The Gradio app in `app/` is kept: it runs
unchanged on a PRO Space, or locally (`python app/app.py`).

## Deploy on Streamlit Community Cloud (one-time, about 5 minutes)

1. Go to **https://share.streamlit.io** and **sign in with GitHub** (the account that owns `Hassan-Adelani-Luqman/imolara-yoruba-sentiment`).
   Authorise Streamlit to access your repositories when asked.
2. Click **Create app → Deploy a public app from GitHub** and fill in:
   - **Repository:** `Hassan-Adelani-Luqman/imolara-yoruba-sentiment`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app/streamlit_app.py`
   - **App URL (optional):** e.g. `imolara-yoruba` → `https://imolara-yoruba.streamlit.app`
3. Open **Advanced settings** and choose **Python 3.12**. No secrets are needed (the model repo is public).
4. Click **Deploy**. The first build installs `streamlit_app/requirements.txt` (Streamlit uses the requirements file next to the
   entry point, not the heavy root `requirements.txt`) and takes a few minutes. The first page load downloads about 140 MB of models (about 1 minute).
5. Test the examples, then put the URL in the README and the report.

**Good to know**
- The app **sleeps after about 12 hours without visitors**. Opening the link wakes it in about 1 minute ("Yes, get this app back up!").
  Open it yourself shortly before the demo recording and before grading.
- Every `git push` to `main` redeploys the app automatically.
- Logs: in the app's **⋮ → Manage app** panel.

## Local run

```bash
.venv/bin/streamlit run streamlit_app/streamlit_app.py
# offline, with local files instead of the Hub:
IMOLARA_ONNX=models/onnx/model_int8.onnx IMOLARA_TOKENIZER=models/deploy_afriberta_mixed3/tokenizer.json \
IMOLARA_TFIDF=models/tfidf_mixed3.joblib .venv/bin/streamlit run streamlit_app/streamlit_app.py
```
