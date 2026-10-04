"""Publish the deployment models to the Hugging Face Hub and (re)deploy the Gradio Space.

  python scripts/deploy_hf.py model    # upload AfriBERTa weights + tfidf_mixed3.joblib + model card
  python scripts/deploy_hf.py space    # create/update the ZeroGPU Space with app/ and src/data.py
  python scripts/deploy_hf.py all

Requires `hf auth login` with a write token. Inputs:
  models/deploy_afriberta_mixed3/   (copied from the Kaggle run: results/kaggle/deploy_afriberta_mixed3/outputs/seed42/model)
  models/tfidf_mixed3.joblib        (python -m src.export_tfidf)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "Hassanadelani1/imolara-afriberta-mixed3"
SPACE_REPO = "Hassanadelani1/imolara"
MODEL_DIR = ROOT / "models" / "deploy_afriberta_mixed3"
TFIDF = ROOT / "models" / "tfidf_mixed3.joblib"
KAGGLE_MODEL = ROOT / "results" / "kaggle" / "deploy_afriberta_mixed3" / "outputs" / "seed42" / "model"


def deploy_model(api: HfApi):
    if not MODEL_DIR.exists():
        shutil.copytree(KAGGLE_MODEL, MODEL_DIR)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=MODEL_REPO, folder_path=str(MODEL_DIR), commit_message="AfriBERTa-large mixed3 (seed 42)")
    api.upload_file(repo_id=MODEL_REPO, path_or_fileobj=str(TFIDF), path_in_repo="tfidf_mixed3.joblib",
                    commit_message="TF-IDF mixed3 baseline (scikit-learn 1.9.1)")
    api.upload_file(repo_id=MODEL_REPO, path_or_fileobj=str(ROOT / "app" / "MODEL_CARD.md"), path_in_repo="README.md",
                    commit_message="Model card")
    print(f"model: https://huggingface.co/{MODEL_REPO}")


def deploy_space(api: HfApi):
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    files = {ROOT / "app" / "app.py": "app.py", ROOT / "app" / "requirements.txt": "requirements.txt",
             ROOT / "app" / "README.md": "README.md", ROOT / "src" / "__init__.py": "src/__init__.py",
             ROOT / "src" / "data.py": "src/data.py"}
    for local, remote in files.items():
        api.upload_file(repo_id=SPACE_REPO, repo_type="space", path_or_fileobj=str(local), path_in_repo=remote,
                        commit_message=f"Update {remote}")
    print(f"space: https://huggingface.co/spaces/{SPACE_REPO}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", choices=["model", "space", "all"])
    a = p.parse_args()
    api = HfApi()
    if a.target in ("model", "all"):
        deploy_model(api)
    if a.target in ("space", "all"):
        deploy_space(api)


if __name__ == "__main__":
    main()
