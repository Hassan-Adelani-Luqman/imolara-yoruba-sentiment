"""Upload the static embeddings to a private Kaggle dataset so RNN kernels use the exact local files.

  python scripts/kaggle_dataset.py            # create <user>/imolara-embeddings, or add a new version

Files: cc.yo.300.vec.gz (fastText, CC BY-SA 3.0) and our Word2Vec (w2v_yo_300.kv + .vectors.npy,
trained by `python -m src.embeddings word2vec` on Yoruba Wikipedia CC BY-SA 4.0 + AfriSenti train).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "embeddings"
FILES = ["cc.yo.300.vec.gz", "w2v_yo_300.kv", "w2v_yo_300.kv.vectors.npy"]
KAGGLE = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def main():
    user = os.environ.get("KAGGLE_USERNAME") or sys.exit("Set KAGGLE_USERNAME")
    stage = ROOT / "kaggle" / "_build" / "embeddings_dataset"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name in FILES:
        os.link(SRC / name, stage / name)      # hard links: no copy of the 150 MB
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "imolara-embeddings", "id": f"{user}/imolara-embeddings",
        "licenses": [{"name": "CC-BY-SA-4.0"}]}, indent=2))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{user}/imolara-embeddings"],
                            capture_output=True, text=True).returncode == 0
    cmd = ["datasets", "version", "-p", str(stage), "-m", "update embeddings"] if exists else \
          ["datasets", "create", "-p", str(stage)]
    result = subprocess.run([KAGGLE, *cmd], capture_output=True, text=True)
    print(result.stdout, result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
