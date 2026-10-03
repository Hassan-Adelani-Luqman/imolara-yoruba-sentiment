"""Run one experiment on a Kaggle T4 GPU and download its results (see docs/KAGGLE_GUIDE.md).

Usage:
  python scripts/kaggle_run.py configs/e5a_afroxlmr_base.yaml --wait
  python scripts/kaggle_run.py configs/e2a_bilstm_fasttext.yaml --module src.rnn --seeds 42
  python scripts/kaggle_run.py configs/e5a_afroxlmr_base.yaml --final --wait      # Phase 6: adds test scores
  python scripts/kaggle_run.py configs/e5c_afroxlmr_large_lora.yaml --fetch-only --with-model

The code (src/, configs/, requirements-kaggle.txt) is packed into the kernel script itself, so
the kernel runs exactly the local working tree. Commit before launching so results are traceable.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATHS = ["src", "configs", "requirements-kaggle.txt"]
KAGGLE = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def kaggle(*args: str) -> str:
    result = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def git_commit() -> str:
    """Current commit hash, suffixed with '-dirty' if there are uncommitted changes to bundled files."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "no-commit"
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *BUNDLE_PATHS], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def bundle_code() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in BUNDLE_PATHS:
            tar.add(ROOT / name, arcname=name,
                    filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


def build_kernel(build_dir: Path, slug: str, user: str, module: str, config: str,
                 seeds: list[int], extra_args: list[str], commit: str):
    build_dir.mkdir(parents=True, exist_ok=True)
    script = (ROOT / "kaggle" / "run_template.py").read_text()
    for key, value in {"__MODULE__": module, "__CONFIG__": config, "__SEEDS__": repr(seeds),
                       "__EXTRA_ARGS__": repr(extra_args), "__GIT_COMMIT__": commit,
                       "__BUNDLE__": bundle_code()}.items():
        script = script.replace(key, value)
    (build_dir / "run.py").write_text(script)
    (build_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": slug,                    # Kaggle derives the slug from the title, so keep them identical
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,          # needed for HF Hub models, AfriSenti TSVs and pip
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }, indent=2))


def wait(kernel: str, poll: int = 60) -> str:
    while True:
        status = kaggle("kernels", "status", kernel).lower()
        print(time.strftime("%H:%M:%S"), status.strip(), flush=True)
        for state in ("complete", "error", "cancel"):
            if state in status:
                return state
        time.sleep(poll)


def fetch(kernel: str, dest: Path, with_model: bool):
    if dest.exists():
        shutil.rmtree(dest)  # a re-run replaces the previous download
    dest.mkdir(parents=True)
    args = ["kernels", "output", kernel, "-p", str(dest), "-o"]
    if not with_model:  # skip weights; metrics, predictions, logs only
        args += ["--file-pattern", r".*\.(json|csv|txt|log|png)$"]
    kaggle(*args)
    print(f"Outputs in {dest.relative_to(ROOT)} (kernel log: {kernel.split('/')[1]}.log)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--module", default="src.train_transformer")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--final", action="store_true", help="pass --final (score the test split; Phase 6 only)")
    p.add_argument("--wait", action="store_true", help="poll until finished, then download and collect")
    p.add_argument("--fetch-only", action="store_true", help="only download outputs of a finished run")
    p.add_argument("--with-model", action="store_true", help="also download model weights")
    a = p.parse_args()

    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise SystemExit("Set KAGGLE_USERNAME (see docs/KAGGLE_GUIDE.md)")
    exp = Path(a.config).stem + ("_final" if a.final else "")
    slug = "imolara-" + exp.replace("_", "-")
    kernel = f"{user}/{slug}"
    dest = ROOT / "results" / "kaggle" / exp

    if not a.fetch_only:
        commit = git_commit()
        if commit.endswith("-dirty") or commit == "no-commit":
            print(f"warning: launching from uncommitted code ({commit})", file=sys.stderr)
        build_dir = ROOT / "kaggle" / "_build" / exp
        build_kernel(build_dir, slug, user, a.module, Path(a.config).resolve().relative_to(ROOT).as_posix(), a.seeds,
                     ["--final"] if a.final else [], commit)
        print(kaggle("kernels", "push", "-p", str(build_dir)).strip())
        print(f"Live log: https://www.kaggle.com/code/{kernel}")
        if not a.wait:
            return
        time.sleep(30)  # give Kaggle time to register the new version before polling
        if wait(kernel) != "complete":
            fetch(kernel, dest, with_model=False)  # fetch the log to debug
            raise SystemExit(f"Run failed; see {dest.relative_to(ROOT)}/{slug}.log")
    fetch(kernel, dest, a.with_model)
    subprocess.run([sys.executable, "-m", "src.evaluate", "collect", str(dest)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
