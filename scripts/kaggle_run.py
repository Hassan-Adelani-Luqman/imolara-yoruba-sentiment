"""Run one experiment on a Kaggle T4 GPU and download its results (see docs/KAGGLE_GUIDE.md).

Usage:
  python scripts/kaggle_run.py configs/e5a_afroxlmr_base.yaml --wait
  python scripts/kaggle_run.py configs/e2a_bilstm_fasttext.yaml --module src.rnn --seeds 42
  python scripts/kaggle_run.py configs/e5a_afroxlmr_base.yaml --final --wait      # Phase 6: adds test scores
  python scripts/kaggle_run.py configs/e5c_afroxlmr_large_lora.yaml --fetch-only --with-model
  python scripts/kaggle_run.py configs/e2b_bilstm_fasttext.yaml --module src.rnn --script --fetch-only   # legacy run

Two kernel formats:
  notebook (default): a readable Kaggle notebook that clones the public GitHub repo at the current
      commit (which must be committed and pushed). The executed notebook, with Kaggle's outputs, is saved
      to notebooks/kaggle/<exp>.ipynb. See scripts/kaggle_notebook.py.
  --script (legacy, used up to the first Phase 4 runs): src/, configs/ and requirements-kaggle.txt are packed
      into the kernel script itself.
Notebook kernels are named imolara-<exp>-nb, script kernels imolara-<exp>.
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


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def repo_clone_url() -> str:
    """https clone URL of the origin remote (works for both https and ssh remotes)."""
    url = git("remote", "get-url", "origin")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    return url if url.endswith(".git") else url + ".git"


def ensure_pushed() -> str:
    """Full commit hash of HEAD, after checking the bundled paths are committed and HEAD is on GitHub
    (pushing if needed), because notebook kernels clone the repo at this commit."""
    if git("status", "--porcelain", "--", *BUNDLE_PATHS, "scripts"):
        raise SystemExit("Commit your changes first: notebook kernels run the committed code from GitHub.")
    sha = git("rev-parse", "HEAD")
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT, check=False)
    if not git("branch", "-r", "--contains", sha):
        result = subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        if result.returncode != 0:
            raise SystemExit(f"HEAD {sha[:7]} is not on GitHub and pushing failed:\n{result.stderr}")
    return sha


def build_kernel(build_dir: Path, slug: str, user: str, module: str, config: str,
                 seeds: list[int], extra_args: list[str], commit: str, datasets: list[str],
                 notebook: bool = False, exp: str = ""):
    build_dir.mkdir(parents=True, exist_ok=True)
    if notebook:
        from kaggle_notebook import build_notebook  # scripts/ is on sys.path when run as a script
        import nbformat
        code_file = f"{slug}.ipynb"
        nbformat.write(build_notebook(exp, config, ROOT / config, module, seeds, extra_args, repo_clone_url(),
                                      commit, uses_embeddings=bool(datasets)), build_dir / code_file)
    else:
        code_file = "run.py"
        script = (ROOT / "kaggle" / "run_template.py").read_text()
        for key, value in {"__MODULE__": module, "__CONFIG__": config, "__SEEDS__": repr(seeds),
                           "__EXTRA_ARGS__": repr(extra_args), "__GIT_COMMIT__": commit,
                           "__BUNDLE__": bundle_code()}.items():
            script = script.replace(key, value)
        (build_dir / code_file).write_text(script)
    (build_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": slug,                    # Kaggle derives the slug from the title, so keep them identical
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook" if notebook else "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,          # needed for HF Hub models, AfriSenti TSVs and pip
        "dataset_sources": datasets, "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }, indent=2))


def push(build_dir: Path, retry_every: int = 120, max_wait_h: float = 6) -> str:
    """Push a kernel version. The CLI exits 0 even when the push is refused, so check its output. When
    both free-tier GPU slots are busy ("Maximum batch GPU session count of 2 reached"), wait and retry."""
    deadline = time.time() + max_wait_h * 3600
    while True:
        out = kaggle("kernels", "push", "-p", str(build_dir)).strip()
        if "successfully pushed" in out.lower():
            return out
        if "session count" in out.lower() and time.time() < deadline:
            print(time.strftime("%H:%M:%S"), "GPU slots busy; retrying in", retry_every, "s", flush=True)
            time.sleep(retry_every)
            continue
        raise SystemExit(f"kernel push failed: {out}")


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


def fetch_notebook(kernel: str, exp: str) -> Path | None:
    """Save the executed notebook (with Kaggle's outputs) to notebooks/kaggle/<exp>.ipynb."""
    tmp = ROOT / "kaggle" / "_build" / "_pull" / exp
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kaggle("kernels", "pull", kernel, "-p", str(tmp))
    found = sorted(tmp.glob("*.ipynb"))
    if not found:
        print("warning: no notebook returned by `kaggle kernels pull`", file=sys.stderr)
        return None
    dest = ROOT / "notebooks" / "kaggle" / f"{exp}.ipynb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(found[0], dest)
    print(f"Executed notebook in {dest.relative_to(ROOT)}")
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--module", default="src.train_transformer")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--final", action="store_true", help="pass --final (score the test split; Phase 6 only)")
    p.add_argument("--wait", action="store_true", help="poll until finished, then download and collect")
    p.add_argument("--fetch-only", action="store_true", help="only download outputs of a finished run")
    p.add_argument("--with-model", action="store_true", help="also download model weights")
    p.add_argument("--script", action="store_true", help="legacy packed-script kernel instead of a notebook")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Kaggle datasets to attach (default: <user>/imolara-embeddings for --module src.rnn)")
    a = p.parse_args()

    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        raise SystemExit("Set KAGGLE_USERNAME (see docs/KAGGLE_GUIDE.md)")
    exp = Path(a.config).stem + ("_final" if a.final else "")
    notebook = not a.script
    slug = "imolara-" + exp.replace("_", "-") + ("-nb" if notebook else "")
    kernel = f"{user}/{slug}"
    dest = ROOT / "results" / "kaggle" / exp

    datasets = a.datasets if a.datasets is not None else (
        [f"{user}/imolara-embeddings"] if a.module == "src.rnn" else [])
    if not a.fetch_only:
        if notebook:
            commit = ensure_pushed()
        else:
            commit = git_commit()
            if commit.endswith("-dirty") or commit == "no-commit":
                print(f"warning: launching from uncommitted code ({commit})", file=sys.stderr)
        build_dir = ROOT / "kaggle" / "_build" / exp
        build_kernel(build_dir, slug, user, a.module, Path(a.config).resolve().relative_to(ROOT).as_posix(), a.seeds,
                     ["--final"] if a.final else [], commit, datasets, notebook=notebook, exp=exp)
        print(push(build_dir))
        print(f"Live log: https://www.kaggle.com/code/{kernel}")
        if not a.wait:
            return
        time.sleep(30)  # give Kaggle time to register the new version before polling
        if wait(kernel) != "complete":
            fetch(kernel, dest, with_model=False)  # fetch the log to debug
            raise SystemExit(f"Run failed; see {dest.relative_to(ROOT)}/{slug}.log")
    fetch(kernel, dest, a.with_model)
    if notebook:
        fetch_notebook(kernel, exp)
    subprocess.run([sys.executable, "-m", "src.evaluate", "collect", str(dest)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
