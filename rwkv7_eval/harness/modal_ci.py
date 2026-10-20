"""CPU test runner on Modal for the RWKV-7 rework (profile hakureirm only).

Usage (from ~/workspace/rwkv7-work, with MODAL_PROFILE=hakureirm):
    modal run modal_ci.py --label baseline-rebased --pytest-args "tests/models/rwkv7 -x -q"

- Dependencies live in the image layer (CPU torch + the repo's [testing] extras).
- The working tree (src/ tests/ utils/ + config files) is mounted at run time, so code changes
  never rebuild the image. PYTHONPATH points at the mounted src/.
- The container exits when the function returns (no keep-warm). The log is returned and written
  locally to logs/<label>-<utc>.log together with the git sha of the mounted tree.
"""

import datetime
import os
import subprocess

import modal

REPO = os.path.expanduser("~/workspace/rwkv7-tf-rework")
HERE = os.path.dirname(os.path.abspath(__file__))
IGNORE = ["**/__pycache__", "**/*.pyc", "**/.pytest_cache", "**/.ruff_cache"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.14.1", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(os.path.join(HERE, "ci_requirements.txt"))
    .pip_install("onnx", "onnxscript", "onnxruntime")  # for the slow ONNX export tests
    .env({"PYTHONPATH": "/repo/src", "TRANSFORMERS_IS_CI": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    # the whole working tree minus git metadata and local agent links
    .add_local_dir(REPO, "/repo", ignore=IGNORE + [".git", ".git/**", ".claude", ".claude/**", ".agents", ".agents/**"])
)

app = modal.App("rwkv7-cpu-ci", image=image)


@app.function(cpu=4.0, memory=8192, timeout=3600)
def run(cmd: list[str], slow: bool = False) -> tuple[int, str]:
    import torch

    if slow:
        os.environ["RUN_SLOW"] = "1"

    header = f"$ {' '.join(cmd)}\nRUN_SLOW={os.environ.get('RUN_SLOW')} torch={torch.__version__} threads={torch.get_num_threads()}\n"
    p = subprocess.run(cmd, cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, header + p.stdout


@app.local_entrypoint()
def main(label: str, pytest_args: str = "", python_args: str = "", slow: bool = False):
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if python_args:
        cmd = ["python", *python_args.split()]
    else:
        cmd = ["python", "-m", "pytest", *pytest_args.split()]
    code, log = run.remote(cmd, slow)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    path = os.path.join(HERE, "logs", f"{label}-{stamp}.log")
    with open(path, "w") as f:
        f.write(f"label={label}\nrepo_sha={sha}\nworking_tree_dirty={bool(dirty)}\nexit={code}\n")
        if dirty:
            f.write("dirty_files:\n" + dirty + "\n")
        f.write("=" * 70 + "\n" + log)
    tail = "\n".join(log.rstrip().splitlines()[-25:])
    print(tail)
    print(f"\n[exit={code}] log: {path}")
