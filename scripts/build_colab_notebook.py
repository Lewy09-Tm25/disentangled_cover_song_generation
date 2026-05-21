"""One-off helper: assemble Cover_Song_Disentangled_Colab.ipynb from covers_training/*.py."""

from __future__ import annotations

import json
from pathlib import Path


def cell_md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def cell_code(body: str) -> dict:
    if not body.endswith("\n"):
        body += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [body],
    }


def wf_cell(rel_posix: str, body: str) -> dict:
    body = "# %%writefile " + rel_posix + "\n" + body
    if not body.endswith("\n"):
        body += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [body],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ct = root / "covers_training"

    files_ordered = [
        ct / "__init__.py",
        ct / "config.py",
        ct / "time_align.py",
        ct / "dataset.py",
        ct / "model.py",
        ct / "orthogonal.py",
        ct / "train.py",
        ct / "infer_eval.py",
    ]

    cells: list[dict] = []

    cells.append(
        cell_md(
            """# Cover song modeling — Colab runnable notebook

Mirrors **`covers_training/`** from your project. Execute **top → bottom**:

1. **Install** libraries (GPU runtime recommended).
2. **Write modules** (`%%writefile`) recreate `covers_training/*.py`.
3. **Configure** tensor directory + manifest paths (Drive or `/content`).
4. **Training** overnight on T4.
5. **Inference + ablations**: token-overlap priming → **one** EnCodec decode each.

Unpack `FINAL_TENSOR_DATASET.zip` and place **`tensor_extraction_checkpoint.csv`** next to `.npy` / `.pt` files."""
        )
    )

    cells.append(
        cell_code(
            """%%capture
# Core runtime deps on Colab
%pip install -q encodec transformers pandas matplotlib tqdm soundfile"""
        )
    )

    cells.append(
        cell_code(
            """import os
import sys

WORKDIR = "/content/cover_training_workspace"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)
os.makedirs(os.path.join(WORKDIR, "covers_training"), exist_ok=True)

if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

print("Using:", os.getcwd())
print("Package dir ready: ./covers_training/")"""
        )
    )

    cells.append(cell_md("## Emit `covers_training/` package"))

    for p in files_ordered:
        rel_posix = "covers_training/" + p.name
        body = p.read_text(encoding="utf-8")
        cells.append(wf_cell(rel_posix, body))

    cells.append(
        cell_md(
            """## Paths + Drive zip extraction (no shell commands)

Set your Drive paths below. The cell mounts Drive, copies/extracts the zip once, and points training to the extracted folder."""
        )
    )

    cells.append(
        cell_code(
            """try:
    from google.colab import drive

    _HAVE_COLAB = True
except Exception:
    _HAVE_COLAB = False

import os
import zipfile
from pathlib import Path

# Colab multiprocessing dataloaders are flaky — keep workers at 0.
os.environ.setdefault("COVER_TRAIN_NUM_WORKERS", "0")

MOUNT_DRIVE = True

if _HAVE_COLAB and MOUNT_DRIVE:
    drive.mount("/content/drive")

# <<< EDIT these 2 paths for your Drive layout >>>
ZIP_ON_DRIVE = "/content/drive/MyDrive/FINAL_TENSOR_DATASET.zip"
MANIFEST_ON_DRIVE = "/content/drive/MyDrive/tensor_extraction_checkpoint.csv"

# Local runtime paths inside Colab workspace:
LOCAL_ZIP = os.path.join(WORKDIR, "FINAL_TENSOR_DATASET.zip")
TENSOR_DIR = os.path.join(WORKDIR, "tensor_dataset")
MANIFEST_CSV = os.path.join(WORKDIR, "tensor_extraction_checkpoint.csv")

OUTPUT_TRAIN_DIR = os.path.join(WORKDIR, "runs", "colab_exp1")
OUTPUT_INFER_DIR = os.path.join(WORKDIR, "infer_demo_out")

os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_INFER_DIR, exist_ok=True)

# Copy manifest from Drive each run (tiny file).
assert os.path.isfile(MANIFEST_ON_DRIVE), f"Missing manifest on Drive: {MANIFEST_ON_DRIVE}"
Path(MANIFEST_CSV).write_bytes(Path(MANIFEST_ON_DRIVE).read_bytes())

# Copy + extract zip only when tensors are absent.
if not os.path.isdir(TENSOR_DIR) or len(os.listdir(TENSOR_DIR)) == 0:
    assert os.path.isfile(ZIP_ON_DRIVE), f"Missing zip on Drive: {ZIP_ON_DRIVE}"
    print("Copying zip from Drive to local runtime...")
    Path(LOCAL_ZIP).write_bytes(Path(ZIP_ON_DRIVE).read_bytes())
    print("Extracting zip (first time only)...")
    with zipfile.ZipFile(LOCAL_ZIP, "r") as zf:
        zf.extractall(TENSOR_DIR)
    print("Extraction complete.")
else:
    print("Tensor directory already present; skipping extraction.")

print("MANIFEST:", MANIFEST_CSV)
print("TENSOR dir exists:", os.path.isdir(TENSOR_DIR))"""
        )
    )

    cells.append(
        cell_md(
            """## Train

Physical batch **4** × accumulation **8** ⇒ effective batch **32**. Checkpoints land in **`OUTPUT_TRAIN_DIR`**."""
        )
    )

    cells.append(
        cell_code(
            """import sys
import covers_training.train as train_pkg

_RESUME = None

argv = ["train.py", "--tensor-dir", TENSOR_DIR, "--manifest-csv", MANIFEST_CSV, "--output-dir", OUTPUT_TRAIN_DIR]
if _RESUME:
    argv += ["--resume", _RESUME]

_old_argv = sys.argv[:]
try:
    # Python-only launch (no `!python` command needed).
    sys.argv = argv
    train_pkg.main()
finally:
    sys.argv = _old_argv"""
        )
    )

    cells.append(
        cell_md(
            """## Inference / grouped ablations

Writes **`.npy` mono waveform** excerpts + **`ablation_metrics.json`**."""
        )
    )

    cells.append(
        cell_code(
            """import os
import sys
import covers_training.infer_eval as infer_pkg

CKPT = os.path.join(OUTPUT_TRAIN_DIR, "checkpoint_best_val.pt")
assert os.path.isfile(CKPT), "Train first or adjust CKPT path."

sys.argv = [
    "infer_eval.py",
    "--checkpoint", CKPT,
    "--tensor-dir", TENSOR_DIR,
    "--manifest-csv", MANIFEST_CSV,
    "--split", "test",
    "--max-songs", "8",
    "--out-dir", OUTPUT_INFER_DIR,
    "--device", "cuda",
]

infer_pkg.main()"""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }

    out_nb = root / "Cover_Song_Disentangled_Colab.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out_nb} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
