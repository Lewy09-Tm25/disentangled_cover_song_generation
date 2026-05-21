# Disentangled Cover Song Generation

A research project for generating cover songs by explicitly disentangling the **invariant** melodic content from the **variant** acoustic style, allowing the system to synthesize audio that preserves the harmonic identity of a source song while adopting the timbre and genre texture of a target cover.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Data Pipeline](#data-pipeline)
- [Model](#model)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Notes on Audio Files](#notes-on-audio-files)

---

## Overview

The core research problem is that cover songs share melodic identity with their originals but differ substantially in acoustic presentation — genre, instrumentation, vocal timbre, rhythm feel, and production style. This project trains a neural model to separate these two axes:

- **Content** — the pitch-class melody, captured as a chromagram, which is invariant across covers of the same song.
- **Style** — the acoustic texture and genre identity, captured as a MERT embedding, which is variant and specific to each recording.

At inference time, the model generates EnCodec tokens conditioned on the content of one song and the style of another, which are then decoded back to audio.

---

## Architecture

The full training workflow is illustrated in the diagram below.

![Architecture Diagram](Cover_Song_Generation_Workflow_Architecture.png)

The system has four main stages:

### 1. Data Pipeline & Input Tensors

Three tensors are constructed for each training example:

| Tensor | Shape | Description |
|---|---|---|
| Content | `[T_chroma, 12]` | Folded pitch classes (chromagram) extracted from the anchor song. Melody-invariant. |
| Style | `[1024]` | MERT hidden-state vector, mean-pooled over a 30-second window. Captures timbre and genre. |
| Target | `[codebooks, T_enc]` | EnCodec RVQ tokens from the target cover recording. Used as the prediction target. |

Anchor and target segments are aligned using Dynamic Time Warping (DTW) on chorus regions, then randomly cropped to aligned 5-second windows at training time.

### 2. Core Architecture

**Harmonic Encoder** (4 transformer blocks, $d_{model} = 512$): Processes the low-dimensional chromagram into a contextualized hidden representation.

**Autoregressive Decoder** (6 transformer blocks): Generates EnCodec tokens autoregressively. Each block contains:
- Self-Attention
- **Content Cross-Attention** (4 heads) — attends to the Harmonic Encoder output
- **Style Cross-Attention** (4 heads) — attends to the MERT style vector
- Feed-Forward Network (FFN, hidden size 2048)

The key novel mechanism is the **separated cross-attention pathway**: content and style queries are routed through entirely independent attention heads, preventing the two information sources from mixing within each block.

### 3. Custom Objective Function

**Generative Baseline — Average Cross-Entropy:** Uniform average CE loss across 8 parallel RVQ codebooks, with label smoothing of 0.05.

**Disentanglement Constraint — Orthogonal Loss:** Penalizes the Frobenius norm of the correlation between the Value ($W_V$) and Output ($W_O$) projection matrices of the Content heads versus the Style heads. This forces the two attention pathways to map to orthogonal subspaces.

**Lambda ($\lambda$) Warmup Schedule:** The orthogonal penalty weight is held at 0.0 for the first 10% of training steps (to allow basic sequence alignment), then linearly ramped to 0.1 to aggressively enforce disentanglement.

$$\lambda(t) = \begin{cases} 0 & t < 0.1 \cdot T_{total} \\ 0.1 \cdot \frac{t - 0.1 \cdot T_{total}}{0.9 \cdot T_{total}} & \text{otherwise} \end{cases}$$

### 4. Inference & Evaluation

**Chunked Autoregression with Context Priming:** Generation proceeds in 5-second chunks. Each chunk uses 2.5 seconds of prompt/history context and generates 2.5 seconds of new tokens, ensuring temporal continuity across the full track.

**Grouped Head Ablation (Acoustic Texture Collapse Test):** Style heads are masked while content heads remain active (and vice versa). When style heads are masked, acoustic texture should collapse while melody remains intact — this validates that the two pathways have truly learned disjoint representations.

**Token Divergence — Levenshtein Distance:** Measures the edit distance between token sequences produced by the baseline model and by each ablated variant, averaged across all 8 RVQ codebooks. A large divergence when style heads are masked confirms that style heads are causally responsible for acoustic texture.

---

## Repository Structure

```
disentangled_cover_song_generation/
│
├── covers_training/           # Core training package
│   ├── model.py               # Harmonic Encoder + Disentangled Decoder
│   ├── dataset.py             # PyTorch Dataset with aligned 5s temporal cropping
│   ├── train.py               # Training loop, AMP, checkpointing, evaluation
│   └── ...
│
├── scripts/                   # Utility and build scripts
│   └── build_colab_notebook.py
│
├── analyze_run_figures/       # Evaluation outputs and analysis plots
│
├── download_songs.py          # Sourcing and downloading song pairs
├── dtw_chorus.py              # DTW-based chorus extraction and alignment
├── pitch_shifting.py          # Locked-pair ±1 semitone augmentation
├── input_tensorization.py     # Offline feature extraction (chroma, MERT, EnCodec)
│
├── augmented_pairs.csv        # Manifest after pitch-shift augmentation
├── downloaded_pairs.csv       # Manifest of downloaded anchor/cover pairs
├── dtw_aligned_pairs.csv      # Manifest after DTW alignment
├── tensor_extraction_checkpoint.csv  # Checkpoint for incremental tensor extraction
│
└── check_downloaded_songs.ipynb  # Notebook for data integrity verification
```

---

## Data Pipeline

The preprocessing pipeline runs offline in sequence before training:

**Step 1 — Download (`download_songs.py`):** Downloads anchor and cover song pairs, producing `downloaded_pairs.csv`.

**Step 2 — Align & Extract Choruses (`dtw_chorus.py`):** Applies DTW on chroma features to align anchor/cover pairs temporally, extracting matched ~30-second chorus segments. Produces `dtw_aligned_pairs.csv`.

**Step 3 — Augment (`pitch_shifting.py`):** Applies locked-pair pitch shifts of {−1, 0, +1} semitones to each aligned pair, tripling the dataset size while preserving relative melodic content. Produces `augmented_pairs.csv`.

**Step 4 — Tensorize (`input_tensorization.py`):** For each augmented pair, extracts and saves:
- Chromagram (content) as `.npy`
- MERT embedding (style) as `.npy`
- EnCodec RVQ tokens (target) as `.pt`

---

## Model

The model configuration used in this project:

| Hyperparameter | Value |
|---|---|
| Model dimension ($d_{model}$) | 512 |
| Encoder blocks | 4 |
| Decoder blocks | 6 |
| Attention heads (total per block) | 8 |
| Content cross-attention heads | 4 |
| Style cross-attention heads | 4 |
| FFN hidden size | 2048 |
| RVQ codebooks | 8 |
| Vocab size per codebook | 1024 |

The architecture is intentionally kept lightweight for training on a single GPU.

---

## Training

**Hardware:** Training was run on an NVIDIA A40 (48GB VRAM).

**Key training decisions:**

- Mixed precision (AMP) enabled when CUDA is available; automatically disabled on CPU.
- Aligned 5-second temporal crops sampled randomly at each training step (anchor and cover cropped at the same temporal offset to preserve alignment).
- Parallel classification loss across all 8 RVQ codebooks simultaneously.
- Label smoothing: 0.05.
- Orthogonal penalty applied to $W_V$ and $W_O$ of content vs. style cross-attention projections.
- Lambda warmup: 0.0 for the first 10% of steps → linear ramp to 0.1.

**Data splits:** Divided by `work_id` to prevent melody leakage between train, validation, and test sets.

---

## Inference & Evaluation

At inference time:

1. Provide a **content source** (any audio; chromagram is extracted).
2. Provide a **style source** (any audio; MERT embedding is extracted).
3. Run chunked autoregressive decoding using 2.5s context priming per 5s chunk.
4. Decode the generated EnCodec token sequence back to audio.

Evaluation metrics:

- **Levenshtein Distance** between baseline and ablated token sequences (averaged across codebooks).
- **Grouped Head Ablation** to verify acoustic texture collapse when style heads are masked, with melody remaining stable when content heads are masked.

---

## Requirements

Core dependencies:

```
torch
torchaudio
transformers         # MERT model
encodec              # EnCodec tokenizer/decoder
librosa              # Chromagram extraction
numpy
pandas
```

A full `requirements.txt` or Colab setup cell is included in the notebook generated by `scripts/build_colab_notebook.py`.

---

## Usage

The project includes a generated Colab notebook (`Cover_Song_Disentangled_Colab.ipynb`) that can be used for preprocessing and experimentation. Full training was run on the A40 GPU at the LUMaA Lab.

1. **Mount Google Drive** and set paths in the paths cell.
2. **Install dependencies** via the setup cell.
3. **Verify tensors** — confirm that `.npy` content/style files and `.pt` target files are present and the manifest CSV path resolves correctly.
4. **Run the Train cell** to start training. The cell calls `train_pkg.main()` via a Python function call; no shell commands are needed.
5. Monitor training logs for per-epoch CE loss and orthogonal penalty values.

> **Note:** AMP is automatically disabled if CUDA is not available.

---

## Notes on Audio Files

Raw audio files (`.mp3`/`.wav`) are **not included in this repository** due to file size constraints. The preprocessing pipeline (`download_songs.py` → `dtw_chorus.py` → `pitch_shifting.py` → `input_tensorization.py`) must be run to produce the tensor files before training. The CSV manifests and pre-extracted tensor checkpoints are included to allow resuming the pipeline from intermediate steps.
