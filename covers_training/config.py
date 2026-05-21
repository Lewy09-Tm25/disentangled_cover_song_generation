"""
Central hyper-parameters and filesystem defaults.

These values mirror the finalized project specification:
- Lightweight Transformer on a T4 (Colab) with gradient accumulation for effective batch 32.
- Discrete EnCodec RVQ targets (multiple codebooks) with averaged cross-entropy.
- Orthogonal regularization warmup on W_V / W_O between Content vs Style cross-attention stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TrainConfig:
    # --- Data layout (offline extraction) ---
    # Nominal chorus length produced by DTW preprocessing (matches dtw_chorus.DURATION_SEC).
    clip_seconds: float = 30.0

    # Training crop length in acoustic token time (paired with harmonic features).
    train_crop_seconds: float = 5.0

    # Inference / evaluation context priming: overlap is token-domain history only (no waveform cross-fade).
    inference_step_seconds: float = 2.5

    # --- Audio / Chromagram bookkeeping (offline script uses these) ---
    chroma_sr: int = 22_050
    chroma_hop: int = 4096

    # --- Model ---
    d_model: int = 512
    n_heads_self: int = 8  # Decoder self-attention head count (split is only for cross-attention).
    n_heads_content_ca: int = 4
    n_heads_style_ca: int = 4
    n_layers_encoder_content: int = 4  # Stacks on top of harmonic encoder input projection.
    n_layers_decoder: int = 6
    dim_ff: int = 2048

    dropout: float = 0.1

    # --- Optimization ---
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_lr_frac: float = 0.05  # Linear LR warmup proportion of optimization steps after orth ramp stabilizes CE.
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05

    # Physical micro-batch vs effective batch size (effective = batch * accumulation).
    batch_size: int = 4
    grad_accum_steps: int = 8

    amp_enabled: bool = True

    # --- Orthogonal loss schedule (lambda_total for L_orthogonal) ---
    # 1) Hold 0 for first orth_zero_frac steps.
    # 2) Linear ramp orth_ramp_frac -> lambda_orth_target.
    # 3) Hold lambda_orth_target thereafter.
    orth_zero_frac: float = 0.10
    orth_ramp_frac: float = 0.10
    lambda_orth_target: float = 0.1

    # --- Training mechanics ---
    num_epochs: int = 80  # Oversized intentionally; overnight runs should early-stop manually if needed.

    checkpoint_every_optimizer_steps: int = 250
    keep_last_ckpt_name: str = "checkpoint_last.pt"
    keep_best_ckpt_name: str = "checkpoint_best_val.pt"

    # --- Reproducibility ---
    seed: int = 42

    # --- Inference / demos ---
    n_demo_pairs: int = 8


def chroma_hz(cfg: TrainConfig) -> float:
    """Chromagram frames per second for librosa CQT extraction used offline."""
    return cfg.chroma_sr / float(cfg.chroma_hop)


def encoder_frames_for_seconds(cfg: TrainConfig, clip_enc_frames: int) -> Tuple[int, int]:
    """Return integer frame counts representing train_crop_seconds and half-step contexts."""
    if clip_enc_frames <= 0:
        raise ValueError("clip_enc_frames must be positive.")

    Lt = max(1, int(round(cfg.train_crop_seconds / cfg.clip_seconds * clip_enc_frames)))
    half = Lt // 2
    half = max(1, half)
    return Lt, half


def split_work_ids(work_ids: list, cfg: TrainConfig) -> Tuple[set, set, set]:
    """ Deterministic 80/10/10 split on melody identity (`work_id`). """
    rng = sorted(set(work_ids))
    import random

    rnd = random.Random(cfg.seed)
    rnd.shuffle(rng)

    n = len(rng)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = set(rng[:n_train])
    val = set(rng[n_train : n_train + n_val])
    test = set(rng[n_train + n_val :])
    return train, val, test
