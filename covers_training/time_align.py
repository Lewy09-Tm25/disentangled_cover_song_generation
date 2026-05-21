"""
Wall-clock aligned slicing between chromagram frames and EnCodec token frames.

The offline corpus stores full ~``clip_seconds`` extracts. Chromagram grids (librosa) and discrete
codec frames (Encodec RVQ) use different native sampling rates — we reuse the proportional mapping that
training crops use so inference windows stay acoustically coherent.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from covers_training.config import TrainConfig


def window_frames_counts(*, tc_full: int, te_full: int, cfg: TrainConfig, window_sec: float) -> Tuple[int, int]:
    """Return (Lc, Lt) harmonic/token frame spans for ``window_sec`` seconds within a clip of length cfg.clip_seconds."""

    Lt = max(1, int(round(window_sec / cfg.clip_seconds * te_full)))

    lc = max(1, int(round(window_sec / cfg.clip_seconds * tc_full)))

    return lc, Lt


def slice_pair_at_wall_start(
    chroma_twelve_by_tc: np.ndarray,
    tokens_k_by_te: torch.Tensor,
    *,
    cfg: TrainConfig,
    wall_start_sec: float,
    window_sec: float,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns:
      * content_crop      float tensor [Lc,12] for the harmonic encoder batch dimension.
      * (optional bookkeeping dict with integer indices).

    Raises if slice would be empty due to malformed tensors — caller must ensure clip lengths sane.
    """

    if chroma_twelve_by_tc.ndim != 2 or chroma_twelve_by_tc.shape[0] != 12:

        raise ValueError(f"Chroma ndarray must be shaped [12, Tc]; got {chroma_twelve_by_tc.shape}.")

    tc_full = chroma_twelve_by_tc.shape[1]

    if tokens_k_by_te.ndim != 2:

        raise ValueError(f"Tokens must be [K, Te]; got {tokens_k_by_te.shape}.")

    te_full = int(tokens_k_by_te.shape[-1])

    lc, Lt = window_frames_counts(tc_full=tc_full, te_full=te_full, cfg=cfg, window_sec=window_sec)

    max_start_tc = max(0, tc_full - lc)

    max_start_te = max(0, te_full - Lt)

    anchor_sec = float(wall_start_sec)

    anchor_sec_clamped = float(min(max(anchor_sec, 0.0), max(cfg.clip_seconds - window_sec, 0.0)))

    start_tc = int(round(anchor_sec_clamped / cfg.clip_seconds * float(tc_full)))

    start_tc = min(max(start_tc, 0), max_start_tc)

    start_te = int(round(anchor_sec_clamped / cfg.clip_seconds * float(te_full)))

    start_te = min(max(start_te, 0), max_start_te)

    chroma_crop = chroma_twelve_by_tc[:, start_tc : start_tc + lc].astype(np.float32)

    tokens_crop = tokens_k_by_te[:, start_te : start_te + Lt].clone()

    content = torch.from_numpy(chroma_crop).transpose(0, 1).contiguous()

    meta = {"start_tc": start_tc, "start_te": start_te, "Lc": lc, "Lt": Lt, "Tc_full": tc_full, "Te_full": te_full}

    return content, tokens_crop, meta


def inference_wall_start_schedule(cfg: TrainConfig) -> list[float]:
    """
    Sliding 5-second analysis windows anchored at ``0``, ``inference_step_seconds``, ``2*inference_step_seconds``, ...

    Example (30 s clip, 5 s windows, 2.5 s step): anchors at 0, 2.5, 5, …, 25 s.
    Overlap supplies **token-domain** prefix context only — no waveform cross-fading.
    """

    starts = []

    s = 0.0

    while s + cfg.train_crop_seconds <= cfg.clip_seconds + 1e-9:

        starts.append(float(round(s, 6)))

        s += float(cfg.inference_step_seconds)

    return starts
