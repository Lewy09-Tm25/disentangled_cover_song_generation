"""
Orthogonal penalty between independently parameterized cross-attention stacks.

We accumulate Frobenius-squared cross-grams between analogous projection matrices::

    Σ_d  ( || W_V^{(d)}_cᵀ · W_V^{(d)}_s ||_F² + || W_O^{(d)}_cᵀ · W_O^{(d)}_s ||_F² )

where superscript ``d`` traverses transformer decoder depths, subscripts denote Content vs Style
explicit cross-attention pathways (implemented in ``model.py``).

This aligns with the class-project objective: mechanically discourage redundant subspaces between
harmonic-structure readers and acoustic-style readers operating in parallel depths.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

from covers_training.model import DecoderBlock

logger = logging.getLogger(__name__)


def _frob_sq_cross(a_proj: nn.Linear, b_proj: nn.Linear) -> torch.Tensor:
    wa = a_proj.weight  # [out,in]

    wb = b_proj.weight

    gram = wa.T @ wb
    frob_sq = gram.pow(2).sum()
    return frob_sq


def orthogonal_penalty_decoder_stack(dec_blocks: Iterable[DecoderBlock], *, device: torch.device) -> torch.Tensor:
    penalty = torch.zeros((), dtype=torch.float32, device=device)

    for depth_idx, dblk in enumerate(dec_blocks):
        cc = dblk.cross_content
        cs = dblk.cross_style

        penalty = penalty + _frob_sq_cross(cc.v_proj, cs.v_proj)
        penalty = penalty + _frob_sq_cross(cc.out_proj, cs.out_proj)

        if not torch.isfinite(penalty):
            logger.error("[OrthPenalty] divergence at decoder depth idx=%s", depth_idx)
            raise FloatingPointError()

    return penalty


def orthogonal_penalty_explicit_cross_attn(dec_blocks, *, device: torch.device) -> torch.Tensor:

    """Alias preserved for readability in trainer logs."""

    return orthogonal_penalty_decoder_stack(dec_blocks, device=device)


def lambda_orthogonal_scheduler(global_step: int, *, total_steps: int, cfg) -> float:
    """
    Piece-wise schedule::
        [0 , z*T)      -> λ = 0
        [z*T, z*r*T) -> linear interp  -> λ_target
        else         -> λ_target

    * ``cfg.orth_zero_frac`` == z above.
    * ``cfg.orth_ramp_frac`` == additional fractional span *after zero-hold* for linear ascent.
      Example defaults (0.1, 0.1): ramp occupies [10%,20%].

    Arguments
    ---------
    global_step
        Counts **optimizer** updates (post gradient accumulation aggregation), NOT micro-batch iters.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    frac_time = global_step / float(total_steps)

    zero_hold = float(cfg.orth_zero_frac)
    ramp_frac = float(cfg.orth_ramp_frac)
    lam_tgt = float(cfg.lambda_orth_target)

    ramp_start = zero_hold

    ramp_end = min(1.0, zero_hold + ramp_frac)

    if frac_time <= ramp_start:
        return 0.0

    if ramp_end <= ramp_start + 1e-12:
        return lam_tgt

    if ramp_start < frac_time <= ramp_end:
        ramp_progress = (frac_time - ramp_start) / max(ramp_end - ramp_start, 1e-12)
        return float(ramp_progress * lam_tgt)

    return lam_tgt
