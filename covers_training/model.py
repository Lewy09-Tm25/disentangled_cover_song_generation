"""
Core sequence-to-sequence translator:
    Chromagram (anchor) ──► harmonic encoder ──► memory A
    MERT pooled vector ──► style projection ──► memory B (single-slot global context)
    EnCodec discrete tokens ──► decoder (dual cross-attention per depth)

IMPORTANT
---------
We purposely avoid packing Content + Style cues into monolithic embeddings so that orthogonal
constraints can manipulate distinct parameter tensors without fragile hook surgery.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from covers_training.config import TrainConfig

logger = logging.getLogger(__name__)


def _masked_softmax_attn(
    q: torch.Tensor,  # [B,H,Lq,Dk]
    k: torch.Tensor,  # [B,H,Lk,Dk]
    v: torch.Tensor,  # [B,H,Lk,Dv]
    key_padding_mask: Optional[torch.Tensor],  # [B,Lk] True==PAD (ignore keys)
    attn_dropout: nn.Dropout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classic scaled dot-product attention with rectangular memory."""
    scaling = q.shape[-1] ** -0.5

    attn_scores = torch.einsum("bhqd,bhkd->bhqk", q * scaling, k)  # [B,H,Lq,Lk]

    if key_padding_mask is not None:
        # attn_scores+: set padded keys massively negative BEFORE softmax for numerical stability.
        mask = key_padding_mask[:, None, None, :]  # [B,1,1,Lk]
        attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

    attn_probs = attn_scores.softmax(dim=-1)
    attn_probs = attn_dropout(attn_probs)

    out = torch.einsum("bhqk,bhkd->bhqd", attn_probs, v)  # [B,H,Lq,Dv_head]
    return out, attn_probs


class ExplicitCrossAttention(nn.Module):
    """
    Multi-head attention with cleanly exposed ``value_proj`` and ``out_proj`` nn.Linear ops.

    This is deliberately *not* a thin wrapper around ``nn.MultiheadAttention`` — that module flattens
    QKV weights in ways that obstruct the explicit cross-subspace regularizer from the blueprint.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("`d_model` must be divisible by `num_heads`.")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # [B,Lq,D]
        kv_memory: torch.Tensor,  # [B,Lk,D]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, lq, _ = query.shape
        _, lk, _ = kv_memory.shape

        q = self.q_proj(query).view(bsz, lq, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,Lq,Dh]
        k = self.k_proj(kv_memory).view(bsz, lk, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_proj(kv_memory).view(bsz, lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out, _dbg_probs = _masked_softmax_attn(q, k, v_t, key_padding_mask, self.attn_dropout)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, lq, self.d_model)

        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out)
        return attn_out


class SinusoidalPositionalEncoding(nn.Module):
    """Standard transformer sinusoid table (additive, non-learned baseline)."""

    def __init__(self, dim: int, max_len: int = 8192, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout))

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / float(dim)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]

        self.register_buffer("_pe_fixed", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        if seq > self._pe_fixed.shape[1]:
            raise ValueError(f"Sequence length={seq} exceeds PE max.")

        emb = self._pe_fixed[:, :seq, :].detach()
        return self.dropout(x + emb.to(dtype=x.dtype, device=x.device))


class DecoderBlock(nn.Module):
    """
    Canonical ordering (per finalized spec):

        LN → Self MHSA → residual →
        LN → Content Cross MHSA → residual →
        LN → Style  Cross MHSA → residual →
        LN → Feed-Forward         → residual
    """

    def __init__(
        self,
        d_model: int,
        *,
        heads_self: int,
        heads_content: int,
        heads_style: int,
        ff_dim: int,
        dropout: float,
        layer_dropout: float,
    ) -> None:
        super().__init__()

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            heads_self,
            dropout=layer_dropout,
            batch_first=True,
        )

        self.norm_cc = nn.LayerNorm(d_model)
        self.cross_content = ExplicitCrossAttention(d_model, heads_content, dropout=dropout)

        self.norm_cs = nn.LayerNorm(d_model)
        self.cross_style = ExplicitCrossAttention(d_model, heads_style, dropout=dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        self.drop_path = nn.Dropout(dropout)  # Residual stochastic depth substitute (cheap dropout residue).

    # Public handles for explicit regularizers -----------------------------
    @property
    def content_attn(self) -> ExplicitCrossAttention:
        return self.cross_content

    @property
    def style_attn(self) -> ExplicitCrossAttention:
        return self.cross_style


class CoverTranslator(nn.Module):
    def __init__(self, *, cfg: TrainConfig, num_codebooks: int, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)

        # --- Harmonized content embedding stack ---
        self.harm_proj = nn.Linear(12, cfg.d_model, bias=False)
        self.pos_harmonic = SinusoidalPositionalEncoding(cfg.d_model, max_len=16_384, dropout=cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            nhead=8,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.harm_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers_encoder_content)

        self.style_in = nn.Linear(1024, cfg.d_model, bias=False)
        nn.init.normal_(self.style_in.weight, std=0.02)

        # Token embedding backbone (summed embeddings — cheap + stable for averaged CE).
        self.codebook_embeds = nn.ModuleList(nn.Embedding(self.vocab_size, cfg.d_model) for _ in range(self.num_codebooks))
        for emb in self.codebook_embeds:
            nn.init.normal_(emb.weight, std=0.02)

        # Learnable BOS composites (one bias per quantized RVQ codeword family).
        self.bos_embeddings = nn.Parameter(torch.zeros(self.num_codebooks, cfg.d_model))
        nn.init.normal_(self.bos_embeddings, std=0.02)

        self.pos_decoding = SinusoidalPositionalEncoding(cfg.d_model, max_len=16_384, dropout=cfg.dropout)

        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(
                cfg.d_model,
                heads_self=cfg.n_heads_self,
                heads_content=cfg.n_heads_content_ca,
                heads_style=cfg.n_heads_style_ca,
                ff_dim=cfg.dim_ff,
                dropout=cfg.dropout,
                layer_dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers_decoder)
        )

        # Per residual-quantizer readout probes.
        self.logit_heads = nn.ModuleList(nn.Linear(cfg.d_model, self.vocab_size, bias=False) for _ in range(self.num_codebooks))

        nn.init.uniform_(self.harm_proj.weight, -1.0 / math.sqrt(12), 1.0 / math.sqrt(12))

    # ---------------------------------------------------------------------

    def _encode_harmonics(self, harmonic_batch: torch.Tensor, pad_mask_bt: Optional[torch.Tensor]) -> torch.Tensor:
        """
        harmonic_batch shape: [B, Lc, 12]
        pad_mask_bt: [B,Lc], True=PADDED (mirrors pytorch transformer convention)
        """
        x = self.harm_proj(harmonic_batch)
        x = self.pos_harmonic(x)
        return self.harm_encoder(x, src_key_padding_mask=pad_mask_bt)

    def embed_autoreg_inputs(self, targets_bkt: torch.Tensor) -> torch.Tensor:
        """Shift-right embedding for causal decoder core (teacher forcing path)."""

        # targets_bkt: [B,K,T]
        bsz, k_cb, tgt_len = targets_bkt.shape
        assert k_cb == self.num_codebooks

        summed = torch.zeros(bsz, tgt_len, self.cfg.d_model, device=targets_bkt.device, dtype=self.harm_proj.weight.dtype)

        # t=0 uses dedicated BOS-like composite embeddings.
        bos_sum = torch.sum(self.bos_embeddings, dim=0, keepdim=True).expand(bsz, -1)  # [B,D]

        summed[:, 0, :] += bos_sum

        if tgt_len <= 1:
            return self.pos_decoding(summed)

        for k_idx in range(self.num_codebooks):
            # Teacher-forcing chain: timestep ``t >= 1`` consumes discrete RVQ selections from ``t-1``.
            tok_emb_prev = self.codebook_embeds[k_idx](targets_bkt[:, k_idx, :-1])
            summed[:, 1:, :] = summed[:, 1:, :] + tok_emb_prev

        return self.pos_decoding(summed)

    # ---------------------------------------------------------------------

    def forward(
        self,
        *,
        content: torch.Tensor,  # [B,Lc,12]
        style: torch.Tensor,  # [B,1024]
        tgt_tokens_bt: torch.Tensor,  # [B,K,Lt_u]
        content_pad_mask: Optional[torch.Tensor] = None,  # [B,Lc] True==PAD (same convention as pytorch)
        token_pad_mask: Optional[torch.Tensor] = None,  # [B,Lt] True==PAD
        ablate_content_xattn: bool = False,
        ablate_style_xattn: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns logits: [B, Lt_eff, K, V] excluding positions which are padded,
        callers still flatten with masks as needed during CE.

        Auxiliary dict carries optional diagnostics hooks (probability maps, etc.).
        """

        harmonic_mem = self._encode_harmonics(content, content_pad_mask)  # [B,Lc,D]

        style_mem = self.style_in(style).unsqueeze(1)  # singleton temporal axis

        tgt_len = tgt_tokens_bt.shape[-1]

        causal_square = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_tokens_bt.device)  # [T,T]

        x = self.embed_autoreg_inputs(tgt_tokens_bt)

        attn_debug: dict[str, torch.Tensor] = {}

        for blk in self.decoder_blocks:
            # ---- Self Attention ----
            n1 = blk.norm_self(x)
            sx, attn_w = blk.self_attn(
                n1,
                n1,
                n1,
                attn_mask=causal_square,
                key_padding_mask=token_pad_mask,
                average_attn_weights=False,
                need_weights=True,
            )
            x = x + sx

            # ---- Content pathway ----
            ncc = blk.norm_cc(x)

            xh = blk.cross_content(
                query=ncc,
                kv_memory=harmonic_mem,
                key_padding_mask=content_pad_mask,
            )
            if ablate_content_xattn:
                xh = xh * 0.0

            x = x + xh

            # ---- Style pathway ----
            nss = blk.norm_cs(x)

            xh = blk.cross_style(
                nss,
                kv_memory=style_mem,
                key_padding_mask=None,
            )

            if ablate_style_xattn:
                xh = xh * 0.0

            x = x + xh

            # Feed-forward.
            xf = blk.norm_ff(x)

            xf = blk.ffn(xf)

            x = x + xf

            # NOTE: attn_w shape [B,num_heads,Q,K] optionally attach first block only — logging optional.
            if not attn_debug and attn_w is not None:
                attn_debug["decoder_self_attn_l0_sample"] = attn_w.detach()

        # Readout logits per quantized RVQ level.
        head_logits = [head(x).unsqueeze(dim=2) for head in self.logit_heads]  # shapes [B,T,1,V]

        logits = torch.cat(head_logits, dim=2)  # [B,T,K,V]

        return logits, attn_debug
