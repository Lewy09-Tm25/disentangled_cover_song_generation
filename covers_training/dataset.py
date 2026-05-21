"""
PyTorch Dataset for precomputed tensors.

IMPORTANT
---------
We deliberately **do not** touch raw WAV files here — only ``.npy`` / ``.pt`` artifacts.

Alignment rule (per project spec)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1) Choose a fractional start ``alpha ∈ [0,1]`` uniformly (or deterministic for sanity checks).
2) Map that fractional progress into encoder-token frames and chromagram frames using the
   lengths of **this** clip. This preserves the relative wall-clock correspondence even when
   librosa chroma grids and EnCodec token grids disagree in absolute frame-rate.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from covers_training.config import TrainConfig, split_work_ids

logger = logging.getLogger(__name__)

SplitLiteral = Literal["train", "val", "test"]


def _load_manifest(manifest_csv: str) -> pd.DataFrame:
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_csv!r}. "
            "Ensure `FINAL_TENSOR_DATASET.zip` is extracted and paired with tensor_extraction_checkpoint.csv."
        )
    df = pd.read_csv(manifest_csv)

    needed = {"work_id", "shift_amount", "content_npy", "style_npy", "target_pt"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"manifest_csv missing columns: {sorted(missing)}")
    return df


class CoverSongTensorDataset(torch.utils.data.Dataset):
    """
    One row = one (anchor chroma crop, frozen global cover style vector, cover token crop).

    The style vector always corresponds to the **full ~30 s** cover chorus used during offline MERT pooling.
    Cropping only touches chroma + discrete tokens — never recomputes MERT.
    """

    def __init__(
        self,
        tensor_dir: str,
        manifest_csv: str,
        split: SplitLiteral,
        cfg: TrainConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.tensor_dir = tensor_dir
        self.cfg = cfg
        self.transform = transform

        if not os.path.isdir(tensor_dir):
            raise FileNotFoundError(f"tensor_dir not found: {tensor_dir}")

        df = _load_manifest(manifest_csv)
        ids = sorted({int(w) for w in df["work_id"].tolist()})

        train_ids, val_ids, test_ids = split_work_ids(ids, cfg)

        if split == "train":
            chosen = train_ids
        elif split == "val":
            chosen = val_ids
        elif split == "test":
            chosen = test_ids
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df_rows = df[df["work_id"].isin(chosen)].reset_index(drop=True)
        logger.info("[Dataset:%s] rows=%d unique_work_ids=%d", split, len(self.df_rows), len(chosen))

        if len(self.df_rows) == 0:
            raise RuntimeError(f"No rows after split `{split}` — check CSV / unzip layout.")

        # Introspection cache (populated lazily once we see tensors on disk).
        self._canonical_shape: Dict[str, int] | None = None

    # ---------------------------------------------------------------------

    def _probe_shapes_once(self, sample_target: torch.Tensor, sample_content: np.ndarray) -> Dict[str, int]:
        """Infer codebook cardinality and vocab size from tensors + EnCodec conventions."""
        if sample_target.ndim != 2:
            raise ValueError(f"Unexpected target_pt rank {sample_target.ndim}; expected [K, T].")

        num_codebooks, enc_T = sample_target.shape
        # cardinality: trust max index + safety margin --- EnCodec bins are contiguous 0..bins-1
        max_idx = int(torch.max(sample_target).item())
        # Common default is 1024; still keep adaptive headroom (+1 inclusive).
        vocab_size = max(1024, max_idx + 1)

        chroma_bins, chroma_T = sample_content.shape
        if chroma_bins != 12:
            logger.warning("[Dataset] Unexpected chromagram bins=%s (expected 12). Continuing anyway.", chroma_bins)

        return {
            "num_codebooks": int(num_codebooks),
            "vocab_size": int(vocab_size),
            "enc_full_T": int(enc_T),
            "chroma_full_T": int(chroma_T),
        }

    # ---------------------------------------------------------------------

    def get_or_build_shape_metadata(self, force_row_idx: Optional[int] = None) -> Dict[str, int]:
        """
        Public helper for trainers: resolves dynamic tensor topology once using a real `.pt`.

        Passing ``force_row_idx`` makes this deterministic during debugging Colab setups.
        """
        if self._canonical_shape is not None:
            return self._canonical_shape

        idx = int(force_row_idx) if force_row_idx is not None else 0

        row = self.df_rows.iloc[idx]
        c_path = os.path.join(self.tensor_dir, row["content_npy"])
        t_path = os.path.join(self.tensor_dir, row["target_pt"])

        chroma_np = np.load(c_path).astype(np.float32, copy=False)
        tgt = torch.load(t_path, map_location="cpu")

        if not torch.is_tensor(tgt):
            raise TypeError(f"target_pt must be torch.Tensor after load(); got {type(tgt)!r}")

        meta = self._probe_shapes_once(tgt, chroma_np)
        self._canonical_shape = meta
        logger.info("[Dataset] Resolved tensor topology: %s", meta)
        return meta

    # ---------------------------------------------------------------------

    def _crop_pair(
        self,
        *,
        chroma: np.ndarray,  # [12, Tc]
        tokens: torch.Tensor,  # [K, Te]
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[str, int]]:
        """
        Returns cropped chromagram (still [12,Lc]), tokens [K,Lt],
        plus debug indices for reproducibility bookkeeping.
        """
        K, Te = tokens.shape
        chroma_np_T = chroma.shape[1]

        # Full clip lengths inferred from tensors on disk for this exemplar (robust vs minor padding).
        enc_T_full = Te

        Lt = max(1, int(round(self.cfg.train_crop_seconds / self.cfg.clip_seconds * enc_T_full)))
        Lc = max(1, int(round(self.cfg.train_crop_seconds / self.cfg.clip_seconds * chroma_np_T)))

        max_start_enc = max(0, enc_T_full - Lt)
        max_start_chr = max(0, chroma_np_T - Lc)

        if max_start_enc == 0:
            start_enc = 0
        else:
            start_enc = int(rng.randint(0, max_start_enc))

        alpha = (start_enc / max_start_enc) if max_start_enc > 0 else 0.0
        start_chr = int(round(alpha * max_start_chr))
        start_chr = int(min(max(start_chr, 0), max_start_chr))

        enc_slice = tokens[:, start_enc : start_enc + Lt]
        chr_slice = chroma[:, start_chr : start_chr + Lc]

        dbg = {
            "Lt": Lt,
            "Lc": Lc,
            "start_enc": start_enc,
            "start_chr": start_chr,
            "Te_full": enc_T_full,
            "Tc_full": chroma_np_T,
        }
        assert enc_slice.shape[1] == Lt
        assert chr_slice.shape[1] == Lc
        return chr_slice.astype(np.float32, copy=False), enc_slice.long(), dbg

    def __len__(self) -> int:
        return len(self.df_rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        row = self.df_rows.iloc[int(index)]

        work_id = int(row["work_id"])
        shift_amt = int(row["shift_amount"])

        c_path = os.path.join(self.tensor_dir, row["content_npy"])
        s_path = os.path.join(self.tensor_dir, row["style_npy"])
        t_path = os.path.join(self.tensor_dir, row["target_pt"])

        if not os.path.isfile(c_path) or not os.path.isfile(s_path) or not os.path.isfile(t_path):
            raise FileNotFoundError(
                f"Missing tensors for work_id={work_id}: {c_path=}, {s_path=}, {t_path=}.\n"
                "Did you unzip `FINAL_TENSOR_DATASET.zip` into `--tensor-dir`?"
            )

        chroma_np = np.load(c_path).astype(np.float32, copy=False)  # [12, T] typically
        style_np = np.load(s_path).astype(np.float32, copy=False).reshape(-1)
        tgt = torch.load(t_path, map_location="cpu")

        # RandomState requires seed in [0, 2**32 - 1]. XOR can exceed that or go negative when shift is -1.
        seed_u32 = (int(self.cfg.seed) ^ int(work_id) ^ int(shift_amt) ^ int(index)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed=seed_u32)

        chroma_crop, tok_crop, _dbg = self._crop_pair(chroma=chroma_np, tokens=tgt, rng=rng)

        # Canonical batch layout -------------------------------------------------
        content = torch.from_numpy(chroma_crop).transpose(0, 1).contiguous()  # [Lc,12]
        style = torch.from_numpy(style_np)  # [1024]

        targets = tok_crop.long()  # [K, Lt]

        sample = {"work_id": work_id, "shift": shift_amt, "content": content, "style": style, "targets": targets}

        if self.transform:
            sample = self.transform(sample)
        return sample


def padded_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Minimal padding-aware collator (handles rare ±1 length mismatches safely across clips).

    Padded chroma/time positions are flagged with booleans suitable for masking / ignoring,
    although the harmonic encoder ignores padding implicitly if all-zero — we additionally return masks.
    """
    bs = len(batch)
    lc_max = max(int(b["content"].shape[0]) for b in batch)
    lt_max = max(int(b["targets"].shape[1]) for b in batch)
    k = int(batch[0]["targets"].shape[0])

    content_pad = torch.zeros(bs, lc_max, 12, dtype=torch.float32)
    tgt_pad = torch.zeros(bs, k, lt_max, dtype=torch.long)
    style = torch.stack([b["style"] for b in batch], dim=0).float()

    content_mask = torch.zeros(bs, lc_max, dtype=torch.bool)
    token_mask = torch.zeros(bs, lt_max, dtype=torch.bool)

    wids = []
    shifts = []

    for i, b in enumerate(batch):
        lc_i = int(b["content"].shape[0])
        lt_i = int(b["targets"].shape[1])
        content_pad[i, :lc_i] = b["content"]
        tgt_pad[i, :, :lt_i] = b["targets"]
        content_mask[i, lc_i:] = True
        token_mask[i, lt_i:] = True
        wids.append(int(b["work_id"]))
        shifts.append(int(b["shift"]))

    out = {
        "content": content_pad,
        "style": style,
        "targets": tgt_pad,
        "content_pad_mask": content_mask,
        "token_pad_mask": token_mask,
        "work_id": torch.tensor(wids, dtype=torch.long),
        "shift": torch.tensor(shifts, dtype=torch.long),
    }
    return out
