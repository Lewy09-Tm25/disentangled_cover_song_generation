"""
Inference / ablation tooling (token-space overlap → assemble RVQ timeline → ONE EnCodec decode).

Rules (from project spec):
- Overlap is **only** discrete token history fed back into the decoder (no waveform cross-fade).
- Concatenate shards on the codec time axis and call EnCodec `.decode(...)` exactly once.

Colab usage example::

    python covers_training/infer_eval.py ^
      --checkpoint training_runs/exp1/checkpoint_best_val.pt ^
      --tensor-dir tensor_dataset ^
      --manifest-csv tensor_extraction_checkpoint.csv ^
      --split test ^
      --out-dir infer_out_run1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # Minimal environments

    def tqdm(it, **_k):  # type: ignore
        return it


from covers_training.config import TrainConfig, encoder_frames_for_seconds, split_work_ids

from covers_training.model import CoverTranslator

from covers_training.time_align import inference_wall_start_schedule
from covers_training.time_align import slice_pair_at_wall_start


logger = logging.getLogger("cover_inference")


def configure_logger(out_dir: Path) -> None:
    fh = logging.FileHandler(out_dir / "infer_eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[fh, logging.StreamHandler(sys.stdout)])


def lev_distance_int_list(a: Sequence[int], b: Sequence[int]) -> int:
    """DP Levenshtein on Python int lists."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = np.arange(lb + 1, dtype=np.int32)
    for i in range(1, la + 1):
        cur = np.zeros(lb + 1, dtype=np.int32)
        cur[0] = i
        ai = int(a[i - 1])
        for j in range(1, lb + 1):
            cost = int(ai != int(b[j - 1]))
            cur[j] = int(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return int(prev[lb])


def _lev_row(ref_row: torch.Tensor, hyp_row: torch.Tensor) -> int:
    return lev_distance_int_list(ref_row.detach().cpu().tolist(), hyp_row.detach().cpu().tolist())


def averaged_codebook_lev(ref_kb_te: torch.Tensor, hyp_kb_te: torch.Tensor) -> float:
    k_codes = ref_kb_te.shape[0]
    vals = [_lev_row(ref_kb_te[k_ix], hyp_kb_te[k_ix]) for k_ix in range(k_codes)]
    return float(np.mean(vals))


def clamp_master(master_kb: torch.Tensor, te_tgt: int) -> torch.Tensor:
    _, tl = master_kb.shape
    if tl == te_tgt:
        return master_kb
    if tl > te_tgt:
        logger.warning("[length] cropping synth timeline %s -> %s", tl, te_tgt)
        return master_kb[:, :te_tgt].contiguous()
    pad_n = te_tgt - tl
    logger.warning("[length] pad synth tail by %s frames", pad_n)
    pad = master_kb[:, -1:].repeat(1, pad_n)
    return torch.cat([master_kb, pad], dim=-1)


@torch.no_grad()
def greedy_extend_tokens(
    model: CoverTranslator,
    *,
    content_b1_lc: torch.Tensor,
    style_b1: torch.Tensor,
    hist_tokens_kp: torch.Tensor,
    gen_frames: int,
    device: torch.device,
    ablate_content: bool,
    ablate_style: bool,
) -> torch.Tensor:
    """
    Hist tail `[K,P]` (may be empty), returns **only** appended `[K,G]`.
    """
    kk, pref = hist_tokens_kp.shape
    if kk != model.num_codebooks:
        raise ValueError(f"Hist K mismatch: hist={hist_tokens_kp.shape}")

    rolling = hist_tokens_kp.to(device=device, dtype=torch.long)
    tails: List[torch.Tensor] = []

    for _ in range(int(gen_frames)):
        if rolling.numel() == 0:
            tiled = torch.zeros(1, model.num_codebooks, 1, dtype=torch.long, device=device)
        else:
            mirror = rolling[:, -1:].clone()
            tiled = torch.cat([rolling, mirror], dim=-1).unsqueeze(0)

        logits, _ = model(
            content=content_b1_lc,
            style=style_b1,
            tgt_tokens_bt=tiled,
            content_pad_mask=None,
            token_pad_mask=None,
            ablate_content_xattn=ablate_content,
            ablate_style_xattn=ablate_style,
        )
        nx = logits[:, -1, :, :].squeeze(0).argmax(dim=-1)
        tails.append(nx.unsqueeze(-1))
        rolling = torch.cat([rolling, nx.unsqueeze(-1)], dim=-1)

    return torch.cat(tails, dim=-1)


@torch.no_grad()
def discrete_chunked_rollout(
    *,
    model: CoverTranslator,
    cfg: TrainConfig,
    chroma_np_xtc: np.ndarray,
    style_np: np.ndarray,
    te_reference_len: int,
    Lt_full: int,
    Lt_half: int,
    device: torch.device,
    grouped_ablate: str,
) -> torch.Tensor:
    """
    Applies wall anchors `[0, 2.5, 5, ...]` consistent with inference_wall_start_schedule.
    grouped_ablate: "none" | "content" | "style"
    """
    ab_c = grouped_ablate == "content"
    ab_s = grouped_ablate == "style"

    style_b = torch.from_numpy(style_np.astype(np.float32)).to(device=device).unsqueeze(0)

    anchors = inference_wall_start_schedule(cfg)
    dummy_kb = torch.zeros(model.num_codebooks, int(te_reference_len), dtype=torch.long)

    master = torch.empty(model.num_codebooks, 0, dtype=torch.long, device=device)

    for i_anchor, anchor_sec in enumerate(tqdm(list(anchors), desc="anchors")):
        content_lc12, _, _meta = slice_pair_at_wall_start(
            chroma_np_xtc,
            dummy_kb.clone(),
            cfg=cfg,
            wall_start_sec=float(anchor_sec),
            window_sec=float(cfg.train_crop_seconds),
        )
        content_b1 = content_lc12.unsqueeze(0).to(device=device)

        if i_anchor == 0:
            hist_kb = torch.empty(model.num_codebooks, 0, dtype=torch.long, device=device)
            gen_n = Lt_full
        else:
            if master.shape[-1] < Lt_half:
                raise RuntimeError(
                    "Token history shorter than overlap length — chunked priming precondition failed."
                    f"(have={master.shape[-1]}, need={Lt_half})"
                )
            hist_kb = master[:, -Lt_half:].contiguous()
            gen_n = Lt_half

        tail_kb = greedy_extend_tokens(
            model,
            content_b1_lc=content_b1,
            style_b1=style_b,
            hist_tokens_kp=hist_kb,
            gen_frames=gen_n,
            device=device,
            ablate_content=ab_c,
            ablate_style=ab_s,
        )
        master = torch.cat([master, tail_kb], dim=-1)

    return clamp_master(master, int(te_reference_len))


@torch.no_grad()
def decode_once_encodec(shared_codec: torch.nn.Module, master_kb_te: torch.Tensor, *, codec_dev: torch.device) -> torch.Tensor:
    """Decode `[K,T]` → waveform tensor `[channels, samples]` (frozen EnCodec, 24 kHz / 6 kbps)."""
    codes_b = master_kb_te.unsqueeze(0).long().to(codec_dev)

    wav = shared_codec.decode([(codes_b, None)])

    wav = wav.detach().squeeze(0).float().cpu()

    return wav


def instantiate_shared_encodec(codec_dev: torch.device) -> torch.nn.Module:

    """Single heavy load — reuse for every qualitative export to stay Colab-efficient."""

    from encodec import EncodecModel as _CodecFactory

    mdl = _CodecFactory.encodec_model_24khz()

    mdl.set_target_bandwidth(6.0)

    mdl.to(codec_dev)

    mdl.eval()

    return mdl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tensor-dir", required=True)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--out-dir", default="./infer_demo_out")
    ap.add_argument("--max-songs", type=int, default=8)
    ap.add_argument("--device", default="cuda")

    cli = ap.parse_args()

    out_dir = Path(cli.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logger(out_dir)

    device = torch.device(str(cli.device))
    codec_device = device

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("[device] CUDA requested but unavailable → using CPU weights + decode.")
        device = torch.device("cpu")
        codec_device = torch.device("cpu")

    cfg = TrainConfig()

    bundle = torch.load(Path(cli.checkpoint).expanduser(), map_location=device)
    model = CoverTranslator(
        cfg=cfg,
        num_codebooks=int(bundle["num_codebooks"]),
        vocab_size=int(bundle["vocab_cap"]),
    )
    model.load_state_dict(bundle["model_sd"])
    model.to(device)
    model.eval()

    df_all = pd.read_csv(cli.manifest_csv)
    uniq_ids = sorted({int(w) for w in df_all["work_id"].tolist()})
    tr, va, te = split_work_ids(uniq_ids, cfg)
    allowed = {"train": tr, "val": va, "test": te}[cli.split]

    preferred = df_all[(df_all["work_id"].isin(allowed)) & (df_all["shift_amount"].astype(int) == 0)]
    if len(preferred) == 0:
        logger.warning("No manifest rows at shift_amount==0; falling back to unfiltered anchors.")
        preferred = df_all[df_all["work_id"].isin(allowed)]

    df_sub = preferred.drop_duplicates(subset=["work_id"], keep="first")
    df_pick = df_sub.head(max(1, int(cli.max_songs))).reset_index(drop=True)

    codec_mdl = instantiate_shared_encodec(codec_device)

    metrics_json: Dict[str, Dict[str, float]] = {}

    for _, row in df_pick.iterrows():
        wid = int(row["work_id"])
        shift = int(row["shift_amount"])

        chroma_np = np.load(Path(cli.tensor_dir) / str(row["content_npy"]))
        style_np = np.load(Path(cli.tensor_dir) / str(row["style_npy"])).astype(np.float32).reshape(-1)
        tgt_ref = torch.load(Path(cli.tensor_dir) / str(row["target_pt"]), map_location="cpu").long()

        Te = int(tgt_ref.shape[-1])
        Lt_full, Lt_half = encoder_frames_for_seconds(cfg, clip_enc_frames=Te)

        logger.info("[song] wid=%s shift=%s Te=%s Lt_full=%s Lt_half=%s", wid, shift, Te, Lt_full, Lt_half)

        def run_mode(tag: str, abmode: str) -> torch.Tensor:
            return discrete_chunked_rollout(
                model=model,
                cfg=cfg,
                chroma_np_xtc=chroma_np,
                style_np=style_np,
                te_reference_len=Te,
                Lt_full=Lt_full,
                Lt_half=Lt_half,
                device=device,
                grouped_ablate=abmode,
            )

        base = run_mode("base", "none")
        no_style = run_mode("mask_style_heads", "style")
        no_cnt = run_mode("mask_content_heads", "content")

        key = f"wid_{wid}_shift_{shift}"

        b_cpu = base.cpu()
        s_cpu = no_style.cpu()
        c_cpu = no_cnt.cpu()

        metrics_json[key] = {
            "lev_mean_vs_gt_base": averaged_codebook_lev(tgt_ref, b_cpu),
            "lev_mean_vs_gt_mask_style_all": averaged_codebook_lev(tgt_ref, s_cpu),
            "lev_mean_vs_gt_mask_content_all": averaged_codebook_lev(tgt_ref, c_cpu),
            "lev_mean_baseline_vs_mask_style_heads": averaged_codebook_lev(b_cpu, s_cpu),
            "lev_mean_baseline_vs_mask_content_heads": averaged_codebook_lev(b_cpu, c_cpu),
        }

        logger.info("[%s] %s", key, metrics_json[key])

        # Persist numpy waveforms decoded once from RVQ timelines (bring into Colab notebook for listening).
        wav_base = decode_once_encodec(codec_mdl, base, codec_dev=codec_device)
        wav_no_style = decode_once_encodec(codec_mdl, no_style, codec_dev=codec_device)
        wav_no_cnt = decode_once_encodec(codec_mdl, no_cnt, codec_dev=codec_device)

        np.save(out_dir / f"{key}_baseline_mono.npy", wav_base.numpy().reshape(-1).astype(np.float32))
        np.save(out_dir / f"{key}_ablate_style_mono.npy", wav_no_style.numpy().reshape(-1).astype(np.float32))
        np.save(out_dir / f"{key}_ablate_content_mono.npy", wav_no_cnt.numpy().reshape(-1).astype(np.float32))

    summary_path = out_dir / "ablation_metrics.json"
    summary_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
