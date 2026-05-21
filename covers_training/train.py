"""
End-to-end training loop for CoverTranslator on precomputed tensors (Colab-friendly).

IMPORTANT
---------
* Do **not** rerun ``input_tensorization.py`` unless you intentionally refresh features.
* Intended launch pattern on Colab::
      python covers_training/train.py --tensor-dir /path/to/tensor_dataset \\
          --manifest-csv tensor_extraction_checkpoint.csv --output-dir ./runs/exp1

This script persists:
  * ``checkpoint_last.pt`` / ``checkpoint_best_val.pt``
  * ``metrics.jsonl`` (one JSON object per optimizer step — easy to reconstruct plots)
"""

from __future__ import annotations

import argparse

import json

import logging

import math

import os

import sys

from pathlib import Path

from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import torch

import torch.nn.functional as F

from torch.cuda.amp import GradScaler

from torch.cuda.amp import autocast

try:

    from tqdm.auto import tqdm

except ImportError:  # ultra-light fallback

    tqdm = None  # type: ignore

from covers_training.config import TrainConfig

from covers_training.dataset import CoverSongTensorDataset

from covers_training.dataset import padded_collate_fn

from covers_training.model import CoverTranslator

from covers_training.orthogonal import lambda_orthogonal_scheduler

from covers_training.orthogonal import orthogonal_penalty_decoder_stack


logger = logging.getLogger("cover_training")


def configure_logger(out_dir: Path) -> None:

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handlers = []

    fh = logging.FileHandler(out_dir / "train.log")

    fh.setFormatter(formatter)

    handlers.append(fh)

    sh = logging.StreamHandler(sys.stdout)

    sh.setFormatter(formatter)

    handlers.append(sh)

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    for h in handlers:

        root_logger.addHandler(h)

    root_logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:

    import random

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def averaged_rvq_cross_entropy(
    logits: torch.Tensor,

    targets: torch.Tensor,

    token_pad_mask: torch.Tensor,

    *,
    label_smoothing: float,
    vocab_bound: int,
) -> torch.Tensor:
    """

    logits:   [B, T, K, V]

    targets:  [B, K, T]

    pad mask: True==PAD timestep (matching batch collation convention).

    """

    b_sz, ti, k_codes, vocab = logits.shape

    if vocab_bound != vocab:

        raise ValueError(f"logit vocab axis ({vocab}) mismatches vocab_bound ({vocab_bound}).")

    eps = torch.finfo(torch.float32).tiny

    valid_mask_bt = (~token_pad_mask).to(dtype=logits.dtype)  # [B,T]

    ce_stack = []

    for k_ix in range(k_codes):

        log_k = logits[:, :, k_ix, :]  # [B,T,V]

        flat_logits = log_k.reshape(b_sz * ti, vocab)

        flat_tgt = targets[:, k_ix, :].reshape(b_sz * ti)

        flat_mask = (~token_pad_mask).reshape(b_sz * ti).float()

        ce_vec = F.cross_entropy(

            flat_logits,

            flat_tgt,

            reduction="none",

            label_smoothing=float(label_smoothing),

        )

        denom = torch.clamp(torch.sum((~token_pad_mask).float()), min=eps)

        ce_k_mean = torch.sum(ce_vec.reshape(b_sz, ti) * (~token_pad_mask).float()) / denom

        ce_stack.append(ce_k_mean)

    return torch.stack(ce_stack).mean(dim=0)


def _save_checkpoint_bundle(

    path: Path,

    *,

    model: torch.nn.Module,

    optimizer: torch.optim.Optimizer,

    scaler: GradScaler,

    global_step_opt: int,

    epoch_ix: int,

    cfg_obj: TrainConfig,

    vocab_bound: int,

    num_cb: int,
) -> None:

    blob = {

        "model_sd": model.state_dict(),

        "optim_sd": optimizer.state_dict(),

        "scaler_sd": scaler.state_dict(),

        "global_step_optimizer": global_step_opt,

        "epoch": epoch_ix,

        "cfg_dict": vars(cfg_obj) if not isinstance(cfg_obj, dict) else cfg_obj,

        "vocab_cap": vocab_bound,

        "num_codebooks": num_cb,

        "torch_version": torch.__version__,
    }

    torch.save(blob, path)

    logging.info("[Checkpoint] Wrote `%s`.", path)


def plot_metrics_jsonl(metrics_path: Path, out_png: Path) -> None:

    xs = []

    ce_vals = []

    orth_vals = []

    lam_vals = []

    totals = []

    with metrics_path.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            payload = json.loads(line)

            if payload.get("kind") != "optimizer_step_summary":

                continue

            xs.append(int(payload.get("global_optimizer_step", 0)))

            ce_vals.append(float(payload.get("train_ce_smooth", payload.get("train_ce_batch", math.nan))))

            orth_vals.append(float(payload.get("orth_penalty_scaled", math.nan)))

            lam_vals.append(float(payload.get("lambda_orth_eff", math.nan)))

            totals.append(float(payload.get("train_total_smooth", payload.get("train_total_batch", math.nan))))

    if len(xs) < 2:

        logging.warning("[Plot] Not enough rows to plot `%s`.", metrics_path)

        return

    fig, axs = plt.subplots(3, 1, figsize=(10.5, 8.25), constrained_layout=True)

    axs[0].plot(xs, totals, lw=2.25, label="smooth total loss proxy")

    axs[0].set_title("Combined training surrogate (moving average envelope)")

    axs[0].set_xlabel("Optimizer step")

    axs[0].grid(True, alpha=0.22)

    axs[1].plot(xs, ce_vals, lw=2.05, label="CE (masked RVQ-average)")

    axs[1].set_title("Multi-codebook cross-entropy (uniform average)")

    axs[1].set_xlabel("Optimizer step")

    axs[1].grid(True, alpha=0.22)

    ax_twin = axs[2]

    ln1 = ax_twin.plot(xs, orth_vals, color="darkgreen", lw=2.05, label="scaled orth penalty λ·L")[0]

    ax2 = ax_twin.twinx()

    ln2 = ax2.plot(xs, lam_vals, color="#c62828", lw=1.7, linestyle="--", label="λ schedule")[0]

    ax_twin.set_title("Orthogonal coupling — penalty vs schedule")

    ax_twin.grid(True, alpha=0.18)

    fig.legend([ln1, ln2], ["λ · Lorth", "λ scalar"], ncol=2, loc="lower center")

    fig.savefig(out_png, dpi=164)

    plt.close(fig)


def build_lr_lambda(total_opt_steps: int, cfg: TrainConfig):

    warmup_steps = int(math.ceil(total_opt_steps * float(cfg.warmup_lr_frac)))

    def lr_lambda(step: int) -> float:

        if warmup_steps <= 0:

            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1.0, float(total_opt_steps)))))

        if step < warmup_steps:

            return float(step) / float(max(1, warmup_steps))

        denom = float(max(1, total_opt_steps - warmup_steps))

        progress_rel = float(step - warmup_steps) / denom

        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress_rel)))

    return lr_lambda


def main() -> None:

    parser = argparse.ArgumentParser(description="Train CoverTranslator")

    parser.add_argument("--tensor-dir", required=True)

    parser.add_argument("--manifest-csv", required=True)

    parser.add_argument("--output-dir", default="./training_runs/exp_default")

    parser.add_argument("--resume", default=None)

    cli = parser.parse_args()

    cfg = TrainConfig()

    root_out = Path(cli.output_dir).resolve()

    root_out.mkdir(parents=True, exist_ok=True)

    configure_logger(root_out)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = bool(cfg.amp_enabled and device.type == "cuda")

    logging.info("[Init] Device=%s | AMP_requested=%s | AMP_active=%s", device.type, cfg.amp_enabled, use_amp)

    train_ds = CoverSongTensorDataset(

        cli.tensor_dir,

        cli.manifest_csv,

        split="train",

        cfg=cfg,

    )

    val_ds = CoverSongTensorDataset(

        cli.tensor_dir,

        cli.manifest_csv,

        split="val",

        cfg=cfg,

    )

    meta = train_ds.get_or_build_shape_metadata()

    num_cb = meta["num_codebooks"]

    vocab_cap = meta["vocab_size"]

    train_loader = torch.utils.data.DataLoader(

        train_ds,

        batch_size=cfg.batch_size,

        shuffle=True,

        drop_last=False,

        num_workers=int(os.environ.get("COVER_TRAIN_NUM_WORKERS", "0")),

        collate_fn=padded_collate_fn,

        persistent_workers=False,

        pin_memory=(device.type == "cuda"),

    )

    val_loader = torch.utils.data.DataLoader(

        val_ds,

        batch_size=cfg.batch_size,

        shuffle=False,

        drop_last=False,

        num_workers=0,

        collate_fn=padded_collate_fn,

        persistent_workers=False,

        pin_memory=(device.type == "cuda"),

    )

    model = CoverTranslator(cfg=cfg, num_codebooks=num_cb, vocab_size=vocab_cap).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = GradScaler(enabled=use_amp)

    epochs = int(cfg.num_epochs)

    batches_per_epoch = max(1, len(train_loader))

    opt_updates_per_epoch = math.ceil(batches_per_epoch / float(cfg.grad_accum_steps))

    total_opt_steps = max(1, opt_updates_per_epoch * epochs)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=build_lr_lambda(total_opt_steps, cfg))

    jsonl_metric_path = root_out / "metrics.jsonl"

    opt_step_global = 0

    momentum_ce = math.nan

    momentum_total = math.nan

    smoothing_beta = 0.98

    best_val_ce = math.inf

    micro_step_epochless = 0

    if cli.resume:

        chk = Path(cli.resume).expanduser()

        if chk.is_file():

            bundle = torch.load(chk, map_location=device)

            model.load_state_dict(bundle["model_sd"])

            optim.load_state_dict(bundle["optim_sd"])

            scaler.load_state_dict(bundle["scaler_sd"])

            opt_step_global = int(bundle.get("global_step_optimizer", 0))

            scheduler.last_epoch = opt_step_global

            logging.info("[Resume] Reloaded `%s` at optimizer-step=%s", chk, opt_step_global)

        else:

            logging.warning("[Resume] Path missing — cold start.`%s`", chk)

    optim.zero_grad(set_to_none=True)

    for epoch_ix in range(epochs):

        model.train()

        banner = f"[Epoch {epoch_ix+1}/{epochs}]"

        iterator = enumerate(train_loader, start=0)

        prog = None

        if tqdm is not None:

            prog = tqdm(iterator, total=batches_per_epoch, desc=f"{banner} train", dynamic_ncols=True)

            iterator_wrap = prog

        else:

            iterator_wrap = iterator

        for batch_step, bundle in iterator_wrap:

            bundle = {

                key: val.to(device) if isinstance(val, torch.Tensor) else val

                for key, val in bundle.items()

            }

            lam_eff = lambda_orthogonal_scheduler(opt_step_global, total_steps=int(total_opt_steps), cfg=cfg)

            micro_step_epochless += 1

            amp_ctx = autocast(enabled=use_amp)

            try:

                with amp_ctx:

                    logits, _attn_unused = model(

                        content=bundle["content"],

                        style=bundle["style"],

                        tgt_tokens_bt=bundle["targets"].long(),

                        content_pad_mask=bundle["content_pad_mask"],

                        token_pad_mask=bundle["token_pad_mask"],

                    )

                    ce_piece = averaged_rvq_cross_entropy(

                        logits,

                        bundle["targets"],

                        bundle["token_pad_mask"],

                        label_smoothing=cfg.label_smoothing,

                        vocab_bound=vocab_cap,

                    )

                with autocast(enabled=False):

                    orth_raw = orthogonal_penalty_decoder_stack(model.decoder_blocks, device=device)

                orth_term = lam_eff * orth_raw

                total_loss_piece = ce_piece.float() + orth_term.float()

            except FloatingPointError as exc:

                raise RuntimeError("Numerics diverged.") from exc

            shard_loss_scaled = total_loss_piece / float(cfg.grad_accum_steps)

            if scaler.is_enabled():

                scaler.scale(shard_loss_scaled).backward()

            else:

                shard_loss_scaled.backward()

            if micro_step_epochless % int(cfg.grad_accum_steps) == 0:

                if scaler.is_enabled():

                    scaler.unscale_(optim)

                model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if scaler.is_enabled():

                    scaler.step(optim)

                    scaler.update()

                else:

                    optim.step()

                scheduler.step()

                optim.zero_grad(set_to_none=True)

                opt_step_global += 1

                if math.isnan(momentum_ce):

                    momentum_ce = float(ce_piece.detach())

                    momentum_total = float(total_loss_piece.detach())

                else:

                    momentum_ce = smoothing_beta * momentum_ce + (1.0 - smoothing_beta) * float(ce_piece.detach())

                    momentum_total = smoothing_beta * momentum_total + (1.0 - smoothing_beta) * float(

                        total_loss_piece.detach()

                    )

                log_record: Dict[str, Any] = {

                    "kind": "optimizer_step_summary",

                    "global_optimizer_step": opt_step_global,

                    "epoch": epoch_ix,

                    "train_ce_batch": float(ce_piece.detach()),

                    "train_total_batch": float(total_loss_piece.detach()),

                    "lambda_orth_eff": lam_eff,

                    "orth_penalty_raw": float(orth_raw.detach()),

                    "orth_penalty_scaled": float(orth_term.detach()),

                    "train_ce_smooth": float(momentum_ce),

                    "train_total_smooth": float(momentum_total),

                    "lr": float(optim.param_groups[0]["lr"]),

                    "grad_norm_clip": cfg.max_grad_norm,

                    "grad_norm_after_clip": float(model_grad_norm.detach()),

                }

                jsonl_metric_path.parent.mkdir(parents=True, exist_ok=True)

                with jsonl_metric_path.open("a", encoding="utf-8") as f_jsonl:

                    f_jsonl.write(json.dumps(log_record))

                    f_jsonl.write("\n")

                ckpt_every = int(cfg.checkpoint_every_optimizer_steps)

                if ckpt_every > 0 and (opt_step_global % ckpt_every == 0):

                    chk_path = root_out / f"checkpoint_step_{opt_step_global:06d}.pt"

                    _save_checkpoint_bundle(

                        chk_path,

                        model=model,

                        optimizer=optim,

                        scaler=scaler,

                        global_step_opt=opt_step_global,

                        epoch_ix=epoch_ix,

                        cfg_obj=cfg,

                        vocab_bound=vocab_cap,

                        num_cb=num_cb,

                    )

                _save_checkpoint_bundle(

                    root_out / cfg.keep_last_ckpt_name,

                    model=model,

                    optimizer=optim,

                    scaler=scaler,

                    global_step_opt=opt_step_global,

                    epoch_ix=epoch_ix,

                    cfg_obj=cfg,

                    vocab_bound=vocab_cap,

                    num_cb=num_cb,

                )

                if tqdm is None and (batch_step % 25 == 0):

                    logging.info(

                        "%s step=%s | CE_ma~%.6f λ=%.6e orth_raw=%.6e",

                        banner,

                        opt_step_global,

                        momentum_ce,

                        lam_eff,

                        float(orth_raw.detach()),
                    )

        # --- VALIDATION ---
        model.eval()

        batch_val_ces: list[float] = []

        with torch.no_grad():

            for val_bundle_raw in val_loader:

                val_bundle = {

                    kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv

                    for kk, vv in val_bundle_raw.items()

                }

                with autocast(enabled=use_amp):

                    v_logits, _v_dbg = model(

                        content=val_bundle["content"],

                        style=val_bundle["style"],

                        tgt_tokens_bt=val_bundle["targets"].long(),

                        content_pad_mask=val_bundle["content_pad_mask"],

                        token_pad_mask=val_bundle["token_pad_mask"],

                    )

                vc = averaged_rvq_cross_entropy(

                    v_logits.float(),

                    val_bundle["targets"],

                    val_bundle["token_pad_mask"],

                    label_smoothing=0.0,

                    vocab_bound=vocab_cap,

                )

                batch_val_ces.append(float(vc.detach().cpu()))

        val_agg = float(np.mean(batch_val_ces)) if batch_val_ces else math.nan

        logging.info("[%s epoch=%s VAL] mean_batch_ce=%f", banner, epoch_ix + 1, val_agg)

        if val_agg < best_val_ce:

            best_val_ce = val_agg

            _save_checkpoint_bundle(

                root_out / cfg.keep_best_ckpt_name,

                model=model,

                optimizer=optim,

                scaler=scaler,

                global_step_opt=opt_step_global,

                epoch_ix=epoch_ix,

                cfg_obj=cfg,

                vocab_bound=vocab_cap,

                num_cb=num_cb,

            )

            logging.info("[%s NEW_BEST_CE] -> %f `%s`", banner, best_val_ce, cfg.keep_best_ckpt_name)

    tail = micro_step_epochless % int(cfg.grad_accum_steps)

    if tail != 0:

        logging.warning(

            "[Train-End] Tail AMP shard remainder (%s / %s) — clearing grads (+ scaler.update iff AMP enabled).",

            tail,

            cfg.grad_accum_steps,

        )

        optim.zero_grad(set_to_none=True)

        if scaler.is_enabled():

            scaler.update()

    plot_metrics_jsonl(jsonl_metric_path, root_out / "loss_curves_bundle.png")


if __name__ == "__main__":

    main()
