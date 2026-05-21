"""
Analyze training metrics.jsonl, optional train.log validation lines, and infer ablation_metrics.json.

Writes PNG figures to --out-dir when plotting is enabled (default): orth penalty + lambda,
validation CE vs epoch, ablation bar chart, scatter base error vs d(style).

Example (Windows paths in quotes):

  python scripts/analyze_run_outputs.py ^
    --metrics-jsonl "project_output/cover_project_full/cover_runs/exp1/metrics.jsonl" ^
    --ablation-json "project_output/cover_project_full/infer_out/ablation_metrics.json" ^
    --train-log "project_output/cover_project_full/cover_runs/exp1/train.log" ^
    --out-dir "project_output/cover_project_full/cover_runs/exp1/analyze_figures"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def load_jsonl_metrics(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") == "optimizer_step_summary":
                rows.append(d)
    return rows


def summarize_training(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No optimizer_step_summary rows found.")
        return

    n = len(rows)
    steps = [int(r["global_optimizer_step"]) for r in rows]

    def col(key: str) -> List[float]:
        return [float(r[key]) for r in rows]

    ce_b = col("train_ce_batch")
    ce_s = col("train_ce_smooth")
    tot_s = col("train_total_smooth")
    lam = col("lambda_orth_eff")
    orth_s = col("orth_penalty_scaled")
    orth_r = col("orth_penalty_raw")
    lr = col("lr")
    gn = col("grad_norm_after_clip")

    print("=" * 72)
    print("TRAINING (metrics.jsonl)")
    print("=" * 72)
    print(f"Rows: {n}  |  global_optimizer_step: {steps[0]} -> {steps[-1]}")
    print()

    print("Cross-entropy (smoothed, beta=0.98):")
    print(f"  min:   {min(ce_s):.6f}  at step {steps[ce_s.index(min(ce_s))]}")
    print(f"  max:   {max(ce_s):.6f}")
    print(f"  first: {ce_s[0]:.6f}  |  last: {ce_s[-1]:.6f}")
    print(f"  drop (first -> last): {ce_s[0] - ce_s[-1]:.6f}  ({(ce_s[0] - ce_s[-1]) / max(abs(ce_s[0]), 1e-9) * 100:.1f}% relative)")
    print()

    w = max(1, n // 20)
    print(f"Window means (first {w} vs last {w} optimizer steps):")
    print(f"  train_ce_smooth:     {mean(ce_s[:w]):.6f}  ->  {mean(ce_s[-w:]):.6f}")
    print(f"  train_total_smooth:  {mean(tot_s[:w]):.6f}  ->  {mean(tot_s[-w:]):.6f}")
    print(f"  train_ce_batch (raw batch, not EMA): first batch mean {mean(ce_b[:w]):.6f}  last batch mean {mean(ce_b[-w:]):.6f}")
    print()

    lam_pos = [i for i, v in enumerate(lam) if v > 1e-12]
    if lam_pos:
        i0 = lam_pos[0]
        print(f"Orthogonality schedule:")
        print(f"  lambda_orth_eff > 0 from global_step {rows[i0]['global_optimizer_step']}  (value {lam[i0]:.6e})")
    else:
        print("lambda_orth_eff stayed 0 for entire run.")
    print(f"  lambda final: {lam[-1]:.6e}  max: {max(lam):.6e}")
    print(f"  orth_penalty_raw:   min {min(orth_r):.4f}  max {max(orth_r):.4f}  last {orth_r[-1]:.4f}")
    print(f"  orth_penalty_scaled last: {orth_s[-1]:.6e}")
    print()

    print(f"LR:  first {lr[0]:.6e}  last {lr[-1]:.6e}")
    sorted_gn = sorted(gn)
    p95_idx = int(0.95 * (len(sorted_gn) - 1))
    print(f"grad_norm_after_clip:  median {median(gn):.4f}  p95 {sorted_gn[p95_idx]:.4f}  max {max(gn):.4f}")
    print()


def parse_val_log(train_log: Path) -> List[Tuple[int, float]]:
    """Return [(epoch, mean_batch_ce), ...] from train.log lines."""
    pat = re.compile(r"\[Epoch\s+(\d+)/\d+\].*epoch=(\d+)\s+VAL\]\s+mean_batch_ce=([0-9.]+)")
    out: List[Tuple[int, float]] = []
    with train_log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                epoch = int(m.group(2))
                ce = float(m.group(3))
                out.append((epoch, ce))
    return out


def summarize_validation(vals: List[Tuple[int, float]]) -> None:
    print("=" * 72)
    print("VALIDATION (from train.log - mean batch CE, no label smoothing)")
    print("=" * 72)
    if not vals:
        print("No VAL lines parsed (check --train-log path or log format).")
        print()
        return

    ces = [c for _, c in vals]
    best_i = min(range(len(ces)), key=lambda i: ces[i])
    best_ep, best_ce = vals[best_i]
    print(f"Epochs logged: {len(vals)}  |  best: epoch {best_ep}  val_ce={best_ce:.6f}")
    print(f"First epoch val_ce: {vals[0][1]:.6f}  |  last epoch val_ce: {vals[-1][1]:.6f}")

    if best_i < len(vals) - 1:
        drift = vals[-1][1] - best_ce
        print(f"Drift after best: last - best = {drift:+.6f} (positive => val worsening / mild overfitting on CE)")
    print()

    # Simple regime split: first third vs last third of epochs
    third = max(1, len(vals) // 3)
    early = mean([c for _, c in vals[:third]])
    late = mean([c for _, c in vals[-third:]])
    print(f"Mean val_ce first ~{third} epochs: {early:.6f}")
    print(f"Mean val_ce last ~{third} epochs:  {late:.6f}")
    if late > early + 1e-3:
        print("  - Validation CE rises in later epochs (typical overfitting or optimization past optimum).")
    elif late < early - 1e-3:
        print("  - Validation still improving on average (unusual if best is not at end - check parsing).")
    else:
        print("  - Early/late val means similar (flat regime).")
    print()


def load_ablation_json(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_ablation(data: Dict[str, Dict[str, float]]) -> None:
    print("=" * 72)
    print("INFERENCE ABLATION (ablation_metrics.json)")
    print("=" * 72)
    if not data:
        print("Empty JSON.")
        return

    keys = sorted(data.keys())
    print(f"Clips: {len(keys)}")
    print()

    hdr = (
        f"{'clip':<22}"
        f"{'lev_gt_base':>12}"
        f"{'d_style_gt':>11}"
        f"{'d_cnt_gt':>11}"
        f"{'|base-sty|':>11}"
        f"{'|base-cnt|':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    d_style_gt: List[float] = []
    d_cnt_gt: List[float] = []
    d_base_sty: List[float] = []
    d_base_cnt: List[float] = []
    lev_base: List[float] = []

    for k in keys:
        m = data[k]
        b = m["lev_mean_vs_gt_base"]
        s = m["lev_mean_vs_gt_mask_style_all"]
        c = m["lev_mean_vs_gt_mask_content_all"]
        bs = m["lev_mean_baseline_vs_mask_style_heads"]
        bc = m["lev_mean_baseline_vs_mask_content_heads"]
        ds = s - b
        dc = c - b
        print(f"{k:<22}{b:12.3f}{ds:10.3f}{dc:10.3f}{bs:11.3f}{bc:11.3f}")
        lev_base.append(b)
        d_style_gt.append(ds)
        d_cnt_gt.append(dc)
        d_base_sty.append(bs)
        d_base_cnt.append(bc)

    print()
    print("Aggregate (mean over clips):")
    print(f"  lev_mean_vs_gt_base:                    {mean(lev_base):.3f}  (median {median(lev_base):.3f})")
    print(f"  d vs GT when masking style (pos=worse): {mean(d_style_gt):+.3f}")
    print(f"  d vs GT when masking content:             {mean(d_cnt_gt):+.3f}")
    print(f"  |baseline - mask_style| (token change):  {mean(d_base_sty):.3f}")
    print(f"  |baseline - mask_content|:               {mean(d_base_cnt):.3f}")
    print()

    print("Interpretation hints:")
    print("  - d_style / d_content: how much worse GT match gets when that head group is ablated.")
    print("  - |base-sty| large: style ablation strongly changes autoregressive tokens vs full model.")
    print("  - |base-cnt| ~ 0: content ablation barely changes output (content path may be unused for that clip).")
    print()


def plot_orthogonality(rows: List[Dict[str, Any]], out_path: Path, plt: Any) -> None:
    """orth_penalty_raw vs step (log y), lambda_orth_eff on twin axis."""
    steps = [int(r["global_optimizer_step"]) for r in rows]
    orth_r = [float(r["orth_penalty_raw"]) for r in rows]
    lam = [float(r["lambda_orth_eff"]) for r in rows]
    orth_safe = [max(x, 1e-12) for x in orth_r]

    fig, ax1 = plt.subplots(figsize=(10.5, 4.25), constrained_layout=True)
    ax1.semilogy(steps, orth_safe, color="#1565c0", lw=1.0, label="orth_penalty_raw")
    ax1.set_xlabel("global_optimizer_step")
    ax1.set_ylabel("orth_penalty_raw (log scale, min clamp 1e-12)")
    ax1.set_title("Orthogonality: content/style cross-gram penalty vs schedule")
    ax1.grid(True, which="both", alpha=0.28)

    ax2 = ax1.twinx()
    ax2.plot(steps, lam, color="#c62828", lw=1.1, alpha=0.95, linestyle="--", label="lambda_orth_eff")
    ax2.set_ylabel("lambda_orth_eff")
    ax2.set_ylim(-0.01, max(max(lam), 0.01) * 1.15)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=164)
    plt.close(fig)


def plot_val_ce_epochs(vals: List[Tuple[int, float]], out_path: Path, plt: Any) -> None:
    if not vals:
        return
    epochs = [v[0] for v in vals]
    ces = [v[1] for v in vals]
    best_i = min(range(len(ces)), key=lambda i: ces[i])

    fig, ax = plt.subplots(figsize=(10.5, 4.0), constrained_layout=True)
    ax.plot(epochs, ces, lw=1.65, color="#2e7d32", label="mean_batch_ce (val)")
    ax.axvline(epochs[best_i], color="#c62828", ls="--", lw=1.35, alpha=0.9)
    ax.scatter(
        [epochs[best_i]],
        [ces[best_i]],
        color="#c62828",
        s=55,
        zorder=5,
        label=f"best: epoch {epochs[best_i]}  ce={ces[best_i]:.4f}",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mean batch CE (no label smoothing)")
    ax.set_title("Validation cross-entropy vs epoch")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=164)
    plt.close(fig)


def plot_ablation_bars(data: Dict[str, Dict[str, float]], out_path: Path, plt: Any) -> None:
    """Grouped bars: |baseline - mask_style| vs |baseline - mask_content| per clip."""
    if not data:
        return
    keys = sorted(data.keys())
    base_sty: List[float] = []
    base_cnt: List[float] = []
    for k in keys:
        m = data[k]
        base_sty.append(float(m["lev_mean_baseline_vs_mask_style_heads"]))
        base_cnt.append(float(m["lev_mean_baseline_vs_mask_content_heads"]))

    n = len(keys)
    x = list(range(n))
    w = 0.36
    labels = [k.replace("_shift_", "\nshift ") for k in keys]

    fig, ax = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
    ax.bar([i - w / 2 for i in x], base_sty, w, label="|base - mask_style|", color="#6a1b9a")
    ax.bar([i + w / 2 for i in x], base_cnt, w, label="|base - mask_content|", color="#00897b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("mean codebook Levenshtein (vs baseline)")
    ax.set_title("Ablation: token-sequence change when masking style vs content heads")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.28)
    fig.savefig(out_path, dpi=164)
    plt.close(fig)


def plot_base_vs_d_style_scatter(
    data: Dict[str, Dict[str, float]],
    out_path: Path,
    plt: Any,
) -> None:
    """Scatter: lev_mean_vs_gt_base vs (mask_style - base) toward GT match when style ablated."""
    if not data:
        return
    keys = sorted(data.keys())
    lev_b: List[float] = []
    d_style_gt: List[float] = []
    for k in keys:
        m = data[k]
        b = float(m["lev_mean_vs_gt_base"])
        s = float(m["lev_mean_vs_gt_mask_style_all"])
        lev_b.append(b)
        d_style_gt.append(s - b)

    fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
    ax.scatter(lev_b, d_style_gt, s=52, alpha=0.82, edgecolors="k", linewidths=0.35, color="#ef6c00")
    ax.set_xlabel("lev_mean_vs_gt_base (full model vs target tokens)")
    ax.set_ylabel("delta vs GT when masking style (pos = worse match)")
    ax.set_title("Clip difficulty vs impact of style-head ablation on GT match")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.grid(True, alpha=0.28)
    for i, k in enumerate(keys):
        ax.annotate(
            k,
            (lev_b[i], d_style_gt[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6.5,
            alpha=0.92,
        )
    fig.savefig(out_path, dpi=164)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("project_output/cover_project_full/cover_runs/exp1/metrics.jsonl"),
        help="Path to metrics.jsonl from training",
    )
    ap.add_argument(
        "--ablation-json",
        type=Path,
        default=Path("project_output/cover_project_full/infer_out/ablation_metrics.json"),
        help="Path to ablation_metrics.json from infer_eval",
    )
    ap.add_argument(
        "--train-log",
        type=Path,
        default=Path("project_output/cover_project_full/cover_runs/exp1/train.log"),
        help="Optional train.log for per-epoch validation CE",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analyze_run_figures"),
        help="Directory for PNG figures (created if missing)",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Print tables only; do not write matplotlib PNGs",
    )
    args = ap.parse_args()

    if not args.metrics_jsonl.is_file() and not args.ablation_json.is_file():
        ap.error(
            "No inputs found. Set --metrics-jsonl and/or --ablation-json to existing files "
            "(defaults point at project_output/... if present)."
        )

    plt = None if args.no_plots else _try_import_matplotlib()
    if not args.no_plots and plt is None:
        print("[warn] matplotlib not installed; install with `pip install matplotlib` or use --no-plots\n", file=sys.stderr)
        plt = None

    rows: List[Dict[str, Any]] = []
    if args.metrics_jsonl is not None and args.metrics_jsonl.is_file():
        rows = load_jsonl_metrics(args.metrics_jsonl)
        summarize_training(rows)
    elif args.metrics_jsonl is not None:
        print(f"[warn] metrics.jsonl not found: {args.metrics_jsonl}\n")

    vals: List[Tuple[int, float]] = []
    if args.train_log is not None and args.train_log.is_file():
        vals = parse_val_log(args.train_log)
        summarize_validation(vals)
    elif args.train_log is not None:
        print(f"[warn] train.log not found: {args.train_log}\n")

    ablation_data: Optional[Dict[str, Dict[str, float]]] = None
    if args.ablation_json is not None and args.ablation_json.is_file():
        ablation_data = load_ablation_json(args.ablation_json)
        summarize_ablation(ablation_data)
    elif args.ablation_json is not None:
        print(f"[warn] ablation_metrics.json not found: {args.ablation_json}\n")

    if plt is None:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Figures directory:", args.out_dir.resolve())

    if rows:
        plot_orthogonality(rows, args.out_dir / "orth_penalty_vs_step.png", plt)
        print("  wrote", args.out_dir / "orth_penalty_vs_step.png")
    if vals:
        plot_val_ce_epochs(vals, args.out_dir / "val_mean_ce_vs_epoch.png", plt)
        print("  wrote", args.out_dir / "val_mean_ce_vs_epoch.png")
    if ablation_data:
        plot_ablation_bars(ablation_data, args.out_dir / "ablation_token_shift_bars.png", plt)
        print("  wrote", args.out_dir / "ablation_token_shift_bars.png")
        plot_base_vs_d_style_scatter(ablation_data, args.out_dir / "lev_gt_base_vs_d_style_gt_scatter.png", plt)
        print("  wrote", args.out_dir / "lev_gt_base_vs_d_style_gt_scatter.png")
    print()


if __name__ == "__main__":
    main()
