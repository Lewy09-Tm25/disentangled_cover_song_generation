"""
Disentanglement-oriented analysis (thesis add-ons).

  (2) CKA / alignment: linear CKA between pooled harmonic-encoder states and style vectors
      (raw MERT vs projected), plus mean absolute cosine similarity per sample.

  (3) Embedding plots: 2-D t-SNE (default) or UMAP of style vectors and content summaries,
      colored by song (work_id) or cover performance (target_perf_id) from augmented_pairs.csv.

Examples (from repo root):

  python scripts/disentanglement_analysis.py cka ^
    --checkpoint project_output/cover_project_full/cover_runs/exp1/checkpoint_best_val.pt ^
    --tensor-dir path/to/tensor_dataset --manifest-csv tensor_extraction_checkpoint.csv --split val

  python scripts/disentanglement_analysis.py embed ^
    --checkpoint ... --tensor-dir ... --manifest-csv tensor_extraction_checkpoint.csv ^
    --pairs-csv augmented_pairs.csv --out-dir analyze_embed_figs --split val --max-samples 1500

Requires: torch, pandas, numpy, matplotlib, scikit-learn. Optional: umap-learn for --method umap.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root (so `python scripts/disentanglement_analysis.py` finds `covers_training`)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

from covers_training.config import TrainConfig, split_work_ids
from covers_training.dataset import CoverSongTensorDataset, padded_collate_fn
from covers_training.model import CoverTranslator


def train_config_from_bundle(bundle: Dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    raw = bundle.get("cfg_dict")
    if isinstance(raw, dict):
        names = {f.name for f in fields(TrainConfig)}
        for k, v in raw.items():
            if k in names:
                setattr(cfg, k, v)
    return cfg


def load_model(bundle: Dict[str, Any], device: torch.device) -> CoverTranslator:
    cfg = train_config_from_bundle(bundle)
    model = CoverTranslator(
        cfg=cfg,
        num_codebooks=int(bundle["num_codebooks"]),
        vocab_size=int(bundle["vocab_cap"]),
    )
    model.load_state_dict(bundle["model_sd"])
    model.to(device)
    model.eval()
    return model


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al.); X,Y shape (n, p), (n, q), same n."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA: row count mismatch")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xt_x = X.T @ X
    yt_y = Y.T @ Y
    yt_x = Y.T @ X
    num = np.linalg.norm(yt_x, ord="fro") ** 2
    den = np.linalg.norm(xt_x, ord="fro") * np.linalg.norm(yt_y, ord="fro")
    return float(num / max(den, 1e-12))


def mean_abs_cosine_pairs(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean absolute cosine between aligned rows (same dim or project — caller must match dims)."""
    if X.shape != Y.shape:
        raise ValueError("Cosine pairs: shape mismatch; use same dimension (e.g. projected style).")
    xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    cos = np.sum(xn * yn, axis=1)
    return float(np.mean(np.abs(cos)))


def masked_mean_pool(mem: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """mem [B,L,D], pad_mask True=PAD -> [B,D]"""
    valid = (~pad_mask).float().unsqueeze(-1)
    summed = (mem * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1e-6)
    return summed / count


def collect_representations(
    *,
    model: CoverTranslator,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (harm_pooled, style_raw, style_proj) each (N, *) numpy float32."""
    harm_list: List[np.ndarray] = []
    raw_list: List[np.ndarray] = []
    proj_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            content = batch["content"].to(device)
            style = batch["style"].to(device)
            cmask = batch["content_pad_mask"].to(device)

            hm = model._encode_harmonics(content, cmask)
            pooled = masked_mean_pool(hm, cmask)
            sp = model.style_in(style)

            harm_list.append(pooled.cpu().numpy().astype(np.float32))
            raw_list.append(style.cpu().numpy().astype(np.float32))
            proj_list.append(sp.cpu().numpy().astype(np.float32))

    return (
        np.concatenate(harm_list, axis=0),
        np.concatenate(raw_list, axis=0),
        np.concatenate(proj_list, axis=0),
    )


def run_cka(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bundle = torch.load(Path(args.checkpoint).expanduser(), map_location=device)
    model = load_model(bundle, device)

    cfg = train_config_from_bundle(bundle)
    ds = CoverSongTensorDataset(
        tensor_dir=str(args.tensor_dir),
        manifest_csv=str(args.manifest_csv),
        split=args.split,
        cfg=cfg,
    )
    n = len(ds)
    lim = args.max_samples
    if lim is None or lim <= 0 or lim >= n:
        sub = ds
    else:
        indices = np.random.RandomState(args.seed).choice(n, size=lim, replace=False)
        sub = torch.utils.data.Subset(ds, indices.tolist())

    loader = torch.utils.data.DataLoader(
        sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=padded_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    H, S_raw, S_proj = collect_representations(model=model, loader=loader, device=device)

    cka_raw = linear_cka(H, S_raw)
    cka_proj = linear_cka(H, S_proj)
    mac = mean_abs_cosine_pairs(H, S_proj)

    lines = [
        f"samples_used={H.shape[0]}  d_harm={H.shape[1]}  d_style_raw={S_raw.shape[1]}  d_style_proj={S_proj.shape[1]}",
        f"linear_cka(harm_pooled, style_mert_raw)     = {cka_raw:.6f}",
        f"linear_cka(harm_pooled, style_after_linear) = {cka_proj:.6f}",
        f"mean_abs_cosine(harm_pooled, style_proj) [same row] = {mac:.6f}",
        "",
        "Notes:",
        "  - Lower CKA => less linearly shared geometry between pooled harmonic states and style (batch of samples).",
        "  - mean_abs_cosine measures per-sample alignment of harmonic vs projected style in R^d_model.",
    ]
    text = "\n".join(lines)
    print(text)

    out_json = Path(args.out_json) if args.out_json else None
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_samples": int(H.shape[0]),
            "linear_cka_harm_vs_style_raw": cka_raw,
            "linear_cka_harm_vs_style_proj": cka_proj,
            "mean_abs_cosine_harm_vs_style_proj_rows": mac,
            "split": args.split,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[wrote] {out_json}")


def _merge_manifest_pairs(df: pd.DataFrame, pairs_path: Path) -> pd.DataFrame:
    pr = pd.read_csv(pairs_path)
    need = {"work_id", "shift_amount", "target_perf_id", "anchor_perf_id"}
    miss = need - set(pr.columns)
    if miss:
        raise ValueError(f"pairs_csv missing columns: {sorted(miss)}")
    out = df.merge(
        pr[list(need)],
        on=["work_id", "shift_amount"],
        how="left",
    )
    if out["target_perf_id"].isna().any():
        n_miss = int(out["target_perf_id"].isna().sum())
        print(
            f"[warn] {n_miss} rows missing target_perf_id after merge; labeled -1.",
            file=sys.stderr,
        )
        out["target_perf_id"] = out["target_perf_id"].fillna(-1)
        out["anchor_perf_id"] = out["anchor_perf_id"].fillna(-1)
    out["target_perf_id"] = out["target_perf_id"].astype(int)
    out["anchor_perf_id"] = out["anchor_perf_id"].astype(int)
    return out


def _try_import_sklearn():
    try:
        from sklearn.manifold import TSNE

        return TSNE
    except ImportError:
        return None


def _try_import_umap():
    try:
        import umap

        return umap.UMAP
    except ImportError:
        return None


def _fit_2d(Z: np.ndarray, method: str, seed: int, perplexity: float) -> np.ndarray:
    if method == "tsne":
        TSNE = _try_import_sklearn()
        if TSNE is None:
            raise SystemExit("Install scikit-learn for t-SNE: pip install scikit-learn")
        n = Z.shape[0]
        perp = min(float(perplexity), max(5.0, (n - 1) / 3.0))
        ts = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perp,
            random_state=seed,
        )
        return ts.fit_transform(Z)
    if method == "umap":
        UMAP = _try_import_umap()
        if UMAP is None:
            raise SystemExit("Install umap-learn: pip install umap-learn")
        return UMAP(n_components=2, random_state=seed, min_dist=0.1).fit_transform(Z)
    raise ValueError(method)


def _scatter_2d_category(
    xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    label_name: str,
    mod: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = np.where(labels >= 0, labels.astype(np.int64) % int(mod), np.full_like(labels, fill_value=mod, dtype=np.int64))
    uniq = np.sort(np.unique(labs))
    fig, ax = plt.subplots(figsize=(9.0, 7.0), constrained_layout=True)
    try:
        cmap = plt.colormaps["tab20"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20")
    for j, u in enumerate(uniq):
        m = labs == u
        label_txt = "missing" if int(u) == mod else str(int(u))
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=12,
            alpha=0.72,
            label=label_txt,
            color=cmap((j % 20) / 20.0),
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(
        fontsize=7,
        loc="best",
        ncol=2,
        markerscale=1.2,
        title=f"{label_name} mod {mod} (missing sep.)",
        framealpha=0.9,
    )
    fig.savefig(out_path, dpi=164)
    plt.close(fig)


def collect_embed_batches(
    *,
    loader: torch.utils.data.DataLoader,
    model: CoverTranslator,
    device: torch.device,
    lookup: Dict[Tuple[int, int], int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop-aligned tensors (same as training). Returns S, chroma_mean, harm_pooled, work_ids, target_perf."""
    s_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
    w_list: List[int] = []
    t_list: List[int] = []

    with torch.no_grad():
        for batch in loader:
            content = batch["content"].to(device)
            style = batch["style"].to(device)
            cmask = batch["content_pad_mask"].to(device)
            hm = model._encode_harmonics(content, cmask)
            hp = masked_mean_pool(hm, cmask)
            valid = (~cmask).float().unsqueeze(-1)
            c_mean = (content * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-6)

            s_list.append(style.cpu().numpy().astype(np.float32))
            c_list.append(c_mean.cpu().numpy().astype(np.float32))
            h_list.append(hp.cpu().numpy().astype(np.float32))

            for w, sh in zip(batch["work_id"].tolist(), batch["shift"].tolist()):
                w_list.append(int(w))
                t_list.append(int(lookup.get((int(w), int(sh)), -1)))

    return (
        np.concatenate(s_list, axis=0),
        np.concatenate(c_list, axis=0),
        np.concatenate(h_list, axis=0),
        np.asarray(w_list, dtype=np.int64),
        np.asarray(t_list, dtype=np.int64),
    )


def run_embed(args: argparse.Namespace) -> None:
    if _try_import_sklearn() is None:
        raise SystemExit("Install scikit-learn: pip install scikit-learn")

    device = torch.device(args.device)
    bundle = torch.load(Path(args.checkpoint).expanduser(), map_location=device)
    model = load_model(bundle, device)

    cfg = train_config_from_bundle(bundle)
    ds = CoverSongTensorDataset(
        tensor_dir=str(args.tensor_dir),
        manifest_csv=str(args.manifest_csv),
        split=args.split,
        cfg=cfg,
    )
    meta_df = _merge_manifest_pairs(ds.df_rows.copy(), Path(args.pairs_csv))
    lookup: Dict[Tuple[int, int], int] = {}
    for _, row in meta_df.iterrows():
        lookup[(int(row["work_id"]), int(row["shift_amount"]))] = int(row["target_perf_id"])

    n = len(ds)
    lim = args.max_samples
    if lim is None or lim <= 0 or lim >= n:
        sub_d: torch.utils.data.Dataset = ds
    else:
        ix = np.random.RandomState(args.seed).choice(n, size=lim, replace=False).tolist()
        sub_d = torch.utils.data.Subset(ds, ix)

    loader = torch.utils.data.DataLoader(
        sub_d,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=padded_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    S, C_mean, H_pooled, work_ids, target_pf = collect_embed_batches(
        loader=loader, model=model, device=device, lookup=lookup
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method = args.method
    mod = int(args.color_mod)

    Zs = _fit_2d(S, method=method, seed=args.seed, perplexity=args.perplexity)
    _scatter_2d_category(
        Zs,
        target_pf.astype(np.float64),
        f"Style (MERT 1024) -> {method.upper()}  (color: target_perf_id)",
        out_dir / f"embed_style_target_perf_{method}.png",
        "target_perf_id",
        mod,
    )

    Zc = _fit_2d(
        C_mean,
        method=method,
        seed=args.seed + 1,
        perplexity=min(args.perplexity, 30.0),
    )
    _scatter_2d_category(
        Zc,
        work_ids.astype(np.float64),
        f"Mean chroma crop (12-D) -> {method.upper()}  (color: work_id)",
        out_dir / f"embed_chroma_mean_work_id_{method}.png",
        "work_id",
        mod,
    )

    Zh = _fit_2d(H_pooled, method=method, seed=args.seed + 2, perplexity=args.perplexity)
    _scatter_2d_category(
        Zh,
        work_ids.astype(np.float64),
        f"Pooled harmonic encoder -> {method.upper()}  (color: work_id)",
        out_dir / f"embed_harmonic_pooled_work_id_{method}.png",
        "work_id",
        mod,
    )

    meta = {
        "n_points": int(S.shape[0]),
        "split": args.split,
        "method": method,
        "color_mod": mod,
        "note": "Chroma and harmonic features use the same random crop as CoverSongTensorDataset.__getitem__.",
        "outputs": [
            str(out_dir / f"embed_style_target_perf_{method}.png"),
            str(out_dir / f"embed_chroma_mean_work_id_{method}.png"),
            str(out_dir / f"embed_harmonic_pooled_work_id_{method}.png"),
        ],
    }
    (out_dir / "embed_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote figures to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ck = sub.add_parser("cka", help="Linear CKA + cosine alignment between harmonic pool and style")
    ck.add_argument("--checkpoint", type=Path, required=True)
    ck.add_argument("--tensor-dir", type=Path, required=True)
    ck.add_argument("--manifest-csv", type=Path, required=True)
    ck.add_argument("--split", choices=("train", "val", "test"), default="val")
    ck.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help="Subset size; 0 or negative = use full split",
    )
    ck.add_argument("--batch-size", type=int, default=16)
    ck.add_argument("--num-workers", type=int, default=0)
    ck.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ck.add_argument("--seed", type=int, default=42)
    ck.add_argument("--out-json", type=Path, default=None, help="Optional path to save numeric summary JSON")
    ck.set_defaults(func=run_cka)

    em = sub.add_parser("embed", help="t-SNE/UMAP scatter plots for style, chroma, harmonic encodings")
    em.add_argument("--checkpoint", type=Path, required=True)
    em.add_argument("--tensor-dir", type=Path, required=True)
    em.add_argument("--manifest-csv", type=Path, required=True)
    em.add_argument("--pairs-csv", type=Path, required=True, help="augmented_pairs.csv (for performance ids)")
    em.add_argument("--out-dir", type=Path, required=True)
    em.add_argument("--split", choices=("train", "val", "test"), default="val")
    em.add_argument(
        "--max-samples",
        type=int,
        default=1200,
        help="Subset size for plotting speed; 0 or negative = full split",
    )
    em.add_argument("--batch-size", type=int, default=16)
    em.add_argument("--num-workers", type=int, default=0)
    em.add_argument("--method", choices=("tsne", "umap"), default="tsne")
    em.add_argument("--perplexity", type=float, default=30.0)
    em.add_argument("--color-mod", type=int, default=24, help="Label bucket for colors: value %% color_mod")
    em.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    em.add_argument("--seed", type=int, default=42)
    em.set_defaults(func=run_embed)

    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.", file=sys.stderr)
        args.device = "cpu"

    args.func(args)


if __name__ == "__main__":
    main()
