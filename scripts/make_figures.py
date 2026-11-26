"""README figures. Two independent modes, run from the project root:

    python -m scripts.make_figures results
        Local, no torch. Reads .project_state/results/dashboard_data.json and
        writes assets/{training_curve,gap_to_ceiling,postprocess_sweep}.png.

    python -m scripts.make_figures predictions --ckpt runs/run40/best.pt
        Needs torch + the checkpoint (run on the GPU studio). Writes
        assets/predictions.png: image | ground truth | prediction for 2
        fluorescence + 2 brightfield val images, with each image's mAP.

Why two modes: the training/sweep numbers are already saved as JSON, so those
plots must never require a GPU. Only the qualitative grid needs the model.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# Agg = file-only backend. Must be selected BEFORE pyplot is imported, otherwise
# pyplot binds an interactive backend (fails on a headless SSH host).
matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- style ------------------------------------------------------------------
# Categorical slots 1-3 of the validated default palette (CVD-checked: worst
# adjacent deutan dE 9.2). Fixed ORDER, never cycled, and each series keeps its
# color across every figure so "TTA" is always the same hue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    # Recessive frame: drop top/right spines, light horizontal grid only.
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.8, "lines.linewidth": 2.0, "legend.frameon": False,
})

ASSETS = ROOT / "assets"


def _label_end(ax, x, y, text, color):
    """Direct label at a line's last point. Text stays in ink (not series
    color); the small colored marker beside it carries identity."""
    ax.plot([x], [y], "o", ms=5, color=color)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


# ---- mode 1: plots from saved JSON ------------------------------------------
def make_results(state_dir: Path) -> None:
    d = json.loads((state_dir / "results" / "dashboard_data.json").read_text())
    ASSETS.mkdir(exist_ok=True)
    curve = d["curve"]
    ep = np.array([r["epoch"] for r in curve])

    # Figure 1: two panels on SEPARATE axes (loss and mAP have different
    # scales - one y-axis per measure, never a dual-axis chart).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(ep, [r["train_loss"] for r in curve], color=BLUE)
    a1.set_title("Training loss (CE + dice)")
    a1.set_xlabel("epoch")

    series = [("val_map50", "mAP@0.50", BLUE), ("val_map", "mAP (0.50-0.95)", ORANGE),
              ("val_map90", "mAP@0.90", AQUA)]
    for key, name, col in series:
        y = [r[key] for r in curve]
        a2.plot(ep, y, color=col, label=name)
        _label_end(a2, ep[-1], y[-1], name, col)
    best = max(curve, key=lambda r: r["val_map"])
    # Dashed guide at the best epoch: the checkpoint kept is NOT the last one.
    a2.axvline(best["epoch"], color=INK_2, ls="--", lw=1)
    a2.annotate(f"best {best['val_map']:.3f}\n(epoch {best['epoch']})",
                (best["epoch"], 0.05), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK_2)
    a2.set_ylim(0, 1)
    a2.set_xlim(ep[0], ep[-1] + 9)  # right margin for the direct labels
    a2.set_title("Validation mAP (135 full images)")
    a2.set_xlabel("epoch")
    a2.legend(loc="upper left", fontsize=8.5, ncol=3)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curve.png")
    plt.close(fig)

    # Figure 2: where the gap is. Grouped bars at the 3 thresholds where the
    # model's numbers were saved; ceiling = perfect prediction through the
    # same 3-class -> watershed pipeline.
    ceil = d["ceiling"]["per_threshold"]  # 10 values for t = 0.50..0.95
    tta = max(d["tta_seed_sweep"], key=lambda r: r["map"])
    labels = ["IoU 0.50", "IoU 0.75", "IoU 0.90", "mean 0.50-0.95"]
    c_vals = [ceil[0], ceil[5], ceil[8], d["ceiling"]["map"]]
    m_vals = [tta["map_50"], tta["map_75"], tta["map_90"], tta["map"]]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.6))
    # edgecolor=SURFACE + linewidth=2 draws the 2px surface gap between fills.
    b1 = ax.bar(x - w / 2, c_vals, w, color=BLUE, edgecolor=SURFACE, lw=2,
                label="Representation ceiling")
    b2 = ax.bar(x + w / 2, m_vals, w, color=ORANGE, edgecolor=SURFACE, lw=2,
                label="Model (U-Net + TTA)")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8.5, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_title("Gap to ceiling is boundary precision, not detection")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "gap_to_ceiling.png")
    plt.close(fig)

    # Figure 3: post-processing sweep. Same axis (mAP), two inference paths.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for key, name, col in (("seed_sweep", "no TTA", BLUE),
                           ("tta_seed_sweep", "flip TTA", ORANGE)):
        rows = sorted(d[key], key=lambda r: r["seed_threshold"])
        xs = [r["seed_threshold"] for r in rows]
        ys = [r["map"] for r in rows]
        ax.plot(xs, ys, color=col, marker="o", ms=4, label=name)
        i = int(np.argmax(ys))
        ax.annotate(f"{ys[i]:.3f}", (xs[i], ys[i]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8.5)
    # Untuned reference: model output with the default 0.5/0.5 thresholds.
    ax.axhline(best["val_map"], color=INK_2, ls="--", lw=1)
    # Explicit y-range leaves headroom above the peak label and room for the
    # baseline caption, which sits ABOVE its dashed line (axes-fraction x).
    ax.set_ylim(0.565, 0.628)
    ax.annotate(f"untuned thresholds (training run): {best['val_map']:.3f}",
                (0.01, best["val_map"]), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlabel("watershed seed threshold  P(interior)")
    ax.set_title("Post-processing tuning: +0.045 mAP with no retraining")
    # Anchor in the empty band between the curves (~0.60) and the baseline.
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.38), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "postprocess_sweep.png")
    plt.close(fig)
    print(f"wrote 3 figures -> {ASSETS}")


# ---- mode 2: qualitative grid (needs torch + checkpoint) ---------------------
def _color_instances(lab: np.ndarray, seed: int = 0) -> np.ndarray:
    """Instance label map -> RGB with a random color per id, black background.
    Lookup-table indexing (lut[lab]) colors every pixel in one vectorized pass."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(60, 256, size=(int(lab.max()) + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[lab]


def _pick_typical(val, per_modality: int = 2):
    """Per modality, the val images whose instance count is closest to that
    modality's median -> typical images, deterministic, not cherry-picked."""
    picks = []
    for mod in ("fluorescence", "brightfield"):
        recs = [r for r in val if r["modality"] == mod]
        med = np.median([r["n_instances"] for r in recs])
        recs.sort(key=lambda r: abs(r["n_instances"] - med))
        picks += recs[:per_modality]
    return picks


def make_targets(args) -> None:
    """Image | instance labels | 3-class target, from the cache. No model needed.
    Shows WHY the 3-class target exists: touching nuclei get a border wall."""
    from src.data_loader import SampleCache
    from src.dataset_manifest import load_manifest

    manifest, _ = load_manifest(ROOT / args.state_dir)
    val = [r for r in manifest["samples"] if r["split"] == "val"]
    # Highest touching density per modality among typical images: pick the
    # 6 most typical, then keep the 2 densest (more touching pairs to show).
    typical = _pick_typical(val, per_modality=6)  # 6 fluo + 6 bright
    picks = []
    for mod in ("fluorescence", "brightfield"):
        cand = [r for r in typical if r["modality"] == mod]
        # Densest first: instances per pixel -> more touching pairs on show.
        picks += sorted(cand, key=lambda r: -r["n_instances"] / r["area"])[:1]
    cache = SampleCache(ROOT / args.cache_dir, [r["sample_id"] for r in picks],
                        mode="mmap", keep_label=True)
    # Class palette: bg = near-black, interior = blue, border = orange
    # (categorical slots 1-2, same hues as the charts).
    cls_lut = np.array([[24, 24, 23], [0x2a, 0x78, 0xd6], [0xeb, 0x68, 0x34]], np.uint8)

    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3.4 * len(picks)))
    for row, r in zip(axes, picks):
        img, tgt, lab = (np.asarray(a) for a in cache.get(r["sample_id"]))
        # The cache stores brightfield INVERTED (polarity flip baked in at
        # cache time); undo it for display so the image looks as acquired.
        shown = 255 - img if r["modality"] == "brightfield" else img
        # Zoom to a 256x256 window around the densest region so the 1-2 px
        # border wall is visible at README size; window = argmax of a box sum.
        h, w = lab.shape
        s = min(256, h, w)
        fg = (lab > 0).astype(np.float32)
        dens = cv2.boxFilter(fg, -1, (s, s), normalize=False)  # local fg count
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
        y0 = int(np.clip(cy - s // 2, 0, h - s))
        x0 = int(np.clip(cx - s // 2, 0, w - s))
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))
        panels = (shown[sl], _color_instances(lab[sl]), cls_lut[tgt[sl]])
        titles = (f"{r['modality']} (crop {s}x{s})",
                  f"instances ({len(np.unique(lab[sl])) - 1} in crop)",
                  "3-class target: interior / border")
        for ax, im, t in zip(row, panels, titles):
            ax.imshow(im, interpolation="nearest")  # nearest: keep 1px walls crisp
            ax.set_title(t, fontsize=9, loc="left", fontweight="normal")
            ax.axis("off")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "targets.png")
    print(f"wrote {ASSETS / 'targets.png'}")


def make_predictions(args) -> None:
    import torch  # local import: mode 1 must work without torch installed

    from src.data_loader import SampleCache
    from src.dataset_manifest import load_manifest
    from src.inference import sliding_window_probs
    from src.metrics import average_precision
    from src.model import build_model
    from src.postprocess import TUNED_TTA, instances_from_probs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest, _ = load_manifest(ROOT / args.state_dir)
    val = [r for r in manifest["samples"] if r["split"] == "val"]
    picks = _pick_typical(val)

    cache = SampleCache(ROOT / args.cache_dir, [r["sample_id"] for r in picks],
                        mode="mmap", keep_label=True)
    ckpt = torch.load(ROOT / args.ckpt, map_location=device)
    saved = ckpt.get("args", {})
    model = build_model(saved.get("encoder", "resnet34"), classes=3,
                        encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = saved.get("norm", "imagenet")

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    with torch.no_grad():
        for row, r in zip(axes, picks):
            img, _, lab = cache.get(r["sample_id"])
            img = np.asarray(img)
            # Same TTA path as scripts/sweep_postprocess.py: average the 4 flip
            # variants in probability space, undo each flip on the CHW output.
            acc = None
            for flip in ((), (1,), (2,), (1, 2)):
                fimg = np.flip(img, axis=tuple(a - 1 for a in flip)) if flip else img
                p = sliding_window_probs(model, np.ascontiguousarray(fimg), norm=norm,
                                         device=device, amp=True)
                p = np.flip(p, axis=flip) if flip else p
                acc = p if acc is None else acc + p
            probs = acc / 4.0
            pred = instances_from_probs(probs, **TUNED_TTA)
            gt = np.asarray(lab).astype(np.int32)
            ap = average_precision(gt, pred)["map"]
            for ax, im, title in zip(
                row, (img, _color_instances(gt), _color_instances(pred, 1)),
                (f"{r['modality']}  {r['h']}x{r['w']}",
                 f"ground truth ({int(gt.max())})",
                 f"prediction ({int(pred.max())})  mAP {ap:.2f}")):
                ax.imshow(im)
                ax.set_title(title, fontsize=9, loc="left", fontweight="normal")
                ax.axis("off")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "predictions.png")
    print(f"wrote {ASSETS / 'predictions.png'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    s1 = sub.add_parser("results")
    s1.add_argument("--state-dir", default=".project_state")
    s2 = sub.add_parser("predictions")
    s2.add_argument("--ckpt", required=True)
    s3 = sub.add_parser("targets")
    for s in (s2, s3):  # shared data-location flags
        s.add_argument("--state-dir", default=".project_state")
        s.add_argument("--cache-dir", default="data/cache")
    a = p.parse_args(argv)
    if a.mode == "results":
        make_results(ROOT / a.state_dir)
    elif a.mode == "targets":
        make_targets(a)
    else:
        make_predictions(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
