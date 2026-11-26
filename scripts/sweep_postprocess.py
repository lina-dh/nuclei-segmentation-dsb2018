"""Post-processing sweep against a trained checkpoint. Inference-only, no training.

    python -m scripts.sweep_postprocess --ckpt runs/run40/best.pt [--tta]

THE KEY EFFICIENCY TRICK: the sliding-window forward pass does NOT depend on any
post-processing parameter - only watershed does. So probabilities are computed
ONCE for all 135 val images and cached in RAM (~252 MB float32), then every
threshold is evaluated against that cache. A 11-point sweep therefore costs one
inference pass plus 11 cheap watershed passes, not 11 full inference passes.

WHY THE DIRECTION OF seed_threshold IS NOT OBVIOUS:
  LOWER  -> seed regions grow -> adjacent seeds can MERGE through a weak border,
            giving FEWER instances (under-segmentation)
  HIGHER -> seed regions shrink -> one seed can SPLIT into several, giving MORE
            instances, but faint nuclei can vanish entirely, giving fewer
Both directions can reduce the instance count by different mechanisms, so the
response is non-monotonic and has to be measured rather than reasoned about.

Reminder on what the metric does to errors here: a split instance scores 0, not
0.5 (each half has IoU exactly 0.5, which is not > 0.5). Over-segmentation is
punished as hard as a miss.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import SampleCache  # noqa: E402
from src.dataset_manifest import load_manifest  # noqa: E402
from src.inference import DEFAULT_STRIDE, sliding_window_probs  # noqa: E402
from src.metrics import dataset_map  # noqa: E402
from src.model import build_model  # noqa: E402
from src.postprocess import MIN_AREA, instances_from_probs  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--state-dir", default=".project_state")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--tta", action="store_true",
                   help="average probabilities over the 4 D4 flips (free accuracy, 4x inference)")
    p.add_argument("--out", default=".project_state/sweep_results.json")
    # Grids are CLI-settable because a first pass can peak at a grid boundary,
    # and a boundary optimum is not an optimum - it just means the grid was too
    # narrow. These let the sweep be re-run wider without editing code.
    p.add_argument("--seed-grid", default="0.25:0.86:0.05",
                   help="start:stop:step, or a comma-separated list")
    p.add_argument("--fg-grid", default="0.30:0.76:0.05")
    p.add_argument("--area-grid", default="1,5,10,15,20")
    # The three parameters interact, so the sweep is greedy coordinate ascent.
    # These pin the OTHER two while seed_threshold is swept, which lets a second
    # pass re-check the seed optimum at the operating point the first pass found.
    p.add_argument("--seed-sweep-fg", type=float, default=0.5)
    p.add_argument("--seed-sweep-area", type=int, default=MIN_AREA)
    return p.parse_args(argv)


def _grid(spec: str) -> List[float]:
    """Parse 'start:stop:step' or '1,5,10' into a list of values."""
    if ":" in spec:
        a, b, c = (float(x) for x in spec.split(":"))
        return [round(v, 4) for v in np.arange(a, b, c)]
    return [float(x) for x in spec.split(",")]


def evaluate(cached: List[Dict[str, Any]], **pp) -> Dict[str, Any]:
    """Watershed + score the cached probabilities under one parameter set."""
    pairs = []
    n_pred = 0
    for item in cached:
        pred = instances_from_probs(item["probs"], **pp)
        pairs.append((item["sid"], item["gt"], pred))
        n_pred += int(pred.max())
    d = dataset_map(pairs)
    return {
        "map": round(d["map"], 4),
        "map_50": round(d["per_threshold"][0]["mean_precision"], 4),
        "map_75": round(d["per_threshold"][5]["mean_precision"], 4),
        "map_90": round(d["per_threshold"][8]["mean_precision"], 4),
        "pred_instances": n_pred,
    }


@torch.no_grad()
def main(argv=None) -> int:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest, _ = load_manifest(ROOT / args.state_dir)
    val_recs = [r for r in manifest["samples"] if r["split"] == "val"]
    if args.limit:
        val_recs = val_recs[:args.limit]
    val_ids = [r["sample_id"] for r in val_recs]
    cache = SampleCache(ROOT / args.cache_dir, val_ids, mode="mmap", keep_label=True)

    ckpt = torch.load(ROOT / args.ckpt, map_location=device)
    saved = ckpt.get("args", {})
    model = build_model(saved.get("encoder", "resnet34"), classes=3,
                        encoder_weights=None).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = saved.get("norm", "imagenet")
    print(f"ckpt {args.ckpt} | epoch {ckpt.get('epoch', -1) + 1} | "
          f"recorded best_map {ckpt.get('best_map', float('nan')):.4f} | norm={norm}")

    # ---- pass 1: inference once -------------------------------------------
    t0 = time.perf_counter()
    cached: List[Dict[str, Any]] = []
    gt_total = 0
    for r in val_recs:
        sid = r["sample_id"]
        img, _, lab = cache.get(sid)
        probs = sliding_window_probs(
            model, np.asarray(img), crop=args.crop, stride=args.stride,
            norm=norm, device=device, amp=not args.no_amp,
            batch_size=args.batch_size,
        )
        if args.tta:
            # D4 flips only (not the 90-degree rotations): a flip is an exact
            # index reversal, so un-flipping the output is lossless and needs no
            # interpolation. Average in probability space, then renormalize.
            acc = probs.copy()
            for flip in ((1,), (2,), (1, 2)):
                # Flip spatial axes of the HWC image; probs come back CHW so the
                # inverse flip is applied to axes shifted by one.
                fimg = np.flip(np.asarray(img), axis=tuple(a - 1 for a in flip))
                p = sliding_window_probs(
                    model, np.ascontiguousarray(fimg), crop=args.crop,
                    stride=args.stride, norm=norm, device=device,
                    amp=not args.no_amp, batch_size=args.batch_size,
                )
                acc += np.flip(p, axis=flip)
            probs = acc / 4.0
            probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-8)
        gt = np.asarray(lab).astype(np.int32)
        gt_total += int(gt.max())
        cached.append({"sid": sid, "gt": gt, "probs": probs})
    t_inf = time.perf_counter() - t0
    mb = sum(c["probs"].nbytes for c in cached) / 1e6
    print(f"inference: {len(cached)} images in {t_inf:.1f}s "
          f"({'with D4-flip TTA' if args.tta else 'no TTA'}), "
          f"probs cached {mb:.0f} MB | GT instances {gt_total}\n")

    results: Dict[str, Any] = {"ckpt": args.ckpt, "tta": args.tta,
                               "gt_instances": gt_total, "n_images": len(cached)}

    # ---- sweep 1: seed_threshold ------------------------------------------
    print(f"{'seed_thr':>9} {'mAP':>7} {'@.50':>7} {'@.75':>7} {'@.90':>7} "
          f"{'n_pred':>7} {'vs GT':>7}")
    print("-" * 56)
    seed_grid = _grid(args.seed_grid)
    seed_rows = []
    for s in seed_grid:
        r = evaluate(cached, seed_threshold=s, fg_threshold=args.seed_sweep_fg,
                     min_area=args.seed_sweep_area)
        r["seed_threshold"] = s
        seed_rows.append(r)
        delta = 100 * (r["pred_instances"] - gt_total) / gt_total
        star = "  <-- best" if r["map"] == max(x["map"] for x in seed_rows) else ""
        print(f"{s:>9.2f} {r['map']:>7.4f} {r['map_50']:>7.4f} {r['map_75']:>7.4f} "
              f"{r['map_90']:>7.4f} {r['pred_instances']:>7d} {delta:>+6.1f}%{star}")
    best_seed = max(seed_rows, key=lambda r: r["map"])
    results["seed_sweep"] = seed_rows
    results["best_seed"] = best_seed
    print(f"\nbest seed_threshold = {best_seed['seed_threshold']:.2f} "
          f"-> mAP {best_seed['map']:.4f}")

    # ---- sweep 2: fg_threshold at the best seed ----------------------------
    print(f"\n{'fg_thr':>9} {'mAP':>7} {'@.50':>7} {'@.90':>7} {'n_pred':>7}")
    print("-" * 42)
    fg_rows = []
    for f in _grid(args.fg_grid):
        r = evaluate(cached, seed_threshold=best_seed["seed_threshold"],
                     fg_threshold=f, min_area=MIN_AREA)
        r["fg_threshold"] = f
        fg_rows.append(r)
        print(f"{f:>9.2f} {r['map']:>7.4f} {r['map_50']:>7.4f} {r['map_90']:>7.4f} "
              f"{r['pred_instances']:>7d}")
    best_fg = max(fg_rows, key=lambda r: r["map"])
    results["fg_sweep"] = fg_rows
    results["best_fg"] = best_fg

    # ---- sweep 3: min_area -------------------------------------------------
    # NOTE on the 21 px figure: the smallest real GT instance is 21 px, but
    # min_area filters PREDICTED components. A 25 px filter does not
    # automatically destroy a 21 px GT match - it only drops predictions below
    # that size, and a correct prediction of a 21 px nucleus is ~21 px. So the
    # grid may legitimately exceed 21; it is swept, not assumed.
    print(f"\n{'min_area':>9} {'mAP':>7} {'@.50':>7} {'@.90':>7} {'n_pred':>7}")
    print("-" * 42)
    area_rows = []
    for a in [int(x) for x in _grid(args.area_grid)]:
        r = evaluate(cached, seed_threshold=best_seed["seed_threshold"],
                     fg_threshold=best_fg["fg_threshold"], min_area=a)
        r["min_area"] = a
        area_rows.append(r)
        print(f"{a:>9d} {r['map']:>7.4f} {r['map_50']:>7.4f} {r['map_90']:>7.4f} "
              f"{r['pred_instances']:>7d}")
    best_area = max(area_rows, key=lambda r: r["map"])
    results["area_sweep"] = area_rows
    results["best_area"] = best_area

    # ---- final combination -------------------------------------------------
    final = evaluate(cached, seed_threshold=best_seed["seed_threshold"],
                     fg_threshold=best_fg["fg_threshold"],
                     min_area=best_area["min_area"])
    baseline = evaluate(cached, seed_threshold=0.5, fg_threshold=0.5,
                        min_area=MIN_AREA)
    results["baseline"] = baseline
    results["final"] = {**final,
                        "seed_threshold": best_seed["seed_threshold"],
                        "fg_threshold": best_fg["fg_threshold"],
                        "min_area": best_area["min_area"]}

    print("\n" + "=" * 56)
    print(f"baseline (0.50 / 0.50 / {MIN_AREA}): mAP {baseline['map']:.4f}  "
          f"@.50 {baseline['map_50']:.4f}  @.90 {baseline['map_90']:.4f}  "
          f"n={baseline['pred_instances']}")
    print(f"tuned ({best_seed['seed_threshold']:.2f} / "
          f"{best_fg['fg_threshold']:.2f} / {best_area['min_area']}): "
          f"mAP {final['map']:.4f}  @.50 {final['map_50']:.4f}  "
          f"@.90 {final['map_90']:.4f}  n={final['pred_instances']}")
    print(f"gain: {final['map'] - baseline['map']:+.4f} mAP")

    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
