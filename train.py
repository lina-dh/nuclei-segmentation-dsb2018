"""Training loop: U-Net/ResNet34 -> weighted CE + soft dice -> AdamW + cosine, AMP, wandb.

Run on a CUDA machine (from the project root):

    python train.py --epochs 40 --crops-per-epoch 3000 --batch-size 16

Smoke test (no wandb account needed, tiny workload):

    python train.py --epochs 1 --crops-per-epoch 64 --batch-size 8 \
                 --limit-val 6 --wandb-mode offline

VALIDATION IS NOT CROP-BASED. Instance mAP is defined over whole images, so
every val pass tiles each full image with a sliding window, stitches the
probabilities, runs watershed, and scores instances. That makes the val number
directly comparable to a leaderboard submission - and comparable to the 0.9781
representation ceiling measured in scripts/test_metric_watershed.py, which is
the upper bound this run is chasing.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import BG, BORDER, INTERIOR, CropSampler, SampleCache  # noqa: E402
from src.dataset_manifest import load_manifest  # noqa: E402
from src.inference import DEFAULT_STRIDE, sliding_window_probs  # noqa: E402
from src.losses import DEFAULT_CLASS_FREQS, make_loss  # noqa: E402
from src.metrics import dataset_map  # noqa: E402
from src.model import build_model, count_params, param_groups, set_seed  # noqa: E402
from src.postprocess import TUNED, instances_from_probs  # noqa: E402


# =============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- data ----
    p.add_argument("--state-dir", default=".project_state")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--crops-per-epoch", type=int, default=3000,
                   help="epoch length is DECOUPLED from image area by design")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-cache-mode", default="ram", choices=["ram", "mmap"],
                   help="ram = 0.50 GB resident, the hot path; mmap if RAM-tight")
    p.add_argument("--prefetch", type=int, default=2,
                   help="background batch queue depth; 0 disables (order and "
                        "determinism are identical either way)")
    # ---- sampling dials (locked defaults) ----
    p.add_argument("--modality-mix", type=float, default=0.65)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--centroid-p", type=float, default=0.70)
    # ---- model ----
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--norm", default="imagenet",
                   help="MUST be 'imagenet' while the encoder is pretrained")
    # ---- optimization ----
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--encoder-lr-mult", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.03,
                   help="fraction of total steps spent warming up from lr/100")
    p.add_argument("--eta-min-frac", type=float, default=0.01,
                   help="cosine floor as a fraction of lr")
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument("--no-amp", action="store_true")
    # ---- loss ----
    p.add_argument("--weight-scheme", default="inv_sqrt",
                   choices=["none", "inv", "inv_sqrt"])
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    # ---- validation / postprocess ----
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--val-batch-size", type=int, default=8)
    # Defaults are the MEASURED operating point from scripts/sweep_postprocess.py
    # (see src.postprocess.TUNED), not the 0.5/0.5 placeholders: those scored
    # 0.5738 val mAP against 0.6078 for these. Use TUNED_TTA if TTA is enabled.
    p.add_argument("--seed-threshold", type=float, default=TUNED["seed_threshold"],
                   help="interior prob to seed watershed; THE over/under-seg dial")
    p.add_argument("--fg-threshold", type=float, default=TUNED["fg_threshold"])
    p.add_argument("--min-area", type=int, default=TUNED["min_area"])
    p.add_argument("--limit-val", type=int, default=0,
                   help="score only the first N val images (smoke tests)")
    # ---- logging ----
    p.add_argument("--wandb-project", default="nuclei-dsb2018")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--log-every", type=int, default=20, help="steps between loss logs")
    p.add_argument("--n-vis", type=int, default=4, help="val images in the wandb grid")
    # ---- bookkeeping ----
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


# =============================================================================
def prefetch(iterable, depth: int = 2):
    """Run `iterable` on a background thread, yielding items in the SAME order.

    Why: the GPU sustains ~155 crops/s while the sampler produces ~300 crops/s,
    but running them serially in one thread gives only their harmonic
    combination (~100 crops/s) because each waits for the other. Overlapping
    them lets the GPU run at its own rate while the next batch is being built.

    Why a thread and not a process: the sampler's work is almost entirely NumPy,
    OpenCV and albumentations, all of which release the GIL inside their C
    routines - so a thread genuinely overlaps with the Python-side training
    loop. A process would need to pickle the whole RAM cache per worker.

    DETERMINISM IS PRESERVED: a single producer thread consumes the generator in
    order and a FIFO queue returns items in that same order, so the batch
    sequence is byte-identical to the un-prefetched path. `depth` bounds the
    queue, so memory stays at `depth` batches instead of the whole epoch.
    """
    import queue
    import threading

    if depth <= 0:
        yield from iterable
        return

    q: "queue.Queue" = queue.Queue(maxsize=depth)
    # A unique sentinel object, not None: None could in principle be a value.
    done = object()
    err: List[BaseException] = []

    def worker():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as e:   # noqa: BLE001 - re-raised on the main thread
            err.append(e)
        finally:
            q.put(done)

    # daemon=True so a crash in the main thread cannot leave this hanging.
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is done:
            break
        yield item
    t.join()
    if err:
        # Surface the producer's exception on the consumer thread, where the
        # traceback is actually visible.
        raise err[0]


def build_scheduler(optimizer, total_steps: int, args) -> Any:
    """Linear warmup -> cosine annealing, stepped PER ITERATION.

    Warmup exists because of the pretrained encoder: AdamW's first few updates
    are large and poorly conditioned (the second-moment estimate has almost no
    history), and a large step into ImageNet weights destroys the features we
    are paying for. Ramping from lr/100 avoids that.

    Cosine is stepped per iteration, not per epoch, so the schedule is smooth
    and independent of how many crops we choose to call an "epoch" - which
    matters here precisely because epoch length is an arbitrary dial.

    WHY LambdaLR AND NOT SequentialLR(LinearLR, CosineAnnealingLR):
    CosineAnnealingLR takes a single scalar ``eta_min`` shared by every param
    group. With ``--encoder-lr-mult 0.1`` the encoder and decoder start 10x
    apart but both floor at the SAME absolute eta_min, so the ratio silently
    drifts to 1.0 by the end of training - the flag stops doing what it says.
    LambdaLR instead applies one MULTIPLICATIVE factor to each group's own
    base_lr, so every group keeps its ratio and floors at
    ``base_lr_group * eta_min_frac``.
    """
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    floor = args.eta_min_frac

    def factor(step: int) -> float:
        if step < warmup_steps:
            # Linear ramp from 1% to 100% of each group's base lr.
            return 0.01 + (1.0 - 0.01) * (step / warmup_steps)
        # Cosine from 1.0 down to `floor` over the remaining steps.
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


# =============================================================================
@torch.no_grad()
def validate(
    model, val_cache, val_records, loss_fn, args, device: str,
) -> Dict[str, Any]:
    """Full-image sliding-window validation: val loss + instance mAP.

    Returns scalars plus the per-sample predictions needed for the image grid.
    """
    model.eval()
    # val_records is ALREADY the evaluated subset (see main): limiting here too
    # would desync the scored set from the visualization set.
    recs = val_records

    pairs = []
    ce_sum = dice_sum = 0.0
    n_img = 0
    vis: Dict[str, Any] = {}
    t0 = time.perf_counter()

    for r in recs:
        sid = r["sample_id"]
        img, tgt, lab = val_cache.get(sid)
        img = np.asarray(img)
        tgt = np.asarray(tgt)
        gt = np.asarray(lab).astype(np.int32)

        probs = sliding_window_probs(
            model, img, crop=args.crop, stride=args.val_stride,
            norm=args.norm, device=device, amp=not args.no_amp,
            batch_size=args.val_batch_size,
        )

        # FULL-IMAGE val loss. log(p) is a valid logit vector for
        # cross_entropy: log_softmax(log p) == log p when p is normalized, so
        # this reproduces CE exactly without needing the raw tile logits.
        lp = torch.from_numpy(np.log(np.maximum(probs, 1e-8))).unsqueeze(0).to(device)
        tt = torch.from_numpy(tgt.astype(np.int64)).unsqueeze(0).to(device)
        _, parts = loss_fn(lp, tt)
        ce_sum += float(parts["ce"])
        dice_sum += float(parts["dice"])
        n_img += 1

        pred = instances_from_probs(
            probs, seed_threshold=args.seed_threshold,
            fg_threshold=args.fg_threshold, min_area=args.min_area,
        )
        pairs.append((sid, gt, pred))

        if sid in args._vis_ids:
            vis[sid] = {"image": img, "gt": gt, "pred": pred,
                        "modality": r["modality"]}

    d = dataset_map(pairs)
    return {
        "val/map": d["map"],
        "val/map_50": d["per_threshold"][0]["mean_precision"],
        "val/map_75": d["per_threshold"][5]["mean_precision"],
        "val/map_90": d["per_threshold"][8]["mean_precision"],
        "val/ce": ce_sum / max(n_img, 1),
        "val/dice": dice_sum / max(n_img, 1),
        "val/n_images": n_img,
        "val/pred_instances": int(sum(p.max() for _, _, p in pairs)),
        "val/gt_instances": int(sum(g.max() for _, g, _ in pairs)),
        "val/seconds": round(time.perf_counter() - t0, 1),
        "_worst": d["worst"],
        "_vis": vis,
    }


def make_grid_image(vis: Dict[str, Any], side: int = 256) -> Optional[np.ndarray]:
    """Compose [original | GT instances | predicted instances] rows into one image.

    Instance ids are mapped to colors with skimage's label2rgb, which assigns a
    distinct hue per id - so a merged pair shows as one blob in a single color
    and a split instance shows as two colors. That is the fastest visual read on
    whether watershed is over- or under-segmenting.

    Everything is resized to a fixed side for DISPLAY ONLY (val images range
    from 256x256 to 1024x1024 and a grid needs uniform tiles). INTER_NEAREST for
    label maps - interpolating ids would invent nonexistent instances.
    """
    import cv2
    from skimage.color import label2rgb

    if not vis:
        return None

    rows = []
    for sid, d in vis.items():
        img = cv2.resize(d["image"], (side, side), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(d["gt"], (side, side), interpolation=cv2.INTER_NEAREST)
        pr = cv2.resize(d["pred"], (side, side), interpolation=cv2.INTER_NEAREST)
        # bg_label=0 keeps background black instead of giving it a hue.
        gt_rgb = (label2rgb(gt, bg_label=0) * 255).astype(np.uint8)
        pr_rgb = (label2rgb(pr, bg_label=0) * 255).astype(np.uint8)
        # A 2px white separator makes the three panels readable at thumbnail size.
        sep = np.full((side, 2, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([img, sep, gt_rgb, sep, pr_rgb], axis=1))

    hsep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, hsep, r], axis=0)
    return out


# =============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA. AMP disabled; this will be very slow.")
        args.no_amp = True

    set_seed(args.seed)
    state_dir = ROOT / args.state_dir
    out_dir = ROOT / args.out_dir / (args.wandb_run or time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ---------------------------------------------------------------
    manifest, instances = load_manifest(state_dir)
    train_recs = [r for r in manifest["samples"] if r["split"] == "train"]
    val_recs = [r for r in manifest["samples"] if r["split"] == "val"]
    train_ids = [r["sample_id"] for r in train_recs]
    val_ids = [r["sample_id"] for r in val_recs]

    cache_dir = ROOT / args.cache_dir
    # Labels are not needed for training (the 3-class target is), so dropping
    # them saves ~1/3 of the resident cache.
    train_cache = SampleCache(cache_dir, train_ids, mode=args.train_cache_mode,
                              keep_label=False)
    val_cache = SampleCache(cache_dir, val_ids, mode="mmap", keep_label=True)

    sampler = CropSampler(
        train_cache, manifest, instances, split="train", crop=args.crop,
        modality_mix=args.modality_mix, alpha=args.alpha,
        centroid_p=args.centroid_p, augment=True, norm=args.norm, seed=args.seed,
    )

    # The subset actually scored each epoch. --limit-val exists for smoke tests;
    # a real run scores all 135.
    val_eval_recs = val_recs[:args.limit_val] if args.limit_val else val_recs

    # Fixed visualization set, chosen deterministically from the SCORED subset
    # and spanning BOTH modalities: brightfield is only 18.5% of val and is the
    # half most likely to fail silently, so it must be on screen every epoch.
    half = max(1, args.n_vis // 2)
    vis_ids = (
        [r["sample_id"] for r in val_eval_recs if r["modality"] == "fluorescence"][:half]
        + [r["sample_id"] for r in val_eval_recs if r["modality"] == "brightfield"][:half]
    )
    # Fallback: a --limit-val subset may contain only one modality.
    if not vis_ids:
        vis_ids = [r["sample_id"] for r in val_eval_recs[:args.n_vis]]
    args._vis_ids = set(vis_ids)

    # ---- model / loss / optim ----------------------------------------------
    model = build_model(args.encoder, classes=3,
                        encoder_weights=args.encoder_weights).to(device)
    pinfo = count_params(model)

    freqs = DEFAULT_CLASS_FREQS
    cls_path = state_dir / "class_stats.json"
    if cls_path.is_file():
        with open(cls_path, encoding="utf-8") as f:
            freqs = json.load(f)["freqs"]

    loss_fn = make_loss(
        freqs, scheme=args.weight_scheme, ce_weight=args.ce_weight,
        dice_weight=args.dice_weight, label_smoothing=args.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        param_groups(model, args.lr, args.weight_decay, args.encoder_lr_mult),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = int(np.ceil(args.crops_per_epoch / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, args)
    # enabled=False makes GradScaler a transparent no-op, so the training loop
    # needs no if/else around the AMP calls.
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    start_epoch, best_map = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    # ---- wandb --------------------------------------------------------------
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project, name=args.wandb_run, mode=args.wandb_mode,
            dir=str(out_dir),
            config={
                **{k: v for k, v in vars(args).items() if not k.startswith("_")},
                "params": pinfo, "class_freqs": list(freqs),
                "n_train": len(train_ids), "n_val": len(val_ids),
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                "ceiling_map": 0.9781,
            },
        )

    print(f"device={device} | params={pinfo} | train={len(train_ids)} val={len(val_ids)}")
    print(f"steps/epoch={steps_per_epoch} total={total_steps} | amp={not args.no_amp}")
    print(f"class_freqs={[round(f, 4) for f in freqs]} scheme={args.weight_scheme}")

    # ---- train --------------------------------------------------------------
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_ep = time.perf_counter()
        run_ce = run_dice = run_total = 0.0
        n_steps = 0

        for batch in prefetch(sampler.iter_batches(
            args.crops_per_epoch, args.batch_size, epoch=epoch,
            to_float=True, channels_first=True,
        ), depth=args.prefetch):
            x = torch.from_numpy(batch["image"]).to(device, non_blocking=True)
            y = torch.from_numpy(batch["target"].astype(np.int64)).to(device, non_blocking=True)
            ig = torch.from_numpy(batch["ignore"]).to(device, non_blocking=True)

            # set_to_none=True frees the grad buffers instead of zeroing them:
            # slightly faster and it surfaces "forgot to backward" as a None.
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=not args.no_amp):
                logits = model(x)
                total, parts = loss_fn(logits, y, ignore=ig)

            scaler.scale(total).backward()
            if args.grad_clip > 0:
                # unscale_ FIRST: clipping scaled gradients would clip at the
                # wrong magnitude (the scale factor is ~2^16).
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # Scheduler steps per ITERATION, matching build_scheduler's T_max.
            scheduler.step()

            run_ce += float(parts["ce"])
            run_dice += float(parts["dice"])
            run_total += float(parts["total"])
            n_steps += 1
            global_step += 1

            if run is not None and global_step % args.log_every == 0:
                run.log({
                    "train/loss": float(parts["total"]),
                    "train/ce": float(parts["ce"]),
                    "train/dice": float(parts["dice"]),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }, step=global_step)

        train_log = {
            "train/loss_epoch": run_total / max(n_steps, 1),
            "train/ce_epoch": run_ce / max(n_steps, 1),
            "train/dice_epoch": run_dice / max(n_steps, 1),
            "train/seconds": round(time.perf_counter() - t_ep, 1),
            "train/crops_per_s": round(args.crops_per_epoch /
                                       max(time.perf_counter() - t_ep, 1e-9), 1),
        }

        # ---- validate -------------------------------------------------------
        val_log: Dict[str, Any] = {}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            v = validate(model, val_cache, val_eval_recs, loss_fn, args, device)
            vis, worst = v.pop("_vis"), v.pop("_worst")
            val_log = v

            if run is not None:
                import wandb
                grid = make_grid_image(vis)
                if grid is not None:
                    run.log({"val/examples": wandb.Image(
                        grid,
                        caption="rows: " + ", ".join(
                            f"{s[:8]}({vis[s]['modality'][:5]})" for s in vis
                        ) + " | cols: image | GT instances | predicted (watershed)",
                    )}, step=global_step)

            if val_log["val/map"] > best_map:
                best_map = val_log["val/map"]
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_map": best_map, "args": vars(args),
                }, out_dir / "best.pt")

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_map": best_map,
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        }, out_dir / "last.pt")

        merged = {**train_log, **val_log, "epoch": epoch, "best/map": best_map}
        if run is not None:
            run.log(merged, step=global_step)
        # Full dotted keys, NOT the last path segment: train/seconds and
        # val/seconds both end in "seconds", and val/map vs best/map both end in
        # "map", so stripping the prefix printed duplicate-looking fields.
        print(f"[epoch {epoch + 1}/{args.epochs}] "
              + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in merged.items() if not k.startswith("_")))

    if run is not None:
        run.summary["best/map"] = best_map
        run.finish()
    print(f"\nbest val mAP = {best_map:.4f}  (ceiling 0.9781) | ckpt: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
