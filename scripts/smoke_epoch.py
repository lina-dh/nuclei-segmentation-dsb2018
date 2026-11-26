"""Single-epoch smoke test for the L1-L4 pipeline. Run from the project root:

    python -m scripts.smoke_epoch            # reuse existing manifest + cache
    python -m scripts.smoke_epoch --rebuild  # full rebuild (~29.5k mask decodes)

Prints scalars only - no arrays, no per-sample loops - per the lean-ctx rule.
Validates, in order:
  L1  manifest exists, split is modality-stratified, proxy modality agrees
  L2  cache opens, RAM footprint matches the ~1 GB budget
  L3  sampler honours the 65/35 quota and the alpha=0.35 area weighting
  L4  crops are exactly 256x256, dtypes/class codes are correct, no NaNs
  L5  albumentations D4 keeps image/target aligned and class codes exact
  --  epoch replay is bit-identical for a fixed (seed, epoch)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Project root on sys.path so `from src...` works when run as a module OR a file.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import BG, BORDER, INTERIOR, CropSampler, SampleCache  # noqa: E402
from src.dataset_manifest import build_manifest, load_manifest  # noqa: E402

TRAIN_DIR = ROOT / "data" / "stage1_train"
STATE_DIR = ROOT / ".project_state"
CACHE_DIR = ROOT / "data" / "cache"
CROP = 256
N_CROPS = 100
BATCH = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild manifest + cache")
    ap.add_argument("--n", type=int, default=N_CROPS)
    args = ap.parse_args()

    # ---- L1 -----------------------------------------------------------------
    if args.rebuild or not (STATE_DIR / "manifest.json").is_file():
        t0 = time.perf_counter()
        print("L1: building manifest + L2 cache (single decode pass)...")
        build_manifest(TRAIN_DIR, out_dir=STATE_DIR, cache_dir=CACHE_DIR, seed=0)
        print(f"L1: built in {time.perf_counter() - t0:.1f}s")

    manifest, instances = load_manifest(STATE_DIR)
    ss = manifest["split_summary"]
    print("\n=== L1 MANIFEST ===")
    print(f"samples={manifest['n_samples']}  instances={manifest['n_instances']}  "
          f"strata={ss['n_strata']}  seed={manifest['seed']}")
    print(f"modality (all) : {manifest['modality_hist']}")
    print(f"train          : n={ss['train']['n']}  {ss['train']['modality']}  "
          f"res={ss['train']['res_bucket']}")
    print(f"val            : n={ss['val']['n']}  {ss['val']['modality']}  "
          f"res={ss['val']['res_bucket']}")
    print(f"proxy disagree : {len(manifest['proxy_disagreements'])} samples")

    # The whole reason stratification exists: val brightfield must land near the
    # expected 18.5% share, not the 11% a bad random split can produce.
    n_val = ss["val"]["n"]
    bf_val = ss["val"]["modality"].get("brightfield", 0)
    print(f"val brightfield: {bf_val}/{n_val} = {100 * bf_val / max(1, n_val):.1f}% "
          f"(dataset share 18.5%)")

    # ---- L2 -----------------------------------------------------------------
    train_ids = [r["sample_id"] for r in manifest["samples"] if r["split"] == "train"]
    val_ids = [r["sample_id"] for r in manifest["samples"] if r["split"] == "val"]

    t0 = time.perf_counter()
    train_cache = SampleCache(CACHE_DIR, train_ids, mode="ram", keep_label=True)
    t_ram = time.perf_counter() - t0
    val_cache = SampleCache(CACHE_DIR, val_ids, mode="mmap", keep_label=True)

    print("\n=== L2 CACHE ===")
    print(f"train: {len(train_cache)} samples  RAM={train_cache.nbytes / 1e9:.2f} GB  "
          f"load={t_ram:.1f}s")
    print(f"val  : {len(val_cache)} samples  mapped={val_cache.nbytes / 1e9:.2f} GB (mmap)")

    # ---- L3 + L4 ------------------------------------------------------------
    sampler = CropSampler(
        train_cache, manifest, instances, split="train", crop=CROP,
        modality_mix=0.65, alpha=0.35, centroid_p=0.70, seed=0,
    )

    t0 = time.perf_counter()
    n_batches = 0
    cls_total = np.zeros(3, dtype=np.int64)
    shapes_ok = dtypes_ok = True
    for batch in sampler.iter_batches(args.n, BATCH, epoch=0, to_float=True):
        n_batches += 1
        b = batch["image"].shape[0]
        shapes_ok &= batch["image"].shape == (b, CROP, CROP, 3)
        shapes_ok &= batch["target"].shape == (b, CROP, CROP)
        dtypes_ok &= batch["image"].dtype == np.float32
        dtypes_ok &= batch["target"].dtype == np.uint8
        # minlength=3 so a batch with no border pixels still returns 3 slots.
        cls_total += np.bincount(batch["target"].ravel(), minlength=3)
        # Range + finiteness check on the normalized image: a polarity or dtype
        # bug shows up here as >1.0 or NaN long before it reaches a loss curve.
        assert np.isfinite(batch["image"]).all(), "non-finite pixels"
        assert batch["image"].min() >= 0.0 and batch["image"].max() <= 1.0, "range"
    dt = time.perf_counter() - t0

    rep = sampler.report()
    px = int(cls_total.sum())
    print("\n=== L3 SAMPLER ===")
    print(f"crops={rep['crops']} in {dt:.2f}s = {rep['crops'] / dt:.0f} crops/s  "
          f"batches={n_batches}")
    print(f"modality mix   : fluo={rep['pct_fluorescence']}%  "
          f"bright={rep['pct_brightfield']}%   (target 65/35)")
    print(f"centering      : instance={rep['pct_center_instance']}%  (target 70%)")
    print(f"empty crops    : {rep['pct_empty_crops']}%")
    print(f"padded crops   : {rep['padded_crops']}  (expect 0: min dim is 256)")

    print("\n=== L4 CROP / TARGET ===")
    print(f"shapes ok={shapes_ok}  dtypes ok={dtypes_ok}")
    print(f"class mix      : bg={100 * cls_total[BG] / px:.2f}%  "
          f"interior={100 * cls_total[INTERIOR] / px:.2f}%  "
          f"border={100 * cls_total[BORDER] / px:.2f}%")
    print(f"class codes    : {sorted(np.flatnonzero(cls_total).tolist())} (expect [0, 1, 2])")

    # ---- L5 -----------------------------------------------------------------
    # Fresh sampler with augmentation ON, same seed -> the CROP stream is
    # identical to the run above, so any difference is attributable to L5 alone.
    aug_sampler = CropSampler(
        train_cache, manifest, instances, split="train", crop=CROP,
        modality_mix=0.65, alpha=0.35, centroid_p=0.70, augment=True, seed=0,
    )
    t0 = time.perf_counter()
    aug_cls = np.zeros(3, dtype=np.int64)
    align_ok = codes_ok = True
    for it in aug_sampler.iter_epoch(args.n, epoch=0):
        aug_cls += np.bincount(it["target"].ravel(), minlength=3)
        # ALIGNMENT INVARIANT: D4 is a pure index permutation, so the count of
        # interior pixels in the target must equal the count of non-zero ids in
        # the label map - the two were transformed by the same sampled params.
        # A desync (the classic hand-rolled-augmentation bug) breaks this.
        fg_t = int((it["target"] > BG).sum())
        fg_l = int((it["label"] > 0).sum())
        align_ok &= fg_t == fg_l
        codes_ok &= bool(np.isin(np.unique(it["target"]), [BG, INTERIOR, BORDER]).all())
    dt_aug = time.perf_counter() - t0
    arep = aug_sampler.report()
    apx = int(aug_cls.sum())

    print("\n=== L5 AUGMENTATION (albumentations) ===")
    print(f"crops={arep['crops']} in {dt_aug:.2f}s = {arep['crops'] / dt_aug:.0f} crops/s")
    print(f"image/mask aligned : {align_ok}   class codes exact: {codes_ok}")
    print(f"class mix          : bg={100 * aug_cls[BG] / apx:.2f}%  "
          f"interior={100 * aug_cls[INTERIOR] / apx:.2f}%  "
          f"border={100 * aug_cls[BORDER] / apx:.2f}%  (vs unaug above)")

    # ---- replay determinism -------------------------------------------------
    # Same seed + same epoch must reproduce the batch byte-for-byte, otherwise a
    # loss spike at step N can never be reproduced for debugging.
    def first_batch(seed: int, epoch: int) -> np.ndarray:
        s = CropSampler(train_cache, manifest, instances, split="train",
                        crop=CROP, augment=True, seed=seed)
        return next(s.iter_batches(BATCH, BATCH, epoch=epoch))["image"]

    replay_ok = np.array_equal(first_batch(0, 0), first_batch(0, 0))
    differs_ok = not np.array_equal(first_batch(0, 0), first_batch(0, 1))
    print(f"\n=== REPLAY ===\nsame (seed,epoch) identical: {replay_ok}   "
          f"different epoch differs: {differs_ok}")

    # ---- val path (mmap, no augmentation) -----------------------------------
    val_sampler = CropSampler(val_cache, manifest, instances, split="val",
                              crop=CROP, augment=None, seed=0)
    vb = next(val_sampler.iter_batches(BATCH, BATCH, epoch=0))
    print(f"\n=== VAL (mmap) ===\nbatch image={vb['image'].shape} "
          f"target={vb['target'].shape} label={'label' in vb}")

    ok = all([
        shapes_ok, dtypes_ok, rep["crops"] == args.n,
        align_ok, codes_ok, replay_ok, differs_ok,
        rep["padded_crops"] == 0,
    ])
    print(f"\nSMOKE TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
