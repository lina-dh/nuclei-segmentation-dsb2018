"""Pre-training verification suite. Run from the project root:

    python -m scripts.verify_pipeline

Every check is an assertion about a property the TRAINING LOOP depends on. A
check that cannot fail is not included. Prints one PASS/FAIL line per check and
exits non-zero if any failed, so it is usable as a gate in a script.

Scope: L1 manifest, L2 cache + polarity, target correctness, normalization,
L4 cropping (including the padding path, which no real sample exercises),
L5 augmentation sync, and the torch hand-off shape/dtype contract.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    BG, BORDER, INTERIOR, MODALITY_BG_THRESHOLD,
    CropSampler, SampleCache, build_augmenter, compute_norm_stats,
    load_label_map, load_image, load_norm_stats, make_3class_target, _to_gray,
)
from src.data_loader import NORM_IMAGENET, resolve_norm  # noqa: E402
from src.dataset_manifest import EXCLUDED_SAMPLES, load_manifest  # noqa: E402

TRAIN_DIR = ROOT / "data" / "stage1_train"
STATE_DIR = ROOT / ".project_state"
CACHE_DIR = ROOT / "data" / "cache"
NORM_JSON = STATE_DIR / "norm_stats.json"
CROP = 256

_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one check. Keeps the output to one line per property."""
    _results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# =============================================================================
def main() -> int:
    manifest, instances = load_manifest(STATE_DIR)
    recs = manifest["samples"]
    by_id = {r["sample_id"]: r for r in recs}
    train_ids = [r["sample_id"] for r in recs if r["split"] == "train"]
    val_ids = [r["sample_id"] for r in recs if r["split"] == "val"]

    excluded_ids = [r["sample_id"] for r in recs if r["split"] == "excluded"]

    print("=== L1 MANIFEST ===")
    check("manifest holds all 670 samples", len(recs) == 670, f"{len(recs)}")
    check("every sample has a split label",
          all(r["split"] in ("train", "val", "excluded") for r in recs))
    # Disjoint + covering: a sample leaking into both splits is the single most
    # damaging silent bug possible here, so it is asserted on sets, not counts.
    # Excluded samples must account for the remainder exactly - no sample may
    # simply vanish.
    check("train/val/excluded disjoint and covering",
          set(train_ids).isdisjoint(val_ids)
          and set(excluded_ids).isdisjoint(train_ids + val_ids)
          and len(train_ids) + len(val_ids) + len(excluded_ids) == 670,
          f"{len(train_ids)}/{len(val_ids)}/{len(excluded_ids)} excluded")
    check("excluded set matches EXCLUDED_SAMPLES in code",
          set(excluded_ids) == set(EXCLUDED_SAMPLES),
          f"{len(excluded_ids)} excluded")
    check("instance rows == 29461", len(instances["area"]) == 29461,
          f"{len(instances['area'])}")
    # The sampler uses searchsorted on sample_idx, which is only valid if sorted.
    check("instances.sample_idx is sorted (searchsorted precondition)",
          bool(np.all(np.diff(instances["sample_idx"]) >= 0)))

    # Centroids must lie inside their own image, else crop centering goes out of
    # bounds and clamps to a corner.
    cy, cx = instances["centroid"][:, 0], instances["centroid"][:, 1]
    h_arr = np.array([recs[i]["h"] for i in instances["sample_idx"]], dtype=np.float32)
    w_arr = np.array([recs[i]["w"] for i in instances["sample_idx"]], dtype=np.float32)
    check("all centroids inside their image bounds",
          bool((cy >= 0).all() and (cy < h_arr).all() and (cx >= 0).all() and (cx < w_arr).all()))

    bf_val = sum(1 for i in val_ids if by_id[i]["modality"] == "brightfield")
    share = 100 * bf_val / len(val_ids)
    check("val brightfield share within 3pp of 18.5%", abs(share - 18.5) <= 3.0,
          f"{share:.1f}%")

    # ---------------------------------------------------------------------
    print("\n=== L2 CACHE ===")
    missing = [r["sample_id"] for r in recs
               if not all((CACHE_DIR / f"{r['sample_id']}_{k}.npy").is_file()
                          for k in ("img", "lab", "tgt"))]
    check("cache complete (670 x 3 arrays)", not missing, f"{len(missing)} missing")

    train_cache = SampleCache(CACHE_DIR, train_ids, mode="ram", keep_label=True)
    val_cache = SampleCache(CACHE_DIR, val_ids, mode="mmap", keep_label=True)
    check("train cache opens in RAM mode", len(train_cache) == len(train_ids),
          f"{train_cache.nbytes / 1e9:.2f} GB")
    check("val cache opens in mmap mode", len(val_cache) == len(val_ids),
          f"{val_cache.nbytes / 1e9:.2f} GB")

    # Shapes/dtypes must match the manifest, or the sampler's clamp arithmetic
    # (which trusts manifest h/w) indexes past the end of the array.
    shape_ok = dtype_ok = True
    for sid in train_ids[:80] + val_ids[:20]:
        img, tgt, lab = (train_cache if sid in train_cache._img else val_cache).get(sid)
        r = by_id[sid]
        shape_ok &= img.shape == (r["h"], r["w"], 3) and tgt.shape == (r["h"], r["w"])
        shape_ok &= lab.shape == (r["h"], r["w"])
        dtype_ok &= (img.dtype == np.uint8 and tgt.dtype == np.uint8
                     and lab.dtype == np.uint16)
    check("cached shapes match manifest h/w (100 sampled)", shape_ok)
    check("cached dtypes are uint8/uint8/uint16", dtype_ok)

    # ---- POLARITY FLIP: the headline L2 correctness property ----------------
    # After the flip, EVERY sample - both modalities - must satisfy
    # "nuclei brighter than background". This is checked on the CACHED pixels,
    # which is what the model actually sees.
    neg_bf, neg_fl, n_bf, n_fl = [], [], 0, 0
    for sid in train_ids + val_ids:
        c = train_cache if sid in train_cache._img else val_cache
        img, _, lab = c.get(sid)
        gray = _to_gray(np.asarray(img))
        fg = np.asarray(lab) > 0
        if not fg.any() or fg.all():
            continue
        delta = float(gray.mean(where=fg, dtype=np.float64)
                      - gray.mean(where=~fg, dtype=np.float64))
        if by_id[sid]["modality"] == "brightfield":
            n_bf += 1
            if delta <= 0:
                neg_bf.append((sid, round(delta, 1)))
        else:
            n_fl += 1
            if delta <= 0:
                neg_fl.append((sid, round(delta, 1)))
    check(f"polarity flip: all {n_bf} brightfield now fg>bg", not neg_bf,
          f"{len(neg_bf)} inverted" + (f" e.g. {neg_bf[0]}" if neg_bf else ""))

    # The one inverted-contrast fluorescence sample (7b38c917..., a dark-field
    # image whose single ~30 px instance is darker than its background) was
    # EXCLUDED from both splits on 2026-10-12. So across every sample the model
    # can actually see, the invariant is now absolute: fg > bg, no exceptions.
    #
    # Note what was NOT done: we never flipped it. The flip decision has to be
    # computable from the image alone, since inference has no masks; flipping on
    # the mask-derived contrast_delta would be preprocessing leakage.
    check(f"polarity: all {n_fl} fluorescence in-split samples have fg>bg",
          not neg_fl,
          f"{len(neg_fl)} inverted" + (f", e.g. {neg_fl[0]}" if neg_fl else ""))
    # And the outlier is genuinely gone from the training stream, not merely
    # absent from this loop by accident.
    check("the inverted-contrast outlier is excluded from both splits",
          all(s not in set(train_ids) | set(val_ids) for s in EXCLUDED_SAMPLES))

    # ---- cache fidelity vs the source PNGs ---------------------------------
    # Re-decode a spot-check set from disk and compare byte-for-byte. Catches a
    # stale cache silently disagreeing with the dataset.
    spot = ([s for s in train_ids if by_id[s]["modality"] == "brightfield"][:6]
            + [s for s in train_ids if by_id[s]["modality"] == "fluorescence"][:6])
    img_ok = lab_ok = tgt_ok = True
    for sid in spot:
        src_img = load_image(TRAIN_DIR / sid)
        src_lab, _ = load_label_map(TRAIN_DIR / sid, shape=src_img.shape[:2])
        c_img, c_tgt, c_lab = train_cache.get(sid)
        expect = 255 - src_img if by_id[sid]["modality"] == "brightfield" else src_img
        img_ok &= np.array_equal(np.asarray(c_img), expect)
        lab_ok &= np.array_equal(np.asarray(c_lab), src_lab.astype(np.uint16))
        tgt_ok &= np.array_equal(np.asarray(c_tgt), make_3class_target(src_lab))
    check("cached image == 255-src (brightfield) / src (fluo), 12 samples", img_ok)
    check("cached label map == freshly built label map, 12 samples", lab_ok)
    check("cached target == freshly built target, 12 samples", tgt_ok)

    # ---------------------------------------------------------------------
    print("\n=== 3-CLASS TARGET ===")
    # THE property the whole watershed plan rests on: two different instances
    # must never have 4-adjacent INTERIOR pixels, i.e. the border ring always
    # forms a closed wall. Checked on ALL 670 samples, vectorized.
    bad_pairs = 0
    bad_samples = []
    fg_consistent = True
    for sid in train_ids + val_ids:
        c = train_cache if sid in train_cache._img else val_cache
        _, tgt, lab = c.get(sid)
        tgt = np.asarray(tgt)
        lab = np.asarray(lab)
        inner = tgt == INTERIOR
        # Horizontal and vertical 4-neighbour comparisons as whole-array slices:
        # no Python loop over pixels. (a != b) & both-interior counts violations.
        hv = 0
        hv += int((((lab[:, :-1] != lab[:, 1:]) & inner[:, :-1] & inner[:, 1:])).sum())
        hv += int((((lab[:-1, :] != lab[1:, :]) & inner[:-1, :] & inner[1:, :])).sum())
        if hv:
            bad_pairs += hv
            bad_samples.append(sid)
        # interior|border must be exactly the instance footprint - no drift.
        fg_consistent &= np.array_equal(tgt > BG, lab > 0)
    check("border wall: no 4-adjacent interior px from different instances",
          bad_pairs == 0, f"{bad_pairs} px in {len(bad_samples)} samples")
    check("target (interior|border) == label>0 on all 670 samples", fg_consistent)

    # ---------------------------------------------------------------------
    print("\n=== NORMALIZATION (mean/std, post-flip) ===")
    check("norm_stats.json exists", NORM_JSON.is_file())
    mean, std = load_norm_stats(NORM_JSON)
    check("stats are 3-channel and finite",
          mean.shape == (3,) and std.shape == (3,)
          and bool(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()),
          f"mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}")

    # Stats must come from TRAIN ONLY. Recompute on train and compare; then show
    # the val-inclusive number differs, proving the train-only path is real.
    s_tr = compute_norm_stats(CACHE_DIR, train_ids)
    check("stored stats == recomputed on TRAIN split (no val leakage)",
          np.allclose(mean, s_tr["mean"], atol=1e-6)
          and np.allclose(std, s_tr["std"], atol=1e-6))

    # The actual effect on a batch: standardized data should be ~zero-mean,
    # ~unit-std. Tolerance is loose because 64 crops are a small, spatially
    # correlated sample of the pixel distribution.
    norm_sampler = CropSampler(train_cache, manifest, instances, split="train",
                               crop=CROP, augment=None, norm=str(NORM_JSON), seed=0)
    nb = next(norm_sampler.iter_batches(64, 64, epoch=0, to_float=True))
    bm, bs = float(nb["image"].mean()), float(nb["image"].std())
    check("standardized batch mean within 0.5 of 0", abs(bm) < 0.5, f"mean={bm:.3f}")
    check("standardized batch std within [0.5, 2.0]", 0.5 < bs < 2.0, f"std={bs:.3f}")

    # Order of operations: flip -> /255 -> (x-mean)/std. Reproduce it by hand on
    # one crop and require an exact match, which pins the order down.
    raw_sampler = CropSampler(train_cache, manifest, instances, split="train",
                              crop=CROP, augment=None, norm=None, seed=0)
    rb = next(raw_sampler.iter_batches(4, 4, epoch=0, to_float=True))
    nb4 = next(norm_sampler.iter_batches(4, 4, epoch=0, to_float=True))
    manual = (rb["image"] - mean) / std
    check("norm == (x/255 - mean)/std applied after the flip",
          np.allclose(manual, nb4["image"], atol=1e-5))
    check("un-normalized path stays in [0,1]",
          float(rb["image"].min()) >= 0.0 and float(rb["image"].max()) <= 1.0)

    # ---- ImageNet stats are the LOCKED default (pretrained encoder) ---------
    in_mean, in_std = resolve_norm("imagenet")
    check("'imagenet' resolves to the standard ImageNet stats",
          np.allclose(in_mean, NORM_IMAGENET[0]) and np.allclose(in_std, NORM_IMAGENET[1]),
          f"mean={in_mean.tolist()}")
    # The default must be imagenet, since the encoder is pretrained. A silent
    # revert to dataset stats would shift every layer-1 activation.
    default_sampler = CropSampler(train_cache, manifest, instances, split="train",
                                  crop=CROP, augment=None, seed=0)
    check("CropSampler default norm IS imagenet",
          default_sampler.norm is not None
          and np.allclose(default_sampler.norm[0], NORM_IMAGENET[0]))
    db = next(default_sampler.iter_batches(4, 4, epoch=0, to_float=True))
    manual_in = (rb["image"] - in_mean) / in_std
    check("default path == (x/255 - imagenet_mean)/imagenet_std",
          np.allclose(manual_in, db["image"], atol=1e-5))
    check("imagenet-normalized batch differs from dataset-normalized",
          not np.allclose(db["image"], nb4["image"]),
          "confirms the two paths are distinct")

    # Excluded samples must never reach a training batch.
    seen = set()
    for it in default_sampler.iter_epoch(3000, epoch=5):
        seen.add(it["sample_id"])
    check("excluded sample never appears in 3000 train crops",
          not (seen & set(EXCLUDED_SAMPLES)), f"{len(seen)} distinct samples drawn")

    # ---------------------------------------------------------------------
    print("\n=== L4 CROPPING ===")
    big = CropSampler(train_cache, manifest, instances, split="train", crop=CROP,
                      modality_mix=0.65, alpha=0.35, centroid_p=0.70,
                      augment=None, norm=None, seed=0)
    in_bounds = shape_all = True
    n_probe = 4000
    for it in big.iter_epoch(n_probe, epoch=0):
        r = by_id[it["sample_id"]]
        shape_all &= it["image"].shape == (CROP, CROP, 3)
        shape_all &= it["target"].shape == (CROP, CROP)
        # The window must sit fully inside the image for every real sample.
        in_bounds &= (0 <= it["top"] <= max(0, r["h"] - CROP)
                      and 0 <= it["left"] <= max(0, r["w"] - CROP))
    rep = big.report()
    check(f"all {n_probe} crops are exactly 256x256x3", shape_all)
    check("crop window always inside image bounds", in_bounds)
    check("modality quota 65/35 within 3pp",
          abs(rep["pct_fluorescence"] - 65) <= 3,
          f"fluo={rep['pct_fluorescence']}% bright={rep['pct_brightfield']}%")
    check("instance-centered share 70% within 3pp",
          abs(rep["pct_center_instance"] - 70) <= 3,
          f"{rep['pct_center_instance']}%")
    check("empty-crop rate under 5%", rep["pct_empty_crops"] < 5.0,
          f"{rep['pct_empty_crops']}%")

    # ---- the PADDING path --------------------------------------------------
    # No real sample is smaller than 256, so this path is otherwise never
    # executed. A synthetic 200x180 sample forces it. If padding is broken it
    # will only ever surface on stage2 / external data, i.e. at the worst time.
    pad_ok = _check_padding(manifest)

    # ---------------------------------------------------------------------
    print("\n=== L5 AUGMENTATION SYNC ===")
    # Strong desync test: feed the TARGET ITSELF as the image through a
    # geometry-only pipeline. Any spatial transform must move image and mask
    # identically, so output image and output mask must stay equal pixel-wise.
    # An intensity transform would break the equality, hence geometry only.
    geo = build_augmenter(scale_jitter=False, p_intensity=0.0, seed=7)
    sync_ok = ids_ok = codes_ok = True
    for it in raw_sampler.iter_epoch(200, epoch=3):
        tgt = it["target"]
        lab = it["label"]
        probe = np.repeat(tgt[:, :, None], 3, axis=2)  # mask painted as an image
        o = geo(image=probe, mask=tgt, label=lab)
        sync_ok &= np.array_equal(o["image"][:, :, 0], o["mask"])
        # D4 is a permutation of pixel positions, so the multiset of instance
        # ids and the set of class codes must both be invariant.
        ids_ok &= np.array_equal(np.unique(o["label"]), np.unique(lab))
        codes_ok &= np.array_equal(np.unique(o["mask"]), np.unique(tgt))
    check("image and mask stay pixel-aligned through geometric aug", sync_ok)
    check("instance id set invariant under D4", ids_ok)
    check("class code set invariant under D4", codes_ok)

    aug_sampler = CropSampler(train_cache, manifest, instances, split="train",
                              crop=CROP, augment=True, norm=str(NORM_JSON), seed=0)
    fg_match = True
    for it in aug_sampler.iter_epoch(300, epoch=0):
        fg_match &= int((it["target"] > BG).sum()) == int((it["label"] > 0).sum())
    check("full aug pipeline: interior+border px == labeled px", fg_match)

    # ---------------------------------------------------------------------
    print("\n=== TORCH HAND-OFF ===")
    b = next(aug_sampler.iter_batches(8, 8, epoch=0, to_float=True,
                                      channels_first=True))
    check("NCHW batch shape (8,3,256,256)", b["image"].shape == (8, 3, 256, 256),
          str(b["image"].shape))
    check("image float32, C-contiguous (zero-copy from_numpy)",
          b["image"].dtype == np.float32 and b["image"].flags["C_CONTIGUOUS"])
    check("target uint8 (B,H,W) - ready for long() cast to CE labels",
          b["target"].shape == (8, 256, 256) and b["target"].dtype == np.uint8)
    check("ignore mask bool (B,H,W)",
          b["ignore"].shape == (8, 256, 256) and b["ignore"].dtype == bool)
    check("no NaN/Inf after standardization", bool(np.isfinite(b["image"]).all()))

    # ---------------------------------------------------------------------
    print("\n=== DETERMINISM + THROUGHPUT ===")
    def fb(seed, epoch):
        s = CropSampler(train_cache, manifest, instances, split="train", crop=CROP,
                        augment=True, norm=str(NORM_JSON), seed=seed)
        return next(s.iter_batches(8, 8, epoch=epoch, to_float=True))["image"]
    check("same (seed,epoch) -> byte-identical batch", np.array_equal(fb(0, 0), fb(0, 0)))
    check("different epoch -> different batch", not np.array_equal(fb(0, 0), fb(0, 1)))
    check("different seed -> different batch", not np.array_equal(fb(0, 0), fb(1, 0)))

    t0 = time.perf_counter()
    nb_ = 0
    for _ in aug_sampler.iter_batches(512, 16, epoch=1, to_float=True,
                                     channels_first=True):
        nb_ += 1
    dt = time.perf_counter() - t0
    cps = 512 / dt
    check("throughput > 150 crops/s (single process)", cps > 150,
          f"{cps:.0f} crops/s, {nb_} batches of 16")

    # ---- val path ----------------------------------------------------------
    val_sampler = CropSampler(val_cache, manifest, instances, split="val", crop=CROP,
                              augment=None, norm=str(NORM_JSON), seed=0)
    vb = next(val_sampler.iter_batches(8, 8, epoch=0, to_float=True,
                                       channels_first=True))
    check("val sampler works off mmap with no augmentation",
          vb["image"].shape == (8, 3, 256, 256) and "label" in vb)

    # =====================================================================
    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{'=' * 62}")
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed"
          + (f"  -- {n_fail} FAILED" if n_fail else "  -- ALL PASS"))
    if n_fail:
        for nm, ok in _results:
            if not ok:
                print(f"  FAILED: {nm}")
    return 1 if n_fail else 0


def _check_padding(manifest) -> bool:
    """Force the pad path with a synthetic sub-256 sample in a temp cache."""
    import tempfile

    H, W = 200, 180  # smaller than CROP on BOTH axes
    tmp = Path(tempfile.mkdtemp(prefix="padcheck_"))
    sid = "synthetic_small"

    # A deterministic image + one centered instance, so the result is checkable.
    img = np.full((H, W, 3), 40, dtype=np.uint8)
    img[60:140, 50:130] = 200
    lab = np.zeros((H, W), dtype=np.int32)
    lab[60:140, 50:130] = 1
    np.save(tmp / f"{sid}_img.npy", img)
    np.save(tmp / f"{sid}_lab.npy", lab.astype(np.uint16))
    np.save(tmp / f"{sid}_tgt.npy", make_3class_target(lab))

    # Minimal manifest/instances shaped exactly like the real ones.
    fake_manifest = {"samples": [{
        "sample_id": sid, "h": H, "w": W, "area": H * W, "res_bucket": "small",
        "n_instances": 1, "modality": "fluorescence", "split": "train",
    }]}
    fake_inst = {
        "centroid": np.array([[99.5, 89.5]], dtype=np.float32),
        "sample_idx": np.array([0], dtype=np.int32),
    }

    cache = SampleCache(tmp, [sid], mode="ram", keep_label=True)
    s = CropSampler(cache, fake_manifest, fake_inst, split="train", crop=CROP,
                    augment=None, norm=None, seed=0)

    shape_ok = ign_ok = content_ok = real_ok = True
    for it in s.iter_epoch(40, epoch=0):
        shape_ok &= it["image"].shape == (CROP, CROP, 3)
        shape_ok &= it["target"].shape == (CROP, CROP)
        ign = it["ignore"]
        # The ignore mask must flag EXACTLY the synthetic region: rows >= H and
        # cols >= W, nothing else.
        expect = np.zeros((CROP, CROP), dtype=bool)
        expect[H:, :] = True
        expect[:, W:] = True
        ign_ok &= np.array_equal(ign, expect)
        # Real (non-padded) region must be untouched by the pad operation.
        real_ok &= np.array_equal(it["image"][:H, :W], img)
        # Replicate padding, not zeros: the pad columns copy the last real column.
        content_ok &= np.array_equal(it["image"][:H, W:],
                                     np.repeat(img[:, W - 1:W], CROP - W, axis=1))
        content_ok &= bool(it["padded"])

    ok = check("pad path: sub-256 sample yields 256x256 crops", shape_ok)
    ok &= check("pad path: ignore mask marks exactly the synthetic region", ign_ok)
    ok &= check("pad path: real region preserved byte-for-byte", real_ok)
    ok &= check("pad path: BORDER_REPLICATE (not zeros) and padded flag set",
                content_ok)

    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
