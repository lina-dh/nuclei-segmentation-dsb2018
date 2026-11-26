"""L1 - Manifest generation and the frozen stratified train/val split.

The manifest is the pipeline's single source of truth. Everything downstream
(cache, sampler, metrics) reads scalars from here instead of touching PNGs, so
one 29,461-mask decode pass replaces a decode in every epoch.

Two artifacts are written:

    .project_state/manifest.json   670 per-sample scalar records + split labels
    .project_state/instances.npz   per-INSTANCE arrays (centroid, bbox, area)

Why two files: JSON is readable/diffable and reviewable in git, but 29,461
instance rows in JSON would be ~3 MB of text that must be parsed per run. The
npz holds those rows as concatenated typed arrays loaded in one mmap-able read.

Split policy (locked in milestones.md): stratify by modality PRIMARY, with
resolution bucket and instance-density tercile as secondary strata. A plain
random 80/20 can leave as few as 15/124 brightfield samples in val against an
expected 24.8, which makes the val metric a coin flip on the minority modality.

Single-pass design: when ``cache_dir`` is supplied, the L2 cache arrays are
written during the SAME mask-decode pass. Decoding 29,461 PNGs twice (once for
the manifest, once for the cache) would double a ~5 minute step for no reason.
"""

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import numpy as np
from scipy import ndimage as ndi

from src.data_loader import (
    MODALITY_BG_THRESHOLD,
    load_image,
    load_label_map,
    list_samples,
    write_cache_arrays,
    _to_gray,
)

# Default artifact locations. Kept inside .project_state/ because the split is
# project STATE: regenerating it with a different seed invalidates every metric
# ever recorded against it, so it must be versioned, not treated as scratch.
MANIFEST_JSON = "manifest.json"
INSTANCES_NPZ = "instances.npz"

# Resolution buckets for secondary stratification. Boundaries sit between the
# dataset's observed resolution clusters: 256x256 / 256x320 (<=82k px),
# the mid group up to 520x696 (<=400k px), and the 1024x1024+ giants.
RES_SMALL, RES_MID, RES_LARGE = 82_000, 400_000, None


# Samples held out of BOTH splits. Marked ``split="excluded"`` rather than
# deleted, so the manifest still documents them and the decision is auditable.
#
# 7b38c917...: dark-field fluorescence (bg 44.8) with a single annotated
# instance covering 0.03% of the frame (~30 px) whose intensity is DARKER than
# background (delta -19.7). It is the one negative-delta sample that is not
# brightfield. We cannot flip it (the flip must be computable without masks at
# inference), and a 30 px instance with inverted contrast is a pure label-noise
# contribution to the loss. Dropped on 2026-10-12 by decision.
EXCLUDED_SAMPLES = frozenset({
    "7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80",
})


def _res_bucket(area: int) -> str:
    """Map pixel area to a coarse resolution class."""
    if area <= RES_SMALL:
        return "small"
    if area <= RES_MID:
        return "mid"
    return "large"


def build_manifest(
    train_dir: str | Path,
    out_dir: str | Path = ".project_state",
    cache_dir: Optional[str | Path] = None,
    val_frac: float = 0.2,
    seed: int = 0,
    verbose_every: int = 100,
) -> Dict[str, Any]:
    """Decode the dataset once; emit per-sample scalars, per-instance arrays, split.

    Parameters
    ----------
    train_dir : str | Path
        Root with one directory per sample (``data/stage1_train``).
    out_dir : str | Path
        Where manifest.json + instances.npz are written.
    cache_dir : str | Path, optional
        When given, also writes the L2 cache (``<id>_img/_lab/_tgt.npy``) in this
        same pass. Hugely cheaper than a second full decode.
    val_frac : float
        Target validation fraction, applied WITHIN each stratum.
    seed : int
        Split RNG seed. Changing it produces a different frozen split.

    Returns
    -------
    dict
        The in-memory manifest: ``{"samples": [...], "split": {...}, ...}``.

    Notes
    -----
    Memory: one image + one label map + one target at a time. Peak is the
    1040x1388 sample at ~12 MB. Nothing accumulates except scalar records.
    """
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(train_dir)
    records: List[Dict[str, Any]] = []

    # Per-instance rows accumulate into flat Python lists and are converted to
    # typed arrays ONCE at the end. Appending to a growing np.ndarray would
    # reallocate and copy on every one of the 29,461 instances.
    inst_sample_idx: List[int] = []
    inst_cy: List[float] = []
    inst_cx: List[float] = []
    inst_bbox: List[Tuple[int, int, int, int]] = []
    inst_area: List[int] = []

    for si, sample in enumerate(samples):
        sid = sample.name
        image = load_image(sample)
        h, w = image.shape[:2]
        label, _ = load_label_map(sample, shape=(h, w))
        n = int(label.max())

        gray = _to_gray(image)
        fg = label > 0
        fg_px = int(fg.sum())
        bg_px = int(fg.size - fg_px)

        # mean(where=...) accumulates over the mask IN PLACE - no boolean-index
        # copy of the selected pixels, so this is O(1) extra memory.
        fg_mean = float(gray.mean(where=fg, dtype=np.float64)) if fg_px else 0.0
        bg_mean = float(gray.mean(where=~fg, dtype=np.float64)) if bg_px else 0.0

        # AUTHORITATIVE modality label: the mask-derived bg_mean threshold,
        # validated in EDA against k=2 clustering at 100% agreement.
        modality = "fluorescence" if bg_mean < MODALITY_BG_THRESHOLD else "brightfield"
        # The runtime (mask-free) proxy the inference path must use. Stored so a
        # disagreement is visible in the manifest instead of surfacing as a
        # mysterious polarity flip on unseen data.
        median_proxy = float(np.median(gray))
        proxy_modality = (
            "fluorescence" if median_proxy < MODALITY_BG_THRESHOLD else "brightfield"
        )

        # ---- per-instance geometry --------------------------------------------
        # Two single scipy calls instead of an N-iteration Python loop:
        #   find_objects   -> list of (slice_y, slice_x) bounding boxes, by id
        #   center_of_mass -> true centroids (NOT bbox centers: a crescent's bbox
        #                     center can fall outside the instance)
        # ndi works on the int32 label map directly, so no per-instance masking.
        n_truncated = 0
        if n:
            idx = np.arange(1, n + 1)
            boxes = ndi.find_objects(label)
            coms = ndi.center_of_mass(fg, labels=label, index=idx)
            # bincount over the flat label map gives every instance's pixel area
            # in ONE O(px) pass; minlength pads ids that somehow carry no pixels.
            areas = np.bincount(label.ravel(), minlength=n + 1)

            for i, (box, com) in enumerate(zip(boxes, coms), start=1):
                if box is None:
                    continue  # id with no pixels; load_label_map should prevent this
                sy, sx = box
                y0, y1, x0, x1 = int(sy.start), int(sy.stop), int(sx.start), int(sx.stop)
                truncated = y0 == 0 or x0 == 0 or y1 == h or x1 == w
                n_truncated += int(truncated)

                inst_sample_idx.append(si)
                inst_cy.append(float(com[0]))
                inst_cx.append(float(com[1]))
                inst_bbox.append((y0, x0, y1, x1))
                inst_area.append(int(areas[i]))

        records.append({
            "sample_id": sid,
            "h": h,
            "w": w,
            "area": h * w,
            "res_bucket": _res_bucket(h * w),
            "n_instances": n,
            "n_truncated": n_truncated,
            "modality": modality,
            "bg_mean": round(bg_mean, 2),
            "fg_mean": round(fg_mean, 2),
            "contrast_delta": round(fg_mean - bg_mean, 2),
            "median_gray": round(median_proxy, 2),
            "proxy_modality": proxy_modality,
            "fg_fraction": round(fg_px / fg.size, 4),
            # Filled by _assign_split below.
            "split": None,
        })

        # ---- optional L2 cache write (same decode pass) ------------------------
        # write_cache_arrays lives in data_loader (L2 owns the cache format); it
        # flips brightfield polarity here, where the authoritative whole-image
        # modality is known, and rings the 3-class target on the FULL map.
        if cache_dir is not None:
            write_cache_arrays(cache_dir, sid, image, label, modality)

        del image, gray, label, fg
        if verbose_every and (si + 1) % verbose_every == 0:
            print(f"  manifest: {si + 1}/{len(samples)} samples", flush=True)

    # ---- split ----------------------------------------------------------------
    split_summary = _assign_split(records, val_frac=val_frac, seed=seed)

    manifest = {
        "created": date.today().isoformat(),
        "train_dir": str(train_dir).replace("\\", "/"),
        "seed": seed,
        "val_frac": val_frac,
        "n_samples": len(records),
        "n_instances": len(inst_area),
        "modality_hist": dict(Counter(r["modality"] for r in records)),
        "proxy_disagreements": [
            r["sample_id"] for r in records if r["modality"] != r["proxy_modality"]
        ],
        "split_summary": split_summary,
        "samples": records,
    }

    with open(out_dir / MANIFEST_JSON, "w", encoding="utf-8") as f:
        # indent=1 keeps the file diffable in git without tripling its size.
        json.dump(manifest, f, indent=1)

    np.savez_compressed(
        out_dir / INSTANCES_NPZ,
        # Narrow dtypes on purpose: 29,461 rows, and these arrays get memmapped
        # by the sampler on every run. int32/float32 halves the read.
        sample_idx=np.asarray(inst_sample_idx, dtype=np.int32),
        centroid=np.stack([
            np.asarray(inst_cy, dtype=np.float32),
            np.asarray(inst_cx, dtype=np.float32),
        ], axis=1) if inst_cy else np.zeros((0, 2), np.float32),
        bbox=np.asarray(inst_bbox, dtype=np.int32).reshape(-1, 4),
        area=np.asarray(inst_area, dtype=np.int32),
        # sample_ids as a unicode array -> lets a consumer map sample_idx back to
        # an id without re-reading the JSON.
        sample_ids=np.asarray([r["sample_id"] for r in records]),
    )
    return manifest


def _assign_split(
    records: List[Dict[str, Any]], val_frac: float, seed: int
) -> Dict[str, Any]:
    """Mutate records in place with ``split`` in {"train", "val"}; return a summary.

    Stratum key = (modality, res_bucket, density_tercile).

    Why three keys: modality is the locked primary (the 546:124 imbalance), but
    resolution and density are the other two axes that make samples
    non-exchangeable - a val set of only 256x256 sparse images would report a
    number that says nothing about the 1040x1388 dense ones.

    Terciles are computed GLOBALLY (not per modality) so the bucket boundaries
    mean the same thing in every stratum.
    """
    counts = np.array([r["n_instances"] for r in records], dtype=np.float64)
    # percentile, not fixed thresholds: nucleus density is dataset-specific and
    # the EDA distribution is heavily right-skewed (median 27, max 375).
    t1, t2 = np.percentile(counts, [33.3, 66.6])

    def density(n: int) -> str:
        return "sparse" if n <= t1 else ("mid" if n <= t2 else "dense")

    strata: Dict[Tuple[str, str, str], List[int]] = {}
    for i, r in enumerate(records):
        # Excluded samples are labelled and skipped BEFORE stratification, so
        # they cannot shift a stratum's size or its val quota.
        if r["sample_id"] in EXCLUDED_SAMPLES:
            r["split"] = "excluded"
            continue
        key = (r["modality"], r["res_bucket"], density(r["n_instances"]))
        # setdefault returns the existing list or inserts a new one -> single
        # dict lookup instead of an `if key not in` guard plus a second lookup.
        strata.setdefault(key, []).append(i)

    # One Generator seeded once: the whole split is a pure function of `seed`,
    # so re-running regenerates the identical assignment on any machine.
    rng = np.random.default_rng(seed)

    for key in sorted(strata):  # sorted -> stratum visit order is deterministic
        members = strata[key]
        order = rng.permutation(len(members))
        k = len(members)
        n_val = int(round(val_frac * k))
        # A stratum of >=2 must contribute at least one val sample, otherwise the
        # rare strata (large brightfield) vanish from validation entirely - the
        # exact failure stratification exists to prevent.
        if k >= 2:
            n_val = max(1, min(n_val, k - 1))
        else:
            n_val = 0  # a singleton stratum stays in train
        for rank, pos in enumerate(order):
            records[members[pos]]["split"] = "val" if rank < n_val else "train"

    n_excluded = sum(1 for r in records if r["split"] == "excluded")

    def tally(split: str) -> Dict[str, Any]:
        sub = [r for r in records if r["split"] == split]
        return {
            "n": len(sub),
            "modality": dict(Counter(r["modality"] for r in sub)),
            "res_bucket": dict(Counter(r["res_bucket"] for r in sub)),
            "instances": int(sum(r["n_instances"] for r in sub)),
        }

    return {
        "n_strata": len(strata),
        "density_terciles": [round(float(t1), 1), round(float(t2), 1)],
        "n_excluded": n_excluded,
        "excluded_ids": sorted(r["sample_id"] for r in records if r["split"] == "excluded"),
        "train": tally("train"),
        "val": tally("val"),
    }


def patch_manifest_exclusions(out_dir: str | Path = ".project_state") -> Dict[str, Any]:
    """Apply :data:`EXCLUDED_SAMPLES` to an EXISTING manifest.json in place.

    Why this exists instead of just re-running ``build_manifest``: a rebuild
    re-decodes 29,461 mask PNGs (~306s) to recompute scalars that have not
    changed. Exclusion only rewrites a ``split`` field, so it is a pure metadata
    edit.

    IMPORTANT: this does NOT re-randomize the split. Every other sample keeps
    its original train/val assignment, so metrics recorded before the exclusion
    remain comparable. It only ever moves samples INTO ``"excluded"``.
    """
    out_dir = Path(out_dir)
    mpath = out_dir / MANIFEST_JSON
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)

    moved = []
    for r in manifest["samples"]:
        if r["sample_id"] in EXCLUDED_SAMPLES and r["split"] != "excluded":
            moved.append((r["sample_id"], r["split"]))
            r["split"] = "excluded"

    # Rebuild the summary tallies from the patched records so the file stays
    # internally consistent (counts must match the per-sample labels).
    def tally(split: str) -> Dict[str, Any]:
        sub = [r for r in manifest["samples"] if r["split"] == split]
        return {
            "n": len(sub),
            "modality": dict(Counter(r["modality"] for r in sub)),
            "res_bucket": dict(Counter(r["res_bucket"] for r in sub)),
            "instances": int(sum(r["n_instances"] for r in sub)),
        }

    ss = manifest["split_summary"]
    ss["train"], ss["val"] = tally("train"), tally("val")
    ss["n_excluded"] = sum(1 for r in manifest["samples"] if r["split"] == "excluded")
    ss["excluded_ids"] = sorted(
        r["sample_id"] for r in manifest["samples"] if r["split"] == "excluded"
    )

    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return {"moved": moved, "split_summary": ss}


def load_manifest(
    out_dir: str | Path = ".project_state",
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Read back the frozen manifest + instance arrays.

    Returns
    -------
    (manifest, instances)
        ``manifest`` is the parsed JSON. ``instances`` is a plain dict of arrays
        materialized from the npz, because NpzFile decompresses lazily per key
        and the sampler indexes these on every crop.
    """
    out_dir = Path(out_dir)
    mpath = out_dir / MANIFEST_JSON
    if not mpath.is_file():
        raise FileNotFoundError(f"{mpath} missing - run build_manifest() first")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    with np.load(out_dir / INSTANCES_NPZ, allow_pickle=False) as z:
        instances = {k: z[k] for k in z.files}
    return manifest, instances
