"""Pre-flight integrity audit for the 2018 Data Science Bowl nuclei dataset.

Purpose: run ONCE before building the mask-merging loader. Silent data defects
(missing masks/, all-zero masks, image/mask shape mismatch) would otherwise
surface as cryptic broadcast errors deep inside a training loop.

Memory contract: masks are decoded one at a time and immediately reduced to
(shape, max) scalars. Peak RAM = one mask, NOT one sample and NOT the dataset.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Anomaly type tags. Centralized so callers can filter without magic strings.
MISSING_IMAGES_DIR = "missing_images_dir"
MISSING_IMAGE_FILE = "missing_image_file"
MISSING_MASKS_DIR = "missing_masks_dir"
ZERO_MASKS = "zero_masks"
UNREADABLE_IMAGE = "unreadable_image"
UNREADABLE_MASK = "unreadable_mask"
SHAPE_MISMATCH = "shape_mismatch"
EMPTY_MASK = "empty_mask"


def audit_dataset(train_dir: str) -> Dict[str, Any]:
    """Walk every sample directory and validate structure, shapes, and content.

    Parameters
    ----------
    train_dir : str
        Root holding one directory per sample, e.g. ``data/stage1_train``.

    Returns
    -------
    dict
        total_samples      : int   - sample dirs discovered
        valid_samples      : int   - samples with zero anomalies
        total_images       : int   - image files found (expect == total_samples)
        total_mask_dirs    : int   - masks/ dirs found (expect == total_samples)
        total_masks        : int   - individual mask PNGs across dataset
        mask_count_hist    : dict  - {'zero', 'one', 'many'} sample counts
        masks_per_sample   : dict  - {'min', 'max', 'mean', 'median'}
        anomalies          : list  - [{'sample_id', 'issue', 'detail'}, ...]
        anomaly_counts     : dict  - {issue_tag: occurrences}
        extremes           : dict  - sample ids for {'fewest', 'most'} masks
                                     (drives the EDA visualization cell)

    Raises
    ------
    FileNotFoundError
        If ``train_dir`` does not exist (dataset not extracted yet).
    """
    root = Path(train_dir)
    if not root.is_dir():
        # Fail fast with an actionable message: the usual cause is a skipped unzip.
        raise FileNotFoundError(f"{root} not found - extract stage1_train.zip first")

    # iterdir() is a lazy generator -> does NOT build a 670-element list in memory.
    # The is_dir() filter drops stray files (e.g. .DS_Store, .gitkeep) that would
    # otherwise be miscounted as samples.
    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    anomalies: List[Dict[str, str]] = []
    mask_counts: Dict[str, int] = {}  # sample_id -> mask count, for stats + extremes
    total_images = 0
    total_mask_dirs = 0

    for sample in sample_dirs:
        sid = sample.name
        clean = True  # flipped by any anomaly; decides valid_samples at the end

        images_dir = sample / "images"
        masks_dir = sample / "masks"

        # ---- structural checks -------------------------------------------------
        # Shape reference for this sample; stays None if the image is unusable,
        # which disables (rather than falsely fails) the per-mask shape check.
        img_shape = None

        if not images_dir.is_dir():
            anomalies.append({"sample_id": sid, "issue": MISSING_IMAGES_DIR, "detail": "no images/"})
            clean = False
        else:
            # next(glob, None) pulls only the FIRST match and abandons the generator.
            # Cheaper than list(glob)[0] and safe when the directory is empty.
            img_path = next(images_dir.glob("*.png"), None)
            if img_path is None:
                anomalies.append({"sample_id": sid, "issue": MISSING_IMAGE_FILE, "detail": "images/ empty"})
                clean = False
            else:
                total_images += 1
                # IMREAD_UNCHANGED keeps the native channel count (these PNGs are RGBA)
                # so a 4-channel file is not silently reshaped before the shape compare.
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    # cv2.imread returns None (never raises) on a truncated/corrupt file.
                    anomalies.append({"sample_id": sid, "issue": UNREADABLE_IMAGE, "detail": img_path.name})
                    clean = False
                else:
                    # [:2] slices (H, W) off (H, W, C) -> comparable to a 2-D mask shape.
                    img_shape = img.shape[:2]
                del img  # drop the decoded array before touching the masks

        if not masks_dir.is_dir():
            anomalies.append({"sample_id": sid, "issue": MISSING_MASKS_DIR, "detail": "no masks/"})
            clean = False
            mask_counts[sid] = 0
            continue  # nothing left to inspect for this sample

        total_mask_dirs += 1
        mask_paths = sorted(masks_dir.glob("*.png"))  # sorted -> reproducible ordering
        mask_counts[sid] = len(mask_paths)

        if not mask_paths:
            anomalies.append({"sample_id": sid, "issue": ZERO_MASKS, "detail": "masks/ empty"})
            clean = False
            continue

        # ---- per-mask content + dimension checks ------------------------------
        for mp in mask_paths:
            # GRAYSCALE forces a single 2-D channel: masks are binary, so this both
            # halves/quarters the decode cost and makes .shape directly (H, W).
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                anomalies.append({"sample_id": sid, "issue": UNREADABLE_MASK, "detail": mp.name})
                clean = False
                continue

            if img_shape is not None and m.shape != img_shape:
                # Tuple != tuple is an exact elementwise compare; a mismatch here is
                # what later breaks np.stack / broadcasting in the merging loader.
                anomalies.append({
                    "sample_id": sid,
                    "issue": SHAPE_MISMATCH,
                    "detail": f"img{img_shape} vs mask{m.shape} ({mp.name})",
                })
                clean = False

            # m.max() == 0 <=> all pixels zero. Single C-level reduction over the
            # buffer; far cheaper than np.any(m) chaining or a Python-level scan.
            if m.max() == 0:
                anomalies.append({"sample_id": sid, "issue": EMPTY_MASK, "detail": mp.name})
                clean = False

            del m  # release this mask before decoding the next one

    # ---- aggregate ------------------------------------------------------------
    counts = np.array(list(mask_counts.values()), dtype=np.int32)

    hist = {
        "zero": int((counts == 0).sum()),
        "one": int((counts == 1).sum()),
        "many": int((counts > 1).sum()),
    }

    stats = {
        "min": int(counts.min()) if counts.size else 0,
        "max": int(counts.max()) if counts.size else 0,
        "mean": round(float(counts.mean()), 2) if counts.size else 0.0,
        "median": float(np.median(counts)) if counts.size else 0.0,
    }

    anomaly_counts: Dict[str, int] = {}
    for a in anomalies:
        # dict.get(k, 0) + 1 -> tally without pre-seeding every possible key.
        anomaly_counts[a["issue"]] = anomaly_counts.get(a["issue"], 0) + 1

    # Ignore 0-mask samples when picking the "fewest" exemplar: an empty sample is
    # an anomaly, not an interesting low-density image to plot.
    nonzero = {k: v for k, v in mask_counts.items() if v > 0}
    extremes = {
        # min/max with key= returns the KEY (sample_id) whose value is extremal.
        "fewest": min(nonzero, key=nonzero.get) if nonzero else None,
        "most": max(nonzero, key=nonzero.get) if nonzero else None,
    }

    return {
        "total_samples": len(sample_dirs),
        "valid_samples": len(sample_dirs) - len({a["sample_id"] for a in anomalies}),
        "total_images": total_images,
        "total_mask_dirs": total_mask_dirs,
        "total_masks": int(counts.sum()) if counts.size else 0,
        "mask_count_hist": hist,
        "masks_per_sample": stats,
        "anomalies": anomalies,
        "anomaly_counts": anomaly_counts,
        "extremes": extremes,
        "_mask_counts": mask_counts,  # private: per-sample counts for plotting
    }


# =============================================================================
# Deep EDA profiling: dimensions, aspect ratios, foreground/background intensity
# =============================================================================

# Background-intensity cut separating the two acquisition modalities.
# Fluorescence = nuclei glow on a near-black field (bg mean ~ 0-30).
# Brightfield/H&E = stained nuclei on a bright field (bg mean ~ 150-220).
# Heuristic only; the notebook histogram is what actually proves bimodality.
MODALITY_BG_THRESHOLD = 90


def _stats(arr: np.ndarray) -> Dict[str, float]:
    """Reduce a 1-D array to scalar summary stats (never returns the array)."""
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(arr.std()), 2),
    }


def profile_dataset(train_dir: str, compute_intensity: bool = True) -> Dict[str, Any]:
    """Profile image dimensions, aspect ratios, and fg/bg intensity separation.

    Single disk pass by design: dimension profiling and intensity profiling both
    need every mask decoded, and ~29.5k PNG decodes is the dominant cost. Running
    them as two functions would double runtime for zero benefit.

    Parameters
    ----------
    train_dir : str
        Root holding one directory per sample, e.g. ``data/stage1_train``.
    compute_intensity : bool
        False -> skip mask decoding entirely and profile dimensions only
        (seconds instead of minutes). Useful for a quick resolution check.

    Returns
    -------
    dict
        per_sample : list of dict - one record per sample:
            sample_id, h, w, aspect (w/h), channels, n_masks,
            bg_mean, fg_mean, contrast_delta (fg-bg), fg_px, bg_px, modality
        dimensions : dict
            modal (H,W) + its count/pct, unique_count, full distribution,
            deviants list, deviant_sample_count/pct, aspect-ratio stats
        intensity : dict  (empty when compute_intensity=False)
            bg_mean/fg_mean/contrast_delta stat blocks, modality histogram,
            inverted_contrast_count (samples where nuclei are DARKER than bg)

    Notes
    -----
    Memory: per sample we hold one grayscale image + one bool union mask
    (~1.25 bytes/pixel total). Worst case here is ~1.6 MB. Nothing accumulates
    except the 670 scalar records.
    """
    root = Path(train_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found - extract stage1_train.zip first")

    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    per_sample: List[Dict[str, Any]] = []

    for sample in sample_dirs:
        img_path = next((sample / "images").glob("*.png"), None)
        if img_path is None:
            continue  # structural defect; audit_dataset() already reports it

        # IMREAD_UNCHANGED only to read the native channel count for the record.
        # Intensity work uses the grayscale decode below instead.
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        h, w = raw.shape[:2]
        # 2-D array -> len(shape)==2 -> single channel; else shape[2] is the count.
        channels = 1 if raw.ndim == 2 else raw.shape[2]
        del raw

        rec: Dict[str, Any] = {
            "sample_id": sample.name,
            "h": int(h),
            "w": int(w),
            # Aspect as w/h: >1 landscape, <1 portrait, ==1 square.
            "aspect": round(w / h, 3),
            "channels": int(channels),
            "n_masks": 0,
            "bg_mean": None,
            "fg_mean": None,
            "contrast_delta": None,
            "fg_px": 0,
            "bg_px": 0,
            "modality": None,
        }

        if not compute_intensity:
            per_sample.append(rec)
            continue

        # ---- aggregate mask union ---------------------------------------------
        # Union, not a stack: a (375, H, W) stack costs ~375 MB, while one bool
        # (H, W) plane costs ~0.7 MB. Intensity stats only need "is this pixel
        # nucleus or background", so instance identity is irrelevant here.
        agg = np.zeros((h, w), dtype=bool)
        mask_paths = sorted((sample / "masks").glob("*.png"))
        rec["n_masks"] = len(mask_paths)

        for mp in mask_paths:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (h, w):
                continue
            # |= is in-place OR: accumulates into the SAME buffer, so the loop
            # allocates nothing per iteration. `m > 127` binarizes {0,255} -> bool.
            agg |= m > 127
            del m

        # ---- intensity on grayscale -------------------------------------------
        # IMREAD_GRAYSCALE applies cv2's luminance weighting (0.299R+0.587G+0.114B)
        # and drops alpha, collapsing every modality to one comparable scale.
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (h, w):
            per_sample.append(rec)
            continue

        # agg.sum() counts True pixels (bool sums as 0/1). ~ is elementwise NOT.
        fg_px = int(agg.sum())
        bg_px = int(agg.size - fg_px)
        rec["fg_px"], rec["bg_px"] = fg_px, bg_px

        # NUMPY INDEXING NOTE (the key choice in this function):
        #   img[mask > 0]  -> boolean indexing. Builds a 1-D COPY of the selected
        #                     pixels. Readable, but allocates up to H*W bytes.
        #   img[mask == 0] -> same mechanism for the complement.
        #   .mean(where=mask) -> accumulates over the mask IN PLACE. No copy, no
        #                     temporary. Same result, O(1) extra memory.
        # We use `where=` for the reduction; the bracket form is shown in the
        # notebook where a full pixel distribution is genuinely needed.
        if fg_px:
            rec["fg_mean"] = round(float(gray.mean(where=agg, dtype=np.float64)), 2)
        if bg_px:
            rec["bg_mean"] = round(float(gray.mean(where=~agg, dtype=np.float64)), 2)

        if rec["fg_mean"] is not None and rec["bg_mean"] is not None:
            # Signed delta: positive = nuclei brighter than field (fluorescence),
            # negative = nuclei darker (brightfield/H&E stain).
            rec["contrast_delta"] = round(rec["fg_mean"] - rec["bg_mean"], 2)
            rec["modality"] = (
                "fluorescence" if rec["bg_mean"] < MODALITY_BG_THRESHOLD else "brightfield"
            )

        del agg, gray
        per_sample.append(rec)

    # ---- dimension aggregation ------------------------------------------------
    # Counter over (h, w) TUPLES: tuples are hashable, so they key a dict directly;
    # a [h, w] list would raise TypeError: unhashable type.
    dim_counter = Counter((r["h"], r["w"]) for r in per_sample)
    n = len(per_sample)

    # most_common(1) -> [(key, count)]; [0] unpacks that single pair.
    modal_hw, modal_count = dim_counter.most_common(1)[0]

    distribution = [
        {
            "hw": hw,
            "count": c,
            "pct": round(100 * c / n, 2),
            "deviant": hw != modal_hw,
        }
        for hw, c in dim_counter.most_common()  # already sorted count-descending
    ]
    deviants = [d for d in distribution if d["deviant"]]
    deviant_samples = sum(d["count"] for d in deviants)

    aspects = np.array([r["aspect"] for r in per_sample], dtype=np.float64)
    dimensions = {
        "modal": modal_hw,
        "modal_count": modal_count,
        "modal_pct": round(100 * modal_count / n, 2),
        "unique_count": len(dim_counter),
        "distribution": distribution,
        "deviants": deviants,
        "deviant_sample_count": deviant_samples,
        "deviant_pct": round(100 * deviant_samples / n, 2),
        "aspect": {**_stats(aspects), "unique": sorted(set(aspects.tolist()))[:10]},
        "channel_hist": dict(Counter(r["channels"] for r in per_sample)),
    }

    # ---- intensity aggregation ------------------------------------------------
    intensity: Dict[str, Any] = {}
    if compute_intensity:
        # Filter None before array-building: np.array([1.0, None]) yields an
        # object-dtype array whose .mean() fails. Explicit guard keeps dtype float.
        bg = np.array([r["bg_mean"] for r in per_sample if r["bg_mean"] is not None])
        fg = np.array([r["fg_mean"] for r in per_sample if r["fg_mean"] is not None])
        dl = np.array([r["contrast_delta"] for r in per_sample if r["contrast_delta"] is not None])

        intensity = {
            "bg_mean": _stats(bg),
            "fg_mean": _stats(fg),
            "contrast_delta": _stats(dl),
            "modality_hist": dict(Counter(
                r["modality"] for r in per_sample if r["modality"]
            )),
            # Negative delta = inverted contrast. These samples break any model
            # that assumes "nuclei are the bright blobs" and must be normalized.
            "inverted_contrast_count": int((dl < 0).sum()) if dl.size else 0,
            # Low |delta| = weak signal separation -> hardest samples to segment.
            "low_separation_count": int((np.abs(dl) < 20).sum()) if dl.size else 0,
            # Fraction of each image occupied by nuclei: a class-imbalance check.
            "fg_fraction": _stats(np.array([
                r["fg_px"] / (r["fg_px"] + r["bg_px"])
                for r in per_sample if (r["fg_px"] + r["bg_px"])
            ])),
        }

    return {"per_sample": per_sample, "dimensions": dimensions, "intensity": intensity}
