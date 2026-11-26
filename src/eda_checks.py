"""Advanced pre-modeling EDA for the 2018 Data Science Bowl nuclei dataset.

Three checks, each answering a decision that must be made BEFORE architecture:

  A. audit_instances()        -> do instances touch/overlap? are any sub-pixel?
                                 Decides: binary U-Net vs 3-class/watershed target.
  B. cluster_modalities()     -> how many real acquisition modalities exist?
                                 Decides: normalization scheme + split stratification.
  C. check_boundary_quality() -> do mask edges sit on image edges?
                                 Decides: the label-noise ceiling on achievable IoU.

Memory contract (identical to dataset_audit): every sample is reduced to scalars
before the next is opened. Peak RAM = one image + one label map + one accumulator.
No module here ever holds a (N, H, W) mask stack.

Dependencies: cv2, numpy, scipy only. sklearn is deliberately avoided
(scipy.cluster.vq.kmeans2 + a hand-rolled silhouette cover the clustering need).
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2

# 3x3 cross kernel. Cross (not square) = 4-connectivity, matching how the
# challenge's IoU metric treats adjacency; a square kernel would count
# diagonal-only neighbours as touching and inflate the pair count.
_K3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_K3SQ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# A nucleus below this equivalent diameter (px) disappears under a 2x downsample
# and cannot be recovered by any decoder.
SMALL_DIAMETER_PX = 10


def _sample_dirs(root: Path) -> List[Path]:
    """Sorted sample directories. sorted() -> deterministic record ordering."""
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found - extract stage1_train.zip first")
    return sorted(p for p in root.iterdir() if p.is_dir())


def _stats(arr: np.ndarray) -> Dict[str, float]:
    """Scalar summary of a 1-D array. Never returns the array itself."""
    if arr.size == 0:
        return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "p01": 0.0, "p99": 0.0}
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "std": round(float(arr.std()), 3),
        # Percentiles, not min/max, are what you threshold on: extremes are
        # single-sample artifacts, p01/p99 describe the actual working range.
        "p01": round(float(np.percentile(arr, 1)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
    }


# =============================================================================
# CHECK A - instance morphometrics + touching / overlap audit
# =============================================================================

def audit_instances(train_dir: str, stride: int = 1) -> Dict[str, Any]:
    """Per-nucleus morphometrics plus instance adjacency and overlap counts.

    This is the check that decides the model's output representation. A plain
    binary foreground mask cannot separate two nuclei that share a boundary, so
    the touching-pair rate directly predicts how much mAP a binary U-Net loses.

    Parameters
    ----------
    train_dir : str
        Root holding one directory per sample.
    stride : int
        Process every Nth sample. 1 = full dataset (~29.5k mask decodes).

    Returns
    -------
    dict
        instances   : dict of stat blocks over ALL nuclei (area, diameter,
                      eccentricity, solidity, mean_intensity, contrast_vs_bg)
        counts      : small/border/low-contrast instance tallies + pct
        adjacency   : touching-pair and overlap statistics
        per_sample  : one compact record per sample (for plotting)
        arrays      : raw 1-D numpy arrays of per-instance metrics (plotting)
    """
    root = Path(train_dir)
    samples = _sample_dirs(root)[::stride]

    # Flat per-instance metric lists. Appending scalars (not arrays) keeps this
    # O(n_instances) floats ~= 30k * 8 B per metric, i.e. negligible.
    area, diam, ecc, solid, inten, contrast = [], [], [], [], [], []
    n_border = n_small = n_lowcontrast = 0
    per_sample: List[Dict[str, Any]] = []

    for sdir in samples:
        img_path = next((sdir / "images").glob("*.png"), None)
        if img_path is None:
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape

        mask_paths = sorted((sdir / "masks").glob("*.png"))
        n_inst = len(mask_paths)
        if n_inst == 0:
            continue

        # Two accumulators instead of a mask stack:
        #   labels : int32 last-write-wins instance id map, used for adjacency.
        #   acc    : uint16 count of how many masks cover each pixel -> overlap.
        # uint16 caps at 65535 coverings; max instances here is 375, so safe.
        labels = np.zeros((h, w), dtype=np.int32)
        acc = np.zeros((h, w), dtype=np.uint16)

        # Per-instance bbox cache. Storing 4 ints per instance lets the adjacency
        # pass crop instead of scanning the full frame (the key optimization:
        # O(n * h_box * w_box) instead of O(n * H * W)).
        boxes: List[Optional[Tuple[int, int, int, int]]] = [None] * n_inst
        s_area: List[int] = []

        for i, mp in enumerate(mask_paths, start=1):
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (h, w):
                boxes[i - 1] = None
                continue
            b = m > 127  # binarize {0,255} -> bool

            a = int(b.sum())
            if a == 0:
                boxes[i - 1] = None
                continue

            labels[b] = i          # boolean assignment: writes only True pixels
            acc += b               # bool promotes to uint16 -> coverage count

            # np.nonzero on each axis is cheaper than argwhere: it avoids
            # materializing an (a, 2) coordinate array just to take min/max.
            rows = np.nonzero(b.any(axis=1))[0]
            cols = np.nonzero(b.any(axis=0))[0]
            r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
            boxes[i - 1] = (r0, r1, c0, c1)
            s_area.append(a)

            # --- morphometrics on the bbox crop (uint8 for cv2 compatibility) ---
            crop = b[r0:r1 + 1, c0:c1 + 1].astype(np.uint8)

            # Equivalent diameter = diameter of a circle with the same area.
            # Scale-free size proxy that is robust to elongation.
            d = 2.0 * np.sqrt(a / np.pi)

            # cv2.moments returns raw (m..) and CENTRAL (mu..) moments in one C
            # pass. Normalizing central second moments by m00 gives the
            # covariance of the pixel cloud; its eigenvalues are the squared
            # semi-axes of the equivalent ellipse -> eccentricity.
            mo = cv2.moments(crop, binaryImage=True)
            m00 = mo["m00"]
            if m00 > 0:
                u20, u02, u11 = mo["mu20"] / m00, mo["mu02"] / m00, mo["mu11"] / m00
                common = np.sqrt(max(4.0 * u11 ** 2 + (u20 - u02) ** 2, 0.0))
                l1, l2 = (u20 + u02 + common) / 2.0, (u20 + u02 - common) / 2.0
                e = float(np.sqrt(1.0 - l2 / l1)) if l1 > 1e-9 and l2 >= 0 else 0.0
            else:
                e = 0.0

            # Solidity = area / convex-hull area. <1 means concave boundary,
            # which for a nucleus usually means an annotation merged two cells.
            cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                # vstack all contours: an instance can be split into pieces, and
                # one hull over all of them is the correct convex envelope.
                hull = cv2.convexHull(np.vstack(cnts))
                ha = cv2.contourArea(hull)
                sol = float(a / ha) if ha > 0 else 1.0
            else:
                sol = 1.0

            # Per-instance intensity: where= avoids the 1-D copy that
            # gray[b] would allocate.
            mi = float(gray.mean(where=b, dtype=np.float64))

            area.append(a); diam.append(d); ecc.append(e)
            solid.append(min(sol, 1.0)); inten.append(mi)

            if d < SMALL_DIAMETER_PX:
                n_small += 1
            # Touching the frame edge => the nucleus is truncated; its shape
            # statistics are censored and it should not train a shape prior.
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                n_border += 1

            del m, b, crop

        if not s_area:
            continue

        # --- sample-level background reference ---------------------------------
        union = acc > 0
        bg_px = int((~union).sum())
        bg_mean = float(gray.mean(where=~union, dtype=np.float64)) if bg_px else 0.0

        # Per-instance contrast against the sample's own background. Absolute
        # intensity is meaningless across modalities; this delta is comparable.
        k = len(s_area)
        for mi in inten[-k:]:
            c = mi - bg_mean
            contrast.append(c)
            if abs(c) < 10:
                n_lowcontrast += 1

        # --- overlap -----------------------------------------------------------
        overlap_px = int((acc > 1).sum())

        # --- adjacency via bbox-cropped dilation -------------------------------
        # For each instance: dilate its mask 1px, subtract itself to get a
        # 1-px RING, then read the label map under that ring. Any nonzero,
        # non-self label there is a 4-connected neighbour.
        pairs = set()
        for idx, bx in enumerate(boxes, start=1):
            if bx is None:
                continue
            r0, r1, c0, c1 = bx
            # Pad by 1 and clip to the frame so the dilation has room to grow.
            pr0, pr1 = max(r0 - 1, 0), min(r1 + 2, h)
            pc0, pc1 = max(c0 - 1, 0), min(c1 + 2, w)
            lab_crop = labels[pr0:pr1, pc0:pc1]

            # Reconstruct this instance's footprint from the label map. Pixels
            # lost to last-write-wins overlap are excluded, which is harmless:
            # an overlapping pair is already counted by the overlap metric.
            self_crop = (lab_crop == idx).astype(np.uint8)
            if not self_crop.any():
                continue
            ring = cv2.dilate(self_crop, _K3, iterations=1) - self_crop
            nb = np.unique(lab_crop[ring.astype(bool)])
            for j in nb:
                if j != 0 and j != idx:
                    # frozenset -> unordered pair, so (i,j) and (j,i) dedupe.
                    pairs.add(frozenset((int(idx), int(j))))

        n_touching_inst = len({i for p in pairs for i in p})

        per_sample.append({
            "sample_id": sdir.name,
            "n_inst": k,
            "h": h, "w": w,
            "touching_pairs": len(pairs),
            "touching_inst": n_touching_inst,
            "touching_inst_pct": round(100 * n_touching_inst / k, 2),
            "overlap_px": overlap_px,
            "overlap_pct_of_fg": round(100 * overlap_px / max(int(union.sum()), 1), 4),
            "median_area": float(np.median(s_area)),
            "bg_mean": round(bg_mean, 2),
            "density": round(k / (h * w) * 1e4, 3),  # instances per 100x100 px
        })

        del labels, acc, gray, union

    # ---- aggregate ------------------------------------------------------------
    arr = {
        "area": np.array(area, dtype=np.float64),
        "diameter": np.array(diam, dtype=np.float64),
        "eccentricity": np.array(ecc, dtype=np.float64),
        "solidity": np.array(solid, dtype=np.float64),
        "mean_intensity": np.array(inten, dtype=np.float64),
        "contrast_vs_bg": np.array(contrast, dtype=np.float64),
    }
    n_inst_total = int(arr["area"].size)

    tp = np.array([r["touching_pairs"] for r in per_sample], dtype=np.float64)
    ti = np.array([r["touching_inst_pct"] for r in per_sample], dtype=np.float64)
    ov = np.array([r["overlap_px"] for r in per_sample], dtype=np.float64)

    return {
        "n_samples": len(per_sample),
        "n_instances": n_inst_total,
        "instances": {k: _stats(v) for k, v in arr.items()},
        "counts": {
            "small_instances": n_small,
            "small_pct": round(100 * n_small / max(n_inst_total, 1), 2),
            "border_instances": n_border,
            "border_pct": round(100 * n_border / max(n_inst_total, 1), 2),
            "lowcontrast_instances": n_lowcontrast,
            "lowcontrast_pct": round(100 * n_lowcontrast / max(n_inst_total, 1), 2),
        },
        "adjacency": {
            "total_touching_pairs": int(tp.sum()),
            "touching_pairs_per_sample": _stats(tp),
            "touching_instance_pct_per_sample": _stats(ti),
            "samples_with_any_touching": int((tp > 0).sum()),
            "samples_with_any_touching_pct": round(100 * (tp > 0).sum() / max(len(per_sample), 1), 2),
            "total_overlap_px": int(ov.sum()),
            "samples_with_overlap": int((ov > 0).sum()),
            "overlap_px_per_sample": _stats(ov),
        },
        "per_sample": per_sample,
        "arrays": arr,
    }


# =============================================================================
# CHECK B - modality clustering + split stratification risk
# =============================================================================

def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score, numpy-only (sklearn is unavailable here).

    s(i) = (b - a) / max(a, b);  a = mean intra-cluster distance,
    b = mean distance to the nearest OTHER cluster. Range [-1, 1], higher better.
    """
    n = X.shape[0]
    uniq = np.unique(labels)
    if uniq.size < 2 or n < 3:
        return float("nan")

    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y  -> one (n,n) matrix instead of an
    # (n,n,d) broadcast temporary. At n=670 this is ~3.6 MB, not 18 MB.
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    D = np.sqrt(np.maximum(d2, 0.0))

    s = np.zeros(n)
    for c in uniq:
        inc = labels == c
        n_c = int(inc.sum())
        if n_c <= 1:
            continue  # singleton clusters have undefined silhouette -> leave 0
        # Sum over own cluster minus self (diagonal is 0) / (n_c - 1).
        a = D[np.ix_(inc, inc)].sum(axis=1) / (n_c - 1)
        b = np.full(n_c, np.inf)
        for o in uniq:
            if o == c:
                continue
            outc = labels == o
            b = np.minimum(b, D[np.ix_(inc, outc)].mean(axis=1))
        s[inc] = (b - a) / np.maximum(a, b)
    return round(float(s.mean()), 4)


def cluster_modalities(
    train_dir: str,
    per_sample: Optional[List[Dict[str, Any]]] = None,
    k_values: Tuple[int, ...] = (2, 3, 4, 5),
    val_frac: float = 0.2,
    n_trials: int = 400,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cluster samples on appearance features; quantify random-split risk.

    Validates (or refutes) the ``bg_mean < 90`` modality heuristic from
    dataset_audit, and measures how badly a non-stratified train/val split can
    under-represent a small modality.

    Parameters
    ----------
    train_dir : str
        Root holding one directory per sample.
    per_sample : list of dict, optional
        ``profile_dataset()["per_sample"]`` records. When given, bg/fg/contrast
        features are reused instead of recomputed (no mask decoding needed).
    k_values : tuple of int
        Candidate cluster counts to score.
    val_frac, n_trials, seed
        Monte-Carlo parameters for the split-risk simulation.

    Returns
    -------
    dict
        features, scores (per k), best_k, clusters (sizes + centroids in
        original units), crosstab vs the bg<90 heuristic, split_risk.
    """
    root = Path(train_dir)
    samples = _sample_dirs(root)

    # Index the optional profile records by id for O(1) joins.
    prof_by_id = {r["sample_id"]: r for r in (per_sample or [])}

    ids: List[str] = []
    feats: List[List[float]] = []
    names = ["gray_mean", "gray_std", "sat_mean", "color_spread", "bg_mean", "contrast_delta"]

    for sdir in samples:
        img_path = next((sdir / "images").glob("*.png"), None)
        if img_path is None:
            continue
        # Images only - no masks. 670 decodes, so this check runs in seconds.
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # HSV saturation separates stained (purple/pink H&E) from grayscale
        # fluorescence far more cleanly than RGB means do.
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

        # float64 means over uint8 planes; axis=(0,1) reduces spatially and
        # leaves the 3 channel means.
        ch = bgr.reshape(-1, 3).mean(axis=0)

        rec = prof_by_id.get(sdir.name, {})
        feats.append([
            float(gray.mean()),
            float(gray.std()),
            float(sat.mean()),
            float(ch.max() - ch.min()),        # channel imbalance = colour cast
            float(rec.get("bg_mean") or gray.mean()),
            float(rec.get("contrast_delta") or 0.0),
        ])
        ids.append(sdir.name)
        del bgr, gray, sat

    X_raw = np.array(feats, dtype=np.float64)

    # z-score: kmeans uses Euclidean distance, so an unscaled feature with a
    # larger range (gray_mean 0-255) would dominate one with a small range.
    mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X_raw - mu) / sd

    scores = {}
    best_k, best_s, best_labels, best_cent = None, -np.inf, None, None
    for k in k_values:
        # minit="++" = k-means++ seeding: picks far-apart initial centroids, so
        # the result is stable without multiple restarts. seed fixes the RNG.
        cent, lab = kmeans2(X, k, minit="++", seed=seed, iter=50)
        # Within-cluster sum of squares (inertia) for an elbow read.
        wcss = float(((X - cent[lab]) ** 2).sum())
        sil = _silhouette(X, lab)
        scores[k] = {"silhouette": sil, "wcss": round(wcss, 1),
                     "sizes": sorted(Counter(lab.tolist()).values(), reverse=True)}
        if sil > best_s:
            best_k, best_s, best_labels, best_cent = k, sil, lab, cent

    # Centroids back in original units: readable ("bg_mean 12" not "z -0.9").
    cent_raw = best_cent * sd + mu
    clusters = []
    for c in range(best_k):
        inc = best_labels == c
        clusters.append({
            "cluster": int(c),
            "size": int(inc.sum()),
            "pct": round(100 * inc.sum() / len(ids), 2),
            "centroid": {n: round(float(v), 2) for n, v in zip(names, cent_raw[c])},
        })
    clusters.sort(key=lambda d: -d["size"])

    # Cross-tab against the bg<90 heuristic: does clustering agree, or split
    # one heuristic class into several real sub-populations?
    crosstab: Dict[str, Dict[str, int]] = {}
    for sid, lab in zip(ids, best_labels):
        heur = (prof_by_id.get(sid, {}) or {}).get("modality") or "unknown"
        crosstab.setdefault(heur, {}).setdefault(f"c{int(lab)}", 0)
        crosstab[heur][f"c{int(lab)}"] += 1

    # ---- split-risk simulation ------------------------------------------------
    # Random (unstratified) splits: how often does the rarest cluster land with
    # too few validation samples to produce a usable per-modality score?
    rng = np.random.default_rng(seed)
    n = len(ids)
    n_val = int(round(val_frac * n))
    sizes = np.array([c["size"] for c in clusters])
    rarest = int(sizes.min())
    rare_c = int(clusters[int(np.argmin(sizes))]["cluster"])
    rare_mask = best_labels == rare_c

    val_counts = np.empty(n_trials, dtype=np.int32)
    for t in range(n_trials):
        # permutation[:n_val] is a uniform sample WITHOUT replacement.
        sel = rng.permutation(n)[:n_val]
        val_counts[t] = int(rare_mask[sel].sum())

    expected = rarest * val_frac
    split_risk = {
        "n": n, "n_val": n_val,
        "rarest_cluster": rare_c, "rarest_size": rarest,
        "expected_rare_in_val": round(expected, 2),
        "observed": _stats(val_counts.astype(np.float64)),
        # Deviating >30% from expectation makes a per-modality metric unstable.
        "pct_trials_off_by_30pct": round(
            100 * float((np.abs(val_counts - expected) > 0.3 * expected).mean()), 2
        ),
        "worst_case_rare_in_val": int(val_counts.min()),
    }

    return {
        "feature_names": names,
        "feature_stats": {n_: _stats(X_raw[:, i]) for i, n_ in enumerate(names)},
        "scores": scores,
        "best_k": best_k,
        "best_silhouette": round(float(best_s), 4),
        "clusters": clusters,
        "crosstab_vs_heuristic": crosstab,
        "split_risk": split_risk,
        "_labels": {sid: int(l) for sid, l in zip(ids, best_labels)},
        "_X": X_raw,
        "_labels_arr": best_labels,
    }


# =============================================================================
# CHECK C - mask/image boundary agreement (annotation quality)
# =============================================================================

def check_boundary_quality(train_dir: str, stride: int = 1) -> Dict[str, Any]:
    """Measure whether annotation boundaries sit on real image edges.

    Establishes the label-noise ceiling. If masks are systematically dilated or
    eroded relative to the true nucleus edge, no architecture recovers that IoU;
    the fix is a boundary offset correction on the targets.

    Method per sample
    -----------------
    1. Sobel gradient magnitude, normalized by the image's own p01-p99 range so
       bright-field and fluorescence are on a comparable scale.
    2. edge_snap = mean gradient on the mask boundary ring / mean gradient on a
       background control ring 5-9 px outside the mask.
       >> 1  = annotations track real edges.  ~1 = annotations are arbitrary.
    3. offset scan: re-measure the boundary gradient for the mask eroded and
       dilated by k in [-3..+3]. The argmax k is the systematic offset in px.
       k=0 means the annotation is correctly registered.

    Parameters
    ----------
    train_dir : str
        Root holding one directory per sample.
    stride : int
        Process every Nth sample. 1 = full dataset.

    Returns
    -------
    dict
        edge_snap / offset stat blocks, offset histogram, flagged sample lists,
        and per-sample records.
    """
    root = Path(train_dir)
    samples = _sample_dirs(root)[::stride]

    offsets = np.arange(-3, 4)  # negative = erode, positive = dilate
    per_sample: List[Dict[str, Any]] = []

    for sdir in samples:
        img_path = next((sdir / "images").glob("*.png"), None)
        if img_path is None:
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape

        mask_paths = sorted((sdir / "masks").glob("*.png"))
        if not mask_paths:
            continue

        # Union only: boundary agreement is a pixel-level property, so instance
        # identity is unnecessary. One bool plane instead of a (N,H,W) stack.
        union = np.zeros((h, w), dtype=bool)
        for mp in mask_paths:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (h, w):
                continue
            union |= m > 127
            del m
        if not union.any():
            continue

        # --- normalized gradient magnitude -------------------------------------
        # Light blur first: Sobel on raw 8-bit amplifies shot noise, which would
        # make a random ring look as "edgy" as a real boundary.
        g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        # ksize=3 Sobel = 3x3 separable derivative. CV_32F output prevents the
        # negative-lobe clipping that an 8-bit ddepth would cause.
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # Robust scale: p99-p01 of the IMAGE, not the gradient, so the ratio is
        # dimensionless and cross-modality comparable.
        lo, hi = np.percentile(gray, [1, 99])
        rng_i = max(float(hi - lo), 1.0)
        mag /= rng_i

        u8 = union.astype(np.uint8)

        # --- control ring: background, 5-9 px outside the mask -----------------
        far = cv2.dilate(u8, _K3SQ, iterations=4)
        near = cv2.dilate(u8, _K3SQ, iterations=2)
        ctrl = (far - near).astype(bool)
        ctrl_mean = float(mag.mean(where=ctrl, dtype=np.float64)) if ctrl.any() else np.nan

        # --- offset scan --------------------------------------------------------
        scan = []
        for k in offsets:
            if k < 0:
                shifted = cv2.erode(u8, _K3SQ, iterations=int(-k))
            elif k > 0:
                shifted = cv2.dilate(u8, _K3SQ, iterations=int(k))
            else:
                shifted = u8
            if not shifted.any():
                scan.append(np.nan)
                continue
            # MORPH_GRADIENT = dilate - erode -> a ~2px boundary ring in one op.
            band = cv2.morphologyEx(shifted, cv2.MORPH_GRADIENT, _K3SQ).astype(bool)
            scan.append(float(mag.mean(where=band, dtype=np.float64)) if band.any() else np.nan)

        scan_a = np.array(scan, dtype=np.float64)
        b_mean = scan_a[offsets == 0][0]  # the as-annotated boundary score
        # nanargmax ignores degenerate offsets (mask eroded to nothing).
        best_off = int(offsets[int(np.nanargmax(scan_a))]) if np.isfinite(scan_a).any() else 0

        per_sample.append({
            "sample_id": sdir.name,
            "n_inst": len(mask_paths),
            "boundary_grad": round(b_mean, 4),
            "control_grad": round(ctrl_mean, 4),
            # The headline metric.
            "edge_snap": round(float(b_mean / ctrl_mean), 3) if ctrl_mean and np.isfinite(ctrl_mean) else np.nan,
            "best_offset_px": best_off,
            # Relative gain from shifting: >5% means a real registration error,
            # not scan noise.
            "offset_gain_pct": round(100 * float(np.nanmax(scan_a) / b_mean - 1.0), 2) if b_mean else 0.0,
        })

        del gray, g, gx, gy, mag, union, u8

    snap = np.array([r["edge_snap"] for r in per_sample], dtype=np.float64)
    snap_f = snap[np.isfinite(snap)]
    off = np.array([r["best_offset_px"] for r in per_sample], dtype=np.float64)
    gain = np.array([r["offset_gain_pct"] for r in per_sample], dtype=np.float64)

    return {
        "n_samples": len(per_sample),
        "edge_snap": _stats(snap_f),
        "offset_px": _stats(off),
        "offset_gain_pct": _stats(gain),
        # Counter over int offsets -> the registration histogram. A mass at 0
        # means well-registered annotations.
        "offset_hist": dict(sorted(Counter(int(o) for o in off).items())),
        "well_registered_pct": round(100 * float((off == 0).mean()), 2) if off.size else 0.0,
        # snap <= 1.1 => boundary is no edgier than random background.
        "arbitrary_boundary_samples": [r["sample_id"] for r in per_sample
                                       if np.isfinite(r["edge_snap"]) and r["edge_snap"] <= 1.1],
        "worst_10": sorted((r for r in per_sample if np.isfinite(r["edge_snap"])),
                           key=lambda r: r["edge_snap"])[:10],
        "per_sample": per_sample,
        "arrays": {"edge_snap": snap_f, "offset_px": off, "offset_gain_pct": gain},
    }
