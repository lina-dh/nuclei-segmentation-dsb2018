"""Data access layer for the 2018 Data Science Bowl nuclei dataset.

Dataset layout on disk (one directory per sample, confirmed from stage1_train.zip):

    data/stage1_train/
        <sample_id>/
            images/<sample_id>.png      <- 1 RGBA microscopy image
            masks/<mask_id>.png         <- N binary masks, 1 per nucleus instance

Design rules, all derived from the frozen EDA facts in .project_state/milestones.md:

1. INSTANCES ARE STORED AS AN int32 LABEL MAP, NEVER AS AN (N, H, W) STACK.
   EDA proved 0 overlapping pixels across all 29,461 instances, so a single
   (H, W) label map is a LOSSLESS encoding. Cost comparison for the worst
   sample (375 masks @ ~1000x1000):  bool stack = 375 MB   vs   int32 map = 4 MB.

2. THE TRAINING TARGET IS 3-CLASS {0=bg, 1=interior, 2=border}, NOT BINARY.
   78.1% of samples contain touching instances (5,247 touching pairs), which a
   binary foreground mask physically cannot separate. The border class creates a
   2px wall between abutting nuclei that watershed can cut at inference.

3. BRIGHTFIELD CONTRAST IS INVERTED (fg-bg = -77.6 vs +48.7 for fluorescence).
   `normalize_image` optionally flips polarity so "nuclei are bright" holds for
   every sample, which is what a single shared encoder needs.

Loaders return NumPy arrays only, never file handles, so callers cannot leak
descriptors across the 670-sample iteration.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import cv2
import numpy as np

# Background-intensity cut separating the two acquisition modalities.
# Validated in EDA: k=2 clustering (silhouette 0.776) agreed with this cheap
# threshold on 100% of 670 samples, so no clustering is needed at runtime.
MODALITY_BG_THRESHOLD = 90

# 3x3 CROSS kernel = 4-connectivity. Cross (not square) is deliberate: with a
# square kernel a diagonal-only neighbour counts as adjacent, which thickens the
# border ring on every staircase edge and over-erodes small nuclei.
_K3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Semantic class codes for the 3-class target. Named so downstream loss weights
# and watershed seeding never index with magic numbers.
BG, INTERIOR, BORDER = 0, 1, 2


def list_samples(train_dir: str | Path) -> List[Path]:
    """Sorted sample directories under ``train_dir``.

    sorted() is load-bearing: it makes sample ORDER deterministic, which is what
    makes a seeded train/val split reproducible across machines and runs.
    """
    root = Path(train_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found - extract stage1_train.zip first")
    # Generator inside sorted(): iterdir() never materializes a 670-element list
    # of its own. The is_dir() filter drops stray files (.DS_Store, .gitkeep).
    return sorted(p for p in root.iterdir() if p.is_dir())


def load_image(sample_dir: str | Path) -> np.ndarray:
    """Load one sample's microscopy image as ``(H, W, 3)`` uint8 RGB.

    Two syntax choices carry the whole function:
      * ``cv2.IMREAD_COLOR`` forces a 3-channel decode, which drops the alpha
        channel for free. EDA confirmed all 670 PNGs are RGBA with constant
        alpha, so nothing is lost and we avoid a manual ``img[..., :3]`` slice
        (that slice would keep a non-contiguous view onto a 4-channel buffer).
      * ``cv2.cvtColor(..., COLOR_BGR2RGB)`` because OpenCV decodes to BGR.
        Without it, matplotlib renders every image with red/blue swapped.
    """
    sample_dir = Path(sample_dir)
    images_dir = sample_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} missing")

    # next(glob, None) pulls only the FIRST match and abandons the generator -
    # cheaper than list(glob)[0] and safe when the directory is empty.
    img_path = next(images_dir.glob("*.png"), None)
    if img_path is None:
        raise FileNotFoundError(f"{images_dir} contains no PNG")

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        # cv2.imread RETURNS None on a corrupt/truncated file, it never raises.
        # Converting that to an exception here stops a silent None from
        # propagating into a training batch as a broadcast error 200 lines later.
        raise OSError(f"unreadable image: {img_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_label_map(
    sample_dir: str | Path,
    shape: Optional[Tuple[int, int]] = None,
    check_overlap: bool = False,
) -> Tuple[np.ndarray, int]:
    """Merge every instance mask PNG into a single int32 label map.

    Parameters
    ----------
    sample_dir : str | Path
        Sample folder containing ``masks/``.
    shape : (H, W), optional
        Expected spatial size. When given, any mask whose shape differs is
        skipped instead of raising - the audit already proved 0 mismatches, so
        this is a guard for future stage2 data, not an expected path.
    check_overlap : bool
        Count pixels that were already claimed by an earlier instance. EDA
        proved this is 0 dataset-wide; enable only when ingesting new data.

    Returns
    -------
    label : np.ndarray
        ``(H, W)`` int32. 0 = background, 1..N = instance ids in sorted-filename
        order. int32 (not uint8/uint16) because 375 instances already exceed
        uint8, and int32 is what cv2.connectedComponents / watershed consume
        without a copy.
    overlap_px : int
        Pixels overwritten by a later instance (0 unless ``check_overlap``).

    Notes
    -----
    Peak RAM = one label map + one decoded mask. Masks are decoded one at a
    time and folded in immediately, so a 375-instance sample costs ~5 MB total
    instead of the 375 MB a bool stack would cost.
    """
    sample_dir = Path(sample_dir)
    masks_dir = sample_dir / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"{masks_dir} missing")

    # sorted() -> instance ids are stable across runs, so a cached label map and
    # a freshly computed one agree index-for-index.
    mask_paths = sorted(masks_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"{masks_dir} is empty")

    label: Optional[np.ndarray] = None
    if shape is not None:
        label = np.zeros(shape, dtype=np.int32)

    overlap_px = 0
    next_id = 1

    for mp in mask_paths:
        # IMREAD_GRAYSCALE: masks are binary, so a 1-channel decode is 3x less
        # work than color AND gives .shape == (H, W) directly.
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue  # corrupt mask; audit_dataset() reports these separately

        if label is None:
            # No shape hint -> adopt the first mask's shape as the reference.
            label = np.zeros(m.shape, dtype=np.int32)
        elif m.shape != label.shape:
            continue

        # `m > 127` binarizes the {0, 255} PNG into a bool mask. Threshold at the
        # midpoint rather than `> 0` so PNG anti-aliasing artifacts (if any ever
        # appear) do not inflate every instance by a halo pixel.
        fg = m > 127
        if not fg.any():
            continue  # all-zero mask contributes no instance and must not burn an id

        if check_overlap:
            # label[fg] is boolean indexing -> 1-D copy of only the selected
            # pixels. np.count_nonzero on it counts pixels already owned.
            overlap_px += int(np.count_nonzero(label[fg]))

        # Boolean-mask ASSIGNMENT writes in place into the existing int32 buffer:
        # no temporary (H, W) array is allocated per instance, unlike
        # `label = np.where(fg, next_id, label)` which allocates a new map each time.
        label[fg] = next_id
        next_id += 1
        del m, fg  # release before the next decode

    if label is None:
        raise OSError(f"no readable masks in {masks_dir}")

    return label, overlap_px


def load_sample(
    sample_dir: str | Path,
    check_overlap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one nuclei sample as an RGB image plus an instance label map.

    Parameters
    ----------
    sample_dir : str | Path
        Path to a single sample folder, e.g. ``data/stage1_train/<sample_id>``.
    check_overlap : bool
        Forwarded to :func:`load_label_map`; raises if any instance overlap is
        found, since overlap would make the label-map encoding lossy.

    Returns
    -------
    image : np.ndarray
        ``(H, W, 3)`` uint8, RGB order, alpha dropped.
    label : np.ndarray
        ``(H, W)`` int32, 0 = background, 1..N = instance ids.

    Raises
    ------
    FileNotFoundError  : ``images/`` or ``masks/`` missing, or ``masks/`` empty.
    OSError            : a file exists but cannot be decoded.
    ValueError         : instance masks overlap (only when ``check_overlap``).
    """
    image = load_image(sample_dir)
    # image.shape[:2] slices (H, W) off (H, W, 3) -> the shape contract the
    # label map must satisfy. Passing it pre-allocates the map at the right size
    # and lets per-mask shape mismatches be detected instead of silently resized.
    label, overlap = load_label_map(
        sample_dir, shape=image.shape[:2], check_overlap=check_overlap
    )
    if check_overlap and overlap:
        raise ValueError(
            f"{Path(sample_dir).name}: {overlap} overlapping px - "
            "label-map encoding would be lossy"
        )
    return image, label


def make_3class_target(label: np.ndarray, border_width: int = 1) -> np.ndarray:
    """Convert an instance label map into the 3-class semantic target.

    Classes: ``0 = background``, ``1 = interior``, ``2 = border``.

    Why this representation
    -----------------------
    Each instance contributes its own 1px-thick boundary ring, so two abutting
    nuclei produce a 2px wall of class 2 between them. Watershed at inference
    seeds on connected components of class 1 and floods until it hits class 0 or
    class 2, which recovers instances a binary mask fuses permanently.

    The vectorized trick (this is the whole function)
    -------------------------------------------------
    Instead of looping over N instances with N full-image erosions, we run TWO
    grayscale morphology ops on the label map itself:

        lo = erode(label)   -> per-pixel MIN over the 3x3 cross neighbourhood
        hi = dilate(label)  -> per-pixel MAX over the same neighbourhood

    A foreground pixel of instance *i* is a boundary pixel of *i* iff some
    neighbour carries a different id - i.e. ``lo != i`` (a neighbour has a
    smaller id, including background 0) or ``hi != i`` (a neighbour has a larger
    id). That single expression captures BOTH instance-vs-background edges and
    instance-vs-instance edges in O(1) passes, independent of N.

    Why float32: cv2's erode/dilate accept 8U/16U/16S/32F/64F but NOT 32S, so
    the int32 map is viewed as float32. Safe here - float32 represents integers
    exactly up to 2^24, and the dataset max is 375 instances.

    Tiny-instance guard
    -------------------
    14.9% of instances are under 10px equivalent diameter; a 1-2px blob is
    ENTIRELY boundary, so naive ringing would leave it with no class-1 pixels and
    make it unseedable by watershed. Any instance left with an empty interior has
    its ring reverted to class 1, trading a possible merge (rare) for a
    guaranteed detection (common).

    Parameters
    ----------
    label : np.ndarray
        ``(H, W)`` integer label map from :func:`load_label_map`.
    border_width : int
        Ring thickness in pixels, applied as ``iterations`` of the 3x3 cross.
        1 keeps the EDA-measured ~1px label-noise floor from being amplified.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 with values in {0, 1, 2}. uint8 (not int64) because a
        3-value target needs 1 byte/px and this is what cross-entropy indexes.
    """
    if border_width < 1:
        raise ValueError("border_width must be >= 1")

    lab32 = label.astype(np.float32, copy=False)

    # iterations=border_width grows the neighbourhood to (2k+1) without building
    # a bigger kernel, keeping the 4-connectivity shape at every thickness.
    lo = cv2.erode(lab32, _K3, iterations=border_width)
    hi = cv2.dilate(lab32, _K3, iterations=border_width)

    fg = label > 0
    # (lo != lab32) | (hi != lab32): "some neighbour is not me". & fg restricts
    # the ring to the INSIDE of each instance, so instances never grow outward
    # and background pixels are never relabelled as border.
    ring = fg & ((lo != lab32) | (hi != lab32))

    # ---- tiny-instance guard --------------------------------------------------
    interior = fg & ~ring
    n = int(label.max())
    if n:
        # bincount over the FLAT label values of interior pixels, with
        # minlength=n+1 so ids absent from `interior` still get a 0 slot.
        # This is one O(px) pass instead of N boolean comparisons.
        interior_counts = np.bincount(label[interior].ravel(), minlength=n + 1)
        starved = np.flatnonzero(interior_counts[1:] == 0) + 1  # +1: ids are 1-based
        if starved.size:
            # np.isin builds the membership mask in C; a Python `for id in starved`
            # loop would cost one full-image compare per starved instance.
            revive = np.isin(label, starved)
            ring &= ~revive

    target = np.zeros(label.shape, dtype=np.uint8)
    # Assign interior FIRST, then overwrite with border, so the ring always wins
    # on contested pixels and the separating wall is never punched through.
    target[fg] = INTERIOR
    target[ring] = BORDER
    return target


def estimate_modality(image: np.ndarray) -> str:
    """Classify acquisition modality from the image alone: fluorescence|brightfield.

    EDA used the mask-derived ``bg_mean`` for this. At inference there are no
    masks, so we use the grayscale MEDIAN as a mask-free proxy: nuclei occupy
    only 14% of pixels on average (fg_fraction mean 0.14), so the median pixel
    is a background pixel in practice.

    np.median (not mean) is the point - the mean is pulled by bright nuclei in
    dense samples, the median is not.
    """
    gray = _to_gray(image)
    return "fluorescence" if float(np.median(gray)) < MODALITY_BG_THRESHOLD else "brightfield"


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Luminance view of an RGB or already-gray image. No copy when already 2-D."""
    if image.ndim == 2:
        return image
    # COLOR_RGB2GRAY applies 0.299R + 0.587G + 0.114B. Must be RGB2GRAY (not
    # BGR2GRAY) because load_image already converted channel order.
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_image(
    image: np.ndarray,
    flip_polarity: bool = True,
    to_float: bool = True,
) -> Tuple[np.ndarray, str]:
    """Put every modality on one convention: nuclei BRIGHT on a dark field.

    Rationale: brightfield samples have inverted contrast (fg-bg = -77.6) while
    fluorescence is positive (+48.7). A single encoder fed both must learn two
    opposite polarities from only 124 brightfield samples. Flipping at load
    removes that burden for free.

    Parameters
    ----------
    flip_polarity : bool
        Invert brightfield samples (``255 - x``). Detection uses
        :func:`estimate_modality`.
    to_float : bool
        Scale to float32 in [0, 1]. uint8 is kept otherwise, which is what the
        visualization cells want.

    Returns
    -------
    (array, modality) : the normalized image and the detected modality label.
    """
    modality = estimate_modality(image)
    out = image
    if flip_polarity and modality == "brightfield":
        # 255 - x on uint8 is a single vectorized pass; cv2.bitwise_not would do
        # the same but only for 8-bit, so the arithmetic form generalizes.
        out = 255 - out
    if to_float:
        # /255.0 promotes to float64 by default (4x the memory of float32), hence
        # the explicit astype first, then an in-place divide on the float32 copy.
        out = out.astype(np.float32)
        out /= 255.0
    return out, modality


# =============================================================================
# Dataset standardization (mean/std), applied AFTER the polarity flip
# =============================================================================
#
# Order is not optional: flip -> /255 -> (x - mean) / std.
# Standardizing BEFORE the flip would compute statistics over a bimodal
# population (bg 8.8 for fluorescence vs 180.3 for brightfield) whose mean sits
# in an empty valley between the two modes and describes neither. Post-flip,
# both modalities share the "nuclei bright on dark field" convention, so one
# mean/std is meaningful.
#
# Stats come from the TRAIN SPLIT ONLY. Including val leaks its pixel
# distribution into training and quietly inflates the val metric.

# ImageNet statistics, for the case where a pretrained encoder is used: its
# early filters were fitted under exactly this normalization.
NORM_IMAGENET = (
    np.array([0.485, 0.456, 0.406], dtype=np.float32),
    np.array([0.229, 0.224, 0.225], dtype=np.float32),
)

NORM_STATS_JSON = "norm_stats.json"


def compute_norm_stats(
    cache_dir: str | Path,
    sample_ids: List[str],
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Exact per-channel mean/std over the CACHED (already polarity-flipped) images.

    Method: accumulate a 256-bin histogram per channel, then derive the moments
    from it. Why a histogram instead of summing pixels:

      * The pixels are uint8, so there are only 256 distinct values. A histogram
        is a LOSSLESS summary of the whole dataset in a (3, 256) int64 array.
      * Moments come out EXACT - no float drift from accumulating ~500M values.
      * Memory is O(1): no float64 promotion of a 1.4 Mpx image, no running sum
        of squares that can overflow or lose precision.
      * np.bincount is a single C pass per channel.

    Variance uses E[x^2] - E[x]^2 computed in float64 on the histogram, which is
    safe here because the bin values are small integers (0-255) and exactly
    representable - the usual catastrophic-cancellation objection to this
    identity does not apply.

    Parameters
    ----------
    cache_dir : str | Path
        Directory of ``<id>_img.npy`` files written by :func:`write_cache_arrays`.
    sample_ids : list of str
        TRAIN split ids only.
    out_path : str | Path, optional
        Where to write the JSON. Pass ``.project_state/norm_stats.json``.

    Returns
    -------
    dict
        ``{"mean": [3 floats], "std": [3 floats], "n_samples", "n_pixels", ...}``
        in [0, 1] scale, i.e. directly usable after the ``/255`` step.
    """
    cache_dir = Path(cache_dir)
    # (3, 256) int64: channel x pixel value. int64 because 500M pixels overflows
    # int32 in a single bin for a near-constant background.
    hist = np.zeros((3, 256), dtype=np.int64)
    n_px = 0
    n_used = 0

    for sid in sample_ids:
        p = cache_dir / f"{sid}_img.npy"
        if not p.is_file():
            continue
        # mmap: the histogram pass touches each page once and never needs the
        # whole 1.4 Mpx image resident.
        arr = np.load(p, mmap_mode="r")
        n_px += arr.shape[0] * arr.shape[1]
        n_used += 1
        for c in range(3):
            # .ravel() on a non-contiguous channel slice copies that channel
            # only (1 byte/px), which bincount requires as a flat 1-D input.
            hist[c] += np.bincount(np.asarray(arr[:, :, c]).ravel(), minlength=256)
        del arr

    if n_used == 0:
        raise FileNotFoundError(f"no cached images found in {cache_dir}")

    vals = np.arange(256, dtype=np.float64)
    total = hist.sum(axis=1).astype(np.float64)        # pixels per channel
    mean255 = (hist * vals).sum(axis=1) / total        # E[x]   in 0-255
    msq255 = (hist * vals ** 2).sum(axis=1) / total    # E[x^2] in 0-255
    var255 = np.maximum(msq255 - mean255 ** 2, 0.0)    # clamp: -1e-12 -> 0

    stats = {
        # /255 so the numbers apply directly to the float32 [0,1] images the
        # collate step produces.
        "mean": [round(float(v), 6) for v in mean255 / 255.0],
        "std": [round(float(v), 6) for v in np.sqrt(var255) / 255.0],
        "mean_255": [round(float(v), 3) for v in mean255],
        "std_255": [round(float(v), 3) for v in np.sqrt(var255)],
        "n_samples": n_used,
        "n_pixels": int(n_px),
        "note": "post-polarity-flip, train split only",
    }

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=1)
    return stats


def load_norm_stats(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``norm_stats.json`` as ``(mean, std)`` float32 arrays of shape (3,)."""
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return (
        np.asarray(s["mean"], dtype=np.float32),
        np.asarray(s["std"], dtype=np.float32),
    )


def resolve_norm(norm: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Normalize the many accepted ``norm`` spellings into ``(mean, std)`` or None.

    Accepts: ``None``/``False`` (no standardization), ``"imagenet"``, a path to a
    norm_stats.json, or an explicit ``(mean, std)`` pair.
    """
    if norm is None or norm is False:
        return None
    if isinstance(norm, str):
        if norm == "imagenet":
            return NORM_IMAGENET
        return load_norm_stats(norm)       # treat any other string as a path
    if isinstance(norm, (tuple, list)) and len(norm) == 2:
        return (
            np.asarray(norm[0], dtype=np.float32).reshape(-1),
            np.asarray(norm[1], dtype=np.float32).reshape(-1),
        )
    raise TypeError(f"unsupported norm spec: {norm!r}")


# =============================================================================
# L2 - decode-once cache
# =============================================================================
#
# Why a cache is MANDATORY for dynamic cropping (not an optimization):
# a random 256x256 crop of the 1040x1388 sample needs a 1.4 Mpx PNG decode to
# keep 65k pixels - 95% of the decode is thrown away, every epoch, and the GPU
# starves. Decoding once into .npy turns every subsequent crop into a memcpy.
#
# Footprint per pixel: img uint8x3 (3B) + label uint16 (2B) + target uint8 (1B)
# = 6 B/px. Across ~167 Mpx of dataset that is ~1.0 GB, ~0.8 GB for the train
# split, which is why the train split is RAM-resident and val is memmapped.

_CACHE_KINDS = ("img", "lab", "tgt")


def write_cache_arrays(
    cache_dir: str | Path,
    sample_id: str,
    image: np.ndarray,
    label: np.ndarray,
    modality: str,
) -> None:
    """Write the three L2 cache arrays for one sample.

    Three decisions are baked in here, all load-bearing:

    1. POLARITY IS FLIPPED AT CACHE TIME, never at crop time. The flip decision
       needs a WHOLE-IMAGE statistic: a 256x256 crop of dense nuclei has a high
       median and would be misclassified as brightfield and wrongly inverted.
       Here the authoritative manifest modality (mask-derived bg_mean) is known.
    2. THE 3-CLASS TARGET IS RINGED ON THE FULL LABEL MAP, before any crop.
       Ringing a cropped map paints a false border wall along the crop seam and
       teaches the model that image edges are nucleus boundaries.
    3. uint16 label map: 375 max instances fits in 16 bits and it halves int32's
       share of a ~1 GB cache.

    np.save (not np.savez) per array so val can be opened with
    ``mmap_mode='r'`` - npz is a zip container and cannot be memory-mapped.
    """
    cache_dir = Path(cache_dir)
    # 255 - x inverts an 8-bit image in one vectorized pass.
    img = 255 - image if modality == "brightfield" else image
    # ascontiguousarray: 255-x already yields a fresh C-contiguous buffer, but
    # the fluorescence branch passes the decoded array straight through, and a
    # non-contiguous input would make np.save write a slow element-wise copy.
    np.save(cache_dir / f"{sample_id}_img.npy", np.ascontiguousarray(img, dtype=np.uint8))
    np.save(cache_dir / f"{sample_id}_lab.npy", label.astype(np.uint16, copy=False))
    np.save(cache_dir / f"{sample_id}_tgt.npy", make_3class_target(label))


class SampleCache:
    """Random-access store of pre-decoded (image, label, target) per sample.

    Parameters
    ----------
    cache_dir : str | Path
        Directory written by :func:`write_cache_arrays`.
    sample_ids : list of str
        Only these samples are opened - pass the train split for the RAM cache
        and the val split for the memmap, so neither pays for the other.
    mode : {"ram", "mmap"}
        ``ram``  -> np.load reads the whole array into process memory.
        ``mmap`` -> np.load(mmap_mode='r') returns a lazy view; pages are faulted
        in by the OS only when a crop touches them, and the page cache is shared.
        Use ram for the hot training path, mmap for val/inference.
    keep_label : bool
        False drops the instance label map (saves 2 B/px, ~1/3 of the cache).
        Training on the 3-class target does not need it; val metrics do.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        sample_ids: List[str],
        mode: str = "ram",
        keep_label: bool = True,
    ) -> None:
        if mode not in ("ram", "mmap"):
            raise ValueError("mode must be 'ram' or 'mmap'")
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.keep_label = keep_label
        # mmap_mode=None makes np.load read eagerly; 'r' makes it a read-only map.
        mm = None if mode == "ram" else "r"

        self._img: Dict[str, np.ndarray] = {}
        self._lab: Dict[str, np.ndarray] = {}
        self._tgt: Dict[str, np.ndarray] = {}
        missing: List[str] = []

        for sid in sample_ids:
            paths = {k: self.cache_dir / f"{sid}_{k}.npy" for k in _CACHE_KINDS}
            if not all(p.is_file() for p in paths.values()):
                missing.append(sid)
                continue
            self._img[sid] = np.load(paths["img"], mmap_mode=mm)
            self._tgt[sid] = np.load(paths["tgt"], mmap_mode=mm)
            if keep_label:
                self._lab[sid] = np.load(paths["lab"], mmap_mode=mm)

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples absent from {self.cache_dir} "
                f"(first: {missing[0]}) - run build_manifest(cache_dir=...) first"
            )

    def __len__(self) -> int:
        return len(self._img)

    @property
    def nbytes(self) -> int:
        """Resident bytes. For mmap mode this is the MAPPED size, not RSS."""
        total = sum(a.nbytes for a in self._img.values())
        total += sum(a.nbytes for a in self._tgt.values())
        total += sum(a.nbytes for a in self._lab.values())
        return total

    def get(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return ``(image, target3, label_or_None)`` for one sample. No copy."""
        return (
            self._img[sample_id],
            self._tgt[sample_id],
            self._lab.get(sample_id),
        )


# =============================================================================
# L5 - augmentation (albumentations)
# =============================================================================
#
# Why albumentations and not hand-rolled numpy: a geometric transform must be
# applied JOINTLY and IDENTICALLY to the image, the 3-class target, the instance
# label map and the ignore mask. Hand-rolling that means re-deriving the same
# random parameters four times and keeping them in sync forever. albumentations
# samples the parameters once per call and dispatches them to every declared
# target, and it knows which transforms are pixel-level (image only) vs
# spatial-level (all targets) - so intensity jitter can never corrupt a mask.
#
# Transform policy (locked in milestones.md):
#   * Geometry = the DIHEDRAL GROUP only (A.D4: 4 rotations x mirror = 8 maps).
#     Microscopy has no canonical orientation, so this is a free 8x expansion,
#     and every map is a pure index permutation -> ZERO interpolation, so class
#     codes {0,1,2} and instance ids survive exactly.
#   * Scale jitter is UPSAMPLE-ONLY [1.0, 1.25] and off by default: median
#     nucleus diameter is 19.7 px and 14.9% of instances are under 10 px, so any
#     downsample destroys instances irrecoverably.
#   * Everything else is intensity-only: brightness/contrast, gamma, Gaussian
#     noise, mild blur - the real nuisance variation across microscopes.

# Target-name -> albumentations target TYPE. "mask" type means nearest-neighbour
# / index-preserving handling and participation in spatial transforms only.
_AUG_ADDITIONAL_TARGETS = {"label": "mask", "ignore": "mask"}


def build_augmenter(
    scale_jitter: bool = False,
    p_intensity: float = 0.8,
    seed: Optional[int] = None,
):
    """Build the L5 albumentations pipeline. Applied POST-crop, at 256x256.

    Parameters
    ----------
    scale_jitter : bool
        Enable the upsample-only Affine scale [1.0, 1.25]. Off by default.
    p_intensity : float
        Probability that the intensity block runs at all. <1 keeps a fraction of
        clean crops in the epoch so the model still sees un-jittered statistics.
    seed : int, optional
        Fixes the augmentation RNG. ``CropSampler`` calls
        ``set_random_seed(seed, epoch)`` per epoch, so epochs stay replayable.

    Returns
    -------
    albumentations.Compose

    Notes
    -----
    Applied after cropping, never before: augmenting a 1040x1388 image to then
    keep 65k pixels would waste ~95% of the work. At 256x256 the whole pipeline
    costs microseconds.
    """
    try:
        import albumentations as A
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "L5 requires albumentations: pip install albumentations"
        ) from e

    transforms = [
        # D4 = the full symmetry group of the square, in ONE transform. Cheaper
        # and more complete than stacking HorizontalFlip + VerticalFlip +
        # RandomRotate90 (which can compose to the same 8 maps but with a
        # non-uniform distribution over them).
        A.D4(p=1.0),
    ]

    if scale_jitter:
        transforms.append(
            A.Affine(
                scale=(1.0, 1.25),          # upsample only - never shrink nuclei
                translate_percent=0.0,      # translation is already the crop offset
                rotate=0.0, shear=0.0,      # rotation is already covered by D4
                interpolation=cv2.INTER_LINEAR,       # smooth for the IMAGE
                mask_interpolation=cv2.INTER_NEAREST,  # NEAREST for masks: linear
                                                       # interpolation would invent
                                                       # class 1.5 / fractional ids
                p=0.3,
            )
        )

    transforms += [
        # OneOf picks exactly ONE member per call (weighted by their p) - keeps
        # the augmentation mild instead of stacking four distortions at once.
        A.OneOf([
            # limit=0.2 -> +/-20%. Larger would push fluorescence crops toward
            # the brightfield intensity regime the polarity flip just removed.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            # std_range is RELATIVE to the dtype max in albumentations 2.x, so
            # (0.01, 0.05) is sigma 2.5-12.75 on uint8 - sensor-noise scale, not
            # the 0.2 default which would be sigma 51 and erase small nuclei.
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            # blur_limit kernel 3-5 px. Beyond that, a 10 px nucleus is gone.
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=p_intensity),
    ]

    return A.Compose(
        transforms,
        # Declares that `label` and `ignore` are mask-type targets. Passing only
        # a subset per call is allowed (verified), so the sampler can omit
        # `ignore` on unpadded crops and `label` when the cache dropped it.
        additional_targets=_AUG_ADDITIONAL_TARGETS,
        seed=seed,
    )


# =============================================================================
# L3/L4 - sampler + crop
# =============================================================================

class CropSampler:
    """Dynamic random-crop sampler with an explicit, 3-level sampling policy.

    The whole dataset-balance argument lives in these three levels. Each is an
    independent dial, which is exactly what pre-computed tiling cannot offer
    (there, balance is welded to pixel area):

    LEVEL 1 - MODALITY QUOTA (``modality_mix``)
        Pick the modality FIRST, from a fixed probability. A fixed quota means
        neither modality can starve no matter the 546:124 sample imbalance, and
        the ratio is a hyperparameter to sweep against per-modality val mAP.
        Area-proportional tiling would instead push brightfield from 18.5% of
        samples to ~82% of the epoch - an uncontrolled over-correction.

    LEVEL 2 - SUB-LINEAR AREA WEIGHTING (``alpha``)
        Within the chosen modality, sample a sample with weight ``area**alpha``.
        alpha=0 is per-image uniform (big images badly under-covered); alpha=1 is
        tiling-equivalent (one 1040x1388 image outvotes twenty 256x256 ones).
        alpha=0.35 turns the dataset's 22x area spread into only ~2.9x sampling
        weight: coverage for large images WITHOUT domination. This is the move
        that decouples spatial coverage from class balance.

    LEVEL 3 - FOREGROUND-BIASED CENTERING (``centroid_p``)
        With probability ``centroid_p`` the crop centers on a random instance
        centroid plus jitter; otherwise uniformly. Rationale: fg_fraction is only
        0.14, so uniform-only centering wastes crops on empty field, while
        foreground-only centering removes true negatives and causes a
        false-positive explosion on blank regions at inference.

    Parameters
    ----------
    cache : SampleCache
    manifest : dict
        Parsed manifest.json (needs ``samples`` with h/w/area/modality/split).
    instances : dict of np.ndarray
        instances.npz contents (needs ``sample_idx`` and ``centroid``).
    split : str
        Which split to draw from ("train" or "val").
    crop : int
        Square crop side in pixels.
    modality_mix : float
        P(fluorescence) per draw. 0.65 is the locked default.
    alpha : float
        Area exponent for level 2. 0.35 is the locked default.
    centroid_p : float
        P(center on an instance) for level 3. 0.70 is the locked default.
    augment : albumentations.Compose | bool | None
        L5 pipeline applied POST-crop. ``True`` builds the default via
        :func:`build_augmenter`; ``None``/``False`` disables it (use for val).
    norm : None | "imagenet" | path | (mean, std)
        Dataset standardization applied in collate, AFTER ``/255``.
        DEFAULT IS ``"imagenet"`` because the locked architecture uses an
        ImageNet-pretrained encoder, whose early filters were fitted under
        exactly those statistics; feeding it dataset stats (mean ~0.097 vs
        0.485) shifts every activation at layer 1. Pass the path to
        ``norm_stats.json`` instead for a from-scratch encoder, or ``None`` to
        keep raw [0,1]. The SAME choice must be used for train, val AND
        inference, or the encoder sees a shifted distribution at test time.
    seed : int
        Base seed; ``iter_epoch(epoch=k)`` derives a per-epoch stream from it so
        any epoch is exactly replayable.
    """

    def __init__(
        self,
        cache: SampleCache,
        manifest: Dict[str, Any],
        instances: Dict[str, np.ndarray],
        split: str = "train",
        crop: int = 256,
        modality_mix: float = 0.65,
        alpha: float = 0.35,
        centroid_p: float = 0.70,
        augment: Any = None,
        norm: Any = "imagenet",
        seed: int = 0,
    ) -> None:
        self.crop = int(crop)
        self.modality_mix = float(modality_mix)
        self.alpha = float(alpha)
        self.centroid_p = float(centroid_p)
        self.seed = int(seed)
        self.cache = cache
        self.split = split
        # `augment is True` (identity check, not truthiness) distinguishes the
        # "build me the default" sentinel from an already-constructed Compose,
        # which is also truthy.
        self.augment = build_augmenter(seed=seed) if augment is True else (augment or None)
        # Resolved once at construction: re-reading norm_stats.json per batch
        # would be a file open per step.
        self.norm = resolve_norm(norm)

        recs = manifest["samples"]
        # Global manifest row index is the key that joins the JSON records to the
        # npz instance rows (instances["sample_idx"] stores that same index).
        self.records = [(i, r) for i, r in enumerate(recs) if r["split"] == split]
        if not self.records:
            raise ValueError(f"no samples with split={split!r}")

        # ---- level 1+2 precomputation -----------------------------------------
        # One index list and one weight vector per modality group, computed once
        # at construction. Doing this per draw would re-normalize 536 weights
        # 3000 times per epoch.
        self._groups: Dict[str, Dict[str, Any]] = {}
        for mod in ("fluorescence", "brightfield"):
            rows = [(gi, r) for gi, r in self.records if r["modality"] == mod]
            if not rows:
                continue
            w = np.array([r["area"] for _, r in rows], dtype=np.float64) ** self.alpha
            self._groups[mod] = {
                "rows": rows,
                # Pre-normalized: rng.choice(p=...) requires sum(p)==1 and
                # re-validates it on every call, so normalize once here.
                "p": w / w.sum(),
            }
        if not self._groups:
            raise ValueError("no modality groups built")

        # ---- level 3 precomputation -------------------------------------------
        # instances["sample_idx"] is sorted ascending (built sample-by-sample in
        # build_manifest), so searchsorted gives each sample's contiguous slice
        # in O(log M) with no per-sample Python filtering.
        self._centroids = instances["centroid"]
        sidx = instances["sample_idx"]
        self._inst_lo = np.searchsorted(sidx, np.arange(len(recs)), side="left")
        self._inst_hi = np.searchsorted(sidx, np.arange(len(recs)), side="right")

        # Shared all-valid ignore mask for the (overwhelmingly common) unpadded
        # case: one 64 KB buffer instead of one allocation per crop. Marked
        # read-only so a caller cannot silently corrupt every future crop.
        self._no_ignore = np.zeros((self.crop, self.crop), dtype=bool)
        self._no_ignore.flags.writeable = False

        # Diagnostics, accumulated across all draws from this sampler.
        self.stats = Counter()

    # -- level 1 ---------------------------------------------------------------
    def _pick_modality(self, rng: np.random.Generator) -> str:
        """Fixed-quota modality draw, with a graceful fallback if a group is empty."""
        want = "fluorescence" if rng.random() < self.modality_mix else "brightfield"
        if want in self._groups:
            return want
        # A split missing a modality entirely (never true here, but true for any
        # single-modality subset) must degrade to the available group, not crash.
        return next(iter(self._groups))

    # -- level 2 ---------------------------------------------------------------
    def _pick_sample(self, rng: np.random.Generator, modality: str) -> Tuple[int, Dict[str, Any]]:
        g = self._groups[modality]
        # Draw an INDEX, not the object: rng.choice on a list of dicts would
        # force conversion to an object-dtype array on every call.
        j = rng.choice(len(g["rows"]), p=g["p"])
        return g["rows"][j]

    # -- level 3 ---------------------------------------------------------------
    def _pick_center(
        self, rng: np.random.Generator, gi: int, h: int, w: int
    ) -> Tuple[int, int]:
        """Crop center: instance centroid + jitter, or uniform."""
        lo, hi = int(self._inst_lo[gi]), int(self._inst_hi[gi])
        if hi > lo and rng.random() < self.centroid_p:
            k = rng.integers(lo, hi)
            cy, cx = self._centroids[k]
            # Jitter at +/- crop/4: enough that the same nucleus is seen at many
            # positions (translation augmentation), small enough that it stays
            # inside the crop instead of landing on the edge.
            j = self.crop // 4
            cy += rng.integers(-j, j + 1)
            cx += rng.integers(-j, j + 1)
            self.stats["center_instance"] += 1
        else:
            cy = rng.integers(0, h)
            cx = rng.integers(0, w)
            self.stats["center_uniform"] += 1
        return int(cy), int(cx)

    # -- L4: the crop ----------------------------------------------------------
    def _crop_at(
        self, arr: np.ndarray, top: int, left: int, pad_mode: int
    ) -> np.ndarray:
        """Slice a crop-sized window, replicate-padding when the image is smaller.

        The slice itself is a VIEW (zero copy). np.array(...) then makes it a
        C-contiguous copy, which is required: downstream augmentation writes in
        place, and writing through a view would permanently corrupt the cache.
        """
        c = self.crop
        win = np.array(arr[top:top + c, left:left + c])
        if win.shape[0] == c and win.shape[1] == c:
            return win
        # Defensive path only: stage1_train's minimum dimension is exactly 256,
        # so this never fires here. Kept for stage2 / external data.
        # BORDER_REPLICATE, not zero-padding: a black border mimics fluorescence
        # background and would shift the encoder's normalization statistics.
        return cv2.copyMakeBorder(
            win, 0, c - win.shape[0], 0, c - win.shape[1], pad_mode
        )

    def sample_one(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Draw a single crop through all three sampling levels."""
        c = self.crop
        modality = self._pick_modality(rng)
        gi, rec = self._pick_sample(rng, modality)
        sid, h, w = rec["sample_id"], rec["h"], rec["w"]

        cy, cx = self._pick_center(rng, gi, h, w)
        # Convert center -> top-left, then clamp so the window stays in bounds.
        # max(0, h - c) handles h < c: the clamp range collapses to [0, 0] and
        # the padding path in _crop_at takes over.
        top = int(np.clip(cy - c // 2, 0, max(0, h - c)))
        left = int(np.clip(cx - c // 2, 0, max(0, w - c)))

        img, tgt, lab = self.cache.get(sid)
        # INTER-mode constants differ per array: the image replicates pixel
        # values, while target/label must replicate CLASS/ID codes - same
        # BORDER_REPLICATE op, but never an interpolating one.
        img_c = self._crop_at(img, top, left, cv2.BORDER_REPLICATE)
        tgt_c = self._crop_at(tgt, top, left, cv2.BORDER_REPLICATE)
        lab_c = self._crop_at(lab, top, left, cv2.BORDER_REPLICATE) if lab is not None else None

        padded = h < c or w < c
        if padded:
            # Mark the synthetic region so the loss can ignore_index it: real
            # supervision does not exist there.
            ignore = np.zeros((c, c), dtype=bool)
            ignore[min(h, c):, :] = True
            ignore[:, min(w, c):] = True
            self.stats["padded"] += 1
        else:
            ignore = self._no_ignore

        # ---- L5: joint augmentation -------------------------------------------
        if self.augment is not None:
            # Build the kwargs conditionally: albumentations accepts a SUBSET of
            # the declared additional_targets, so an absent label map or an
            # unpadded crop costs no transform work at all.
            kw: Dict[str, Any] = {"image": img_c, "mask": tgt_c}
            if lab_c is not None:
                kw["label"] = lab_c
            if padded:
                kw["ignore"] = ignore
            out = self.augment(**kw)
            # One sampled parameter set is applied to every target, which is the
            # entire reason for using the library: image and masks stay aligned.
            img_c, tgt_c = out["image"], out["mask"]
            if lab_c is not None:
                lab_c = out["label"]
            if padded:
                ignore = out["ignore"]
            self.stats["augmented"] += 1

        self.stats["crops"] += 1
        self.stats[f"mod_{modality}"] += 1
        # An "empty" crop has no interior pixels at all. Tracked (not rejected)
        # because some pure-background crops are wanted as true negatives; the
        # rate is the diagnostic that says whether centroid_p needs tuning.
        if not (tgt_c == INTERIOR).any():
            self.stats["empty_crops"] += 1

        return {
            "image": img_c,
            "target": tgt_c,
            "label": lab_c,
            "ignore": ignore,
            "sample_id": sid,
            "modality": modality,
            "top": top,
            "left": left,
            "padded": padded,
        }

    def iter_epoch(self, n: int, epoch: int = 0):
        """Yield ``n`` crops. Epoch length is DECOUPLED from image area by design.

        The per-epoch Generator is seeded from ``(seed, epoch)``, so epoch 7 is
        byte-identical on re-run - essential for debugging a loss spike.
        """
        rng = np.random.default_rng((self.seed, epoch))
        if self.augment is not None:
            # albumentations keeps its OWN RNG, independent of our Generator, so
            # it must be reseeded separately or the aug stream drifts between
            # runs even when the crop stream is identical.
            self.augment.set_random_seed(self.seed * 10_000 + epoch)
        for _ in range(n):
            yield self.sample_one(rng)

    def iter_batches(
        self,
        n: int,
        batch_size: int,
        epoch: int = 0,
        to_float: bool = True,
        channels_first: bool = False,
    ):
        """Yield stacked batches. All crops are crop x crop, so no ragged padding.

        to_float=True converts images to float32 in [0, 1]. The conversion is
        done ONCE per batch on the stacked array rather than per crop: one
        vectorized pass over B*C*H*W instead of B small ones.

        channels_first=True returns NCHW instead of NHWC, which is what
        torch.nn.Conv2d expects. Requires to_float (a uint8 NCHW batch would
        just be transposed again later).
        """
        buf: List[Dict[str, Any]] = []
        for item in self.iter_epoch(n, epoch=epoch):
            buf.append(item)
            if len(buf) == batch_size:
                yield self._collate(buf, to_float, channels_first)
                buf = []
        if buf:  # trailing partial batch - kept, not dropped
            yield self._collate(buf, to_float, channels_first)

    def _collate(
        self,
        items: List[Dict[str, Any]],
        to_float: bool,
        channels_first: bool = False,
    ) -> Dict[str, Any]:
        # np.stack adds a NEW leading batch axis (vs np.concatenate which would
        # glue crops along the height axis into one tall image).
        img = np.stack([it["image"] for it in items])
        if to_float:
            img = img.astype(np.float32)
            img /= 255.0  # in-place on the fresh float32 buffer: no second copy
            if self.norm is not None:
                mean, std = self.norm
                # Broadcasting: img is (B,H,W,3) and mean/std are (3,), so NumPy
                # aligns them on the LAST axis automatically - no reshape needed,
                # and both ops are in-place so no extra B*H*W*3 temporary.
                img -= mean
                img /= std
        elif self.norm is not None:
            raise ValueError("norm requires to_float=True (cannot standardize uint8)")

        if channels_first:
            if not to_float:
                raise ValueError("channels_first requires to_float=True")
            # (B,H,W,C) -> (B,C,H,W). transpose returns a VIEW with permuted
            # strides; ascontiguousarray materializes it, which torch needs for
            # a zero-copy from_numpy (a strided view would force a copy later
            # anyway, so pay it once here).
            img = np.ascontiguousarray(img.transpose(0, 3, 1, 2))

        out = {
            "image": img,
            "target": np.stack([it["target"] for it in items]),
            "ignore": np.stack([it["ignore"] for it in items]),
            "sample_ids": [it["sample_id"] for it in items],
            "modalities": [it["modality"] for it in items],
        }
        if items[0]["label"] is not None:
            out["label"] = np.stack([it["label"] for it in items])
        return out

    def report(self) -> Dict[str, Any]:
        """Scalar diagnostics for the sampling policy. No arrays (lean-ctx rule)."""
        n = max(1, self.stats["crops"])
        return {
            "crops": self.stats["crops"],
            "pct_fluorescence": round(100 * self.stats["mod_fluorescence"] / n, 2),
            "pct_brightfield": round(100 * self.stats["mod_brightfield"] / n, 2),
            "pct_center_instance": round(100 * self.stats["center_instance"] / n, 2),
            "pct_empty_crops": round(100 * self.stats["empty_crops"] / n, 2),
            "padded_crops": self.stats["padded"],
        }


def compute_class_frequencies(
    sampler: "CropSampler", n: int = 2000, epoch: int = 0,
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Measure 3-class pixel frequencies on the ACTUAL crop stream.

    Lives here, not in src.losses, for two reasons: it is pure numpy (so it
    stays importable in an environment without torch) and it needs the sampler.

    Why measure on crops rather than on whole cached targets: the sampler
    deliberately biases toward instance centroids (70% of draws), so its output
    is richer in foreground than the raw images average to. Loss weights must
    match what the model is actually fed.

    Ignored (padded) pixels are excluded, since they contribute no loss.
    """
    counts = np.zeros(3, dtype=np.int64)
    for it in sampler.iter_epoch(n, epoch=epoch):
        tgt, ign = it["target"], it["ignore"]
        if ign.any():
            # Boolean indexing flattens to the kept pixels only.
            tgt = tgt[~ign]
        # minlength pins 3 bins even when a crop contains no border pixels.
        counts += np.bincount(tgt.ravel(), minlength=3)[:3]

    total = int(counts.sum())
    out = {
        "freqs": [round(float(c) / max(total, 1), 6) for c in counts],
        "counts": counts.tolist(),
        "class_order": ["background", "interior", "border"],
        "n_crops": n,
        "n_pixels": total,
    }
    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return out


def sample_summary(sample_dir: str | Path) -> Dict[str, Any]:
    """One-sample scalar report. Returns NO arrays, per the project lean-ctx rule.

    Used by notebook cells to verify the loader without printing pixel data.
    """
    image, label = load_sample(sample_dir)
    target = make_3class_target(label)
    # np.bincount over 3 classes is cheaper than three == comparisons and
    # returns the counts already aligned to the class codes.
    cls = np.bincount(target.ravel(), minlength=3)
    px = int(target.size)
    return {
        "sample_id": Path(sample_dir).name,
        "shape": tuple(image.shape),
        "n_instances": int(label.max()),
        "modality": estimate_modality(image),
        "pct_bg": round(100 * int(cls[BG]) / px, 2),
        "pct_interior": round(100 * int(cls[INTERIOR]) / px, 2),
        "pct_border": round(100 * int(cls[BORDER]) / px, 2),
    }
