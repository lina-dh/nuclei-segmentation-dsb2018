"""3-class semantic map -> instance label map, via seeded watershed.

This module is the other half of the 3-class target decision. The network emits
{0=background, 1=interior, 2=border}; the competition wants separated instances.
Watershed converts one into the other:

    SEEDS    = connected components of the INTERIOR class. The border ring is a
               closed wall (verified across all 670 samples), so two abutting
               nuclei yield two separate interior components rather than one.
    MASK     = interior | border, i.e. the full nucleus footprint. Growth is
               confined to it, so a region can never bleed into background.
    ELEVATION= -interior confidence. Watershed floods from the seeds uphill, so
               the ridge it stops on is the lowest-confidence line between two
               seeds - exactly the border ring.

Why not plain connected components on (interior|border)? Because that re-merges
every touching pair: 78.1% of samples contain at least one, 21.9% of all 29,461
instances touch a neighbour. The border wall exists specifically to be cut here.

MIN_AREA is data-derived, not guessed: the smallest real instance in the
dataset is 21 px, so anything at or above 21 would delete genuine nuclei.
14.9% of instances are under a 10 px equivalent diameter, which is why this
default is deliberately tiny and why downsampling was ruled out in EDA.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from src.data_loader import BG, BORDER, INTERIOR

# SAFETY FLOOR. Smallest real instance is 21 px (verified from the manifest's
# 29,461 areas), so 15 sits below that with margin: it cannot delete a correct
# prediction of even the smallest nucleus.
MIN_AREA = 15

# MEASURED OPERATING POINT (scripts/sweep_postprocess.py against
# runs/run40/best.pt, epoch 30, all 135 val images, 2026-10-12).
#
# Greedy coordinate ascent, run twice until the seed optimum stopped moving,
# with every grid widened until its optimum was INTERIOR rather than at a
# boundary. Baseline (0.50 / 0.50 / 15) scored 0.5738; these score 0.6078
# without TTA and 0.6192 with D4-flip TTA.
#
# min_area 30 EXCEEDS the 21 px safety floor deliberately. The floor protects
# against deleting a true small nucleus; the sweep shows that spurious small
# predictions cost more as false positives than the rare small true nucleus
# they would occasionally remove. That is an empirical trade, so it is a
# separate constant and MIN_AREA stays conservative.
#
# NOTE the TTA interaction: averaging over flips sharpens interior confidence,
# so the best seed threshold DROPS from 0.80 to 0.70 and min_area from 30 to 25.
# Re-tune these whenever the inference path changes.
TUNED = {"seed_threshold": 0.80, "fg_threshold": 0.80, "min_area": 30}
TUNED_TTA = {"seed_threshold": 0.70, "fg_threshold": 0.80, "min_area": 25}

# The response surface is FLAT near the peak (mAP 0.6066-0.6078 across seed
# 0.70-0.80, fg 0.75-0.82, min_area 20-35), so these were tuned on the same 135
# val images we report. The broad trend is real; the final ~0.002 is noise-fit.


def _filter_small(labels: np.ndarray, min_area: int) -> Tuple[np.ndarray, int]:
    """Drop components under ``min_area`` px and relabel 1..N contiguously.

    Contiguous ids matter: the metric's joint-histogram indexing allocates an
    (n_gt+1) x (n_pred+1) table from the MAX id, so sparse ids with a huge max
    would allocate a needlessly large table.
    """
    if labels.max() == 0:
        return labels, 0
    # bincount over the flat label map: one pass gives every component's area.
    areas = np.bincount(labels.ravel())
    too_small = np.flatnonzero(areas < min_area)
    # Index 0 is background; never treat it as a small component.
    too_small = too_small[too_small > 0]

    if too_small.size:
        # A lookup table remaps every id in ONE vectorized indexing operation,
        # instead of `labels[labels == i] = 0` per unwanted component.
        lut = np.arange(labels.max() + 1, dtype=np.int32)
        lut[too_small] = 0
        labels = lut[labels]

    # Re-pack ids: unique(return_inverse) maps surviving ids onto 0..N-1, then
    # reshape restores the image. Background stays 0 because it sorts first.
    uniq, inv = np.unique(labels, return_inverse=True)
    labels = inv.reshape(labels.shape).astype(np.int32)
    return labels, int(labels.max())


def instances_from_target(
    target: np.ndarray,
    min_area: int = MIN_AREA,
) -> np.ndarray:
    """Instance labels from a HARD 3-class map (ground truth, or an argmax).

    Used for two things: the representation-ceiling experiment (how many
    instances survive a perfect prediction) and inference when only the argmax
    is kept.

    Elevation here is the border class itself rather than a probability, since a
    hard map carries no confidence: flooding uphill out of each interior seed
    stops on the border ring.
    """
    interior = target == INTERIOR
    fg = target > BG

    # ndi.label assigns a distinct id to each 4/8-connected component. The
    # default structure is 4-connectivity (cross), which is the conservative
    # choice: 8-connectivity would merge two nuclei touching only at a corner.
    seeds, n_seeds = ndi.label(interior)
    if n_seeds == 0:
        return np.zeros_like(target, dtype=np.int32)

    # Distance transform of the foreground gives a smooth elevation whose ridges
    # sit midway between seeds, which is what makes the cut land on the wall
    # even where the ring is 1 px thin.
    dist = ndi.distance_transform_edt(fg)
    labels = watershed(-dist, markers=seeds, mask=fg)
    labels, _ = _filter_small(labels.astype(np.int32), min_area)
    return labels


def instances_from_probs(
    probs: np.ndarray,
    seed_threshold: float = 0.5,
    fg_threshold: float = 0.5,
    min_area: int = MIN_AREA,
    use_distance: bool = False,
) -> np.ndarray:
    """Instance labels from SOFT 3-class probabilities. The inference path.

    Parameters
    ----------
    probs : np.ndarray
        ``(3, H, W)`` or ``(H, W, 3)``, channel order {bg, interior, border}.
        Must be post-softmax.
    seed_threshold : float
        Interior probability required to seed. The key tuning dial:
        RAISE it to split under-separated clumps (fewer, more confident seeds),
        LOWER it to stop dropping faint nuclei. Tune on val mAP, not by eye.
    fg_threshold : float
        ``1 - P(background)`` required to belong to any instance.
    use_distance : bool
        True -> flood on the distance transform (geometric, ignores confidence).
        False -> flood on ``-P(interior)``, which respects the network's own
        uncertainty and generally cuts better where nuclei differ in size.

    Returns
    -------
    np.ndarray
        int32 label map, ids 1..N contiguous.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 3:
        raise ValueError(f"probs must be 3-D, got {probs.shape}")
    # Accept both layouts: CHW if the first axis is the size-3 one.
    if probs.shape[0] == 3 and probs.shape[-1] != 3:
        p_bg, p_int, p_bor = probs[0], probs[1], probs[2]
    elif probs.shape[-1] == 3:
        p_bg, p_int, p_bor = probs[..., 0], probs[..., 1], probs[..., 2]
    else:
        raise ValueError(f"cannot locate the 3-class axis in {probs.shape}")

    seeds_mask = p_int > seed_threshold
    fg = (1.0 - p_bg) > fg_threshold

    seeds, n_seeds = ndi.label(seeds_mask)
    if n_seeds == 0:
        return np.zeros(p_bg.shape, dtype=np.int32)

    if use_distance:
        elevation = -ndi.distance_transform_edt(fg)
    else:
        # Negated so that high interior confidence is a BASIN (watershed floods
        # from low to high) and the low-confidence border ridge is the divide.
        elevation = -p_int

    labels = watershed(elevation, markers=seeds, mask=fg)
    labels, _ = _filter_small(labels.astype(np.int32), min_area)
    return labels


def instances_from_logits(
    logits: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Softmax over the 3-class axis, then :func:`instances_from_probs`.

    Softmax is computed with the max-subtraction trick: exp of a raw logit can
    overflow float32 (exp(89) is already inf), and subtracting the per-pixel max
    is algebraically identity-preserving.
    """
    logits = np.asarray(logits, dtype=np.float32)
    axis = 0 if (logits.shape[0] == 3 and logits.shape[-1] != 3) else -1
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return instances_from_probs(e / e.sum(axis=axis, keepdims=True), **kwargs)


def summarize_instances(labels: np.ndarray) -> Dict[str, Any]:
    """Scalar summary of a label map. No arrays returned (lean-ctx rule)."""
    n = int(labels.max())
    if n == 0:
        return {"n_instances": 0, "area_min": 0, "area_median": 0.0, "area_max": 0}
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    return {
        "n_instances": n,
        "area_min": int(areas.min()),
        "area_median": float(np.median(areas)),
        "area_max": int(areas.max()),
    }
