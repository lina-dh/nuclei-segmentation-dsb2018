"""Instance-segmentation metrics: the 2018 Data Science Bowl mAP@IoU.

THE COMPETITION METRIC, stated precisely, because everything downstream is
scored by it:

    For each image, for each IoU threshold t in {0.50, 0.55, ..., 0.95}:
        TP(t) = predicted instances matched to a GT instance at IoU > t
        FP(t) = predicted instances with no match
        FN(t) = GT instances with no match
        precision(t) = TP / (TP + FP + FN)          <- NOT the usual TP/(TP+FP)
    image score = mean over the 10 thresholds
    final score = mean over images                  <- per IMAGE, not per instance

Two traps in that definition:

1. The denominator includes FN. This is "precision" only in Kaggle's naming; it
   is really a Jaccard-style detection score. Missing a nucleus is penalized as
   hard as inventing one.
2. The final average is over IMAGES, so a 4-nucleus image carries the same
   weight as a 375-nucleus image. Fixing one sparse image can move the score
   more than fixing one dense image.

Matching needs no greedy loop: at IoU > 0.5 a one-to-one matching is forced.
Proof: if GT instance a matched both b and c, then |a&b| > |a|/2 and
|a&c| > |a|/2, but b and c are disjoint so |a&b| + |a&c| <= |a|. Contradiction.
So simply counting pairs above the threshold is exact for every t >= 0.5.
:func:`assert_unique_matching` asserts this rather than trusting it.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

# The competition grid: 0.50 to 0.95 inclusive, step 0.05.
# arange with a 0.951 stop, not 0.95: floating-point arange would otherwise drop
# the final 0.95 depending on accumulated error.
IOU_THRESHOLDS = np.arange(0.5, 0.951, 0.05)


def instance_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU between every GT instance and every predicted instance.

    Parameters
    ----------
    gt, pred : np.ndarray
        Integer label maps of identical shape. 0 = background; instance ids are
        any positive integers (they need not be contiguous).

    Returns
    -------
    np.ndarray
        Shape ``(n_gt, n_pred)``, float64, indexed by ``id - 1``.

    Notes
    -----
    Built from a single 2-D JOINT HISTOGRAM instead of an n_gt x n_pred loop of
    boolean ANDs. The pairing ``gt * (n_pred + 1) + pred`` encodes each pixel's
    (gt_id, pred_id) pair as one integer, so np.bincount tallies every pair in
    ONE O(pixels) pass; reshaping recovers the 2-D contingency table. For the
    worst sample (375 instances) the loop form would be ~140k boolean ANDs over
    1 Mpx each; this is one pass.

    Marginal sums of that table give the areas for free: row sums are GT areas,
    column sums are prediction areas, so nothing is recomputed.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    # int64 throughout: the packed index can reach n_gt*n_pred which overflows
    # int32 for large instance counts.
    gt_f = gt.reshape(-1).astype(np.int64, copy=False)
    pr_f = pred.reshape(-1).astype(np.int64, copy=False)
    n_gt = int(gt_f.max()) if gt_f.size else 0
    n_pred = int(pr_f.max()) if pr_f.size else 0

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)

    packed = gt_f * (n_pred + 1) + pr_f
    hist = np.bincount(packed, minlength=(n_gt + 1) * (n_pred + 1))
    hist = hist.reshape(n_gt + 1, n_pred + 1)

    # Drop row 0 / column 0: background is not an instance.
    inter = hist[1:, 1:].astype(np.float64)
    area_gt = hist[1:, :].sum(axis=1).astype(np.float64)      # includes bg overlap
    area_pred = hist[:, 1:].sum(axis=0).astype(np.float64)

    # Broadcasting (n_gt,1) + (1,n_pred) -> (n_gt,n_pred) without materializing
    # either operand as a full matrix.
    union = area_gt[:, None] + area_pred[None, :] - inter
    # union is 0 only when both instances are empty, which cannot happen for an
    # id present in the map; the guard keeps it division-safe regardless.
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def assert_unique_matching(iou: np.ndarray, threshold: float) -> None:
    """Verify the one-to-one property that lets us count instead of match.

    Raises AssertionError if any GT or prediction has 2+ partners above the
    threshold. Mathematically impossible for threshold >= 0.5, so a failure
    means the label maps overlap themselves (ids sharing pixels), not that the
    metric is wrong.
    """
    if iou.size == 0:
        return
    hits = iou > threshold
    assert hits.sum(axis=1).max(initial=0) <= 1, "a GT instance matched 2+ predictions"
    assert hits.sum(axis=0).max(initial=0) <= 1, "a prediction matched 2+ GT instances"


def average_precision(
    gt: np.ndarray,
    pred: np.ndarray,
    thresholds: Sequence[float] = IOU_THRESHOLDS,
    iou: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Score ONE image with the competition metric.

    Parameters
    ----------
    gt, pred : np.ndarray
        Integer instance label maps, same shape.
    iou : np.ndarray, optional
        Pre-computed IoU matrix, to avoid recomputing it across calls.

    Returns
    -------
    dict
        ``map`` (mean over thresholds), ``per_threshold`` (list of dicts with
        t/tp/fp/fn/precision), ``n_gt``, ``n_pred``.

    Notes
    -----
    Empty-image convention: both empty -> 1.0 (nothing to find, nothing
    invented). Exactly one empty -> 0.0. Kaggle's grader never ships a GT-empty
    image, but inference on blank tiles produces this case constantly, so it is
    defined explicitly rather than left to divide-by-zero.
    """
    if iou is None:
        iou = instance_iou_matrix(gt, pred)
    n_gt, n_pred = iou.shape

    per_threshold = []
    for t in thresholds:
        if n_gt == 0 and n_pred == 0:
            tp = fp = fn = 0
            precision = 1.0
        else:
            # Strict > t, per the competition statement.
            tp = int((iou > t).sum())
            fp = n_pred - tp
            fn = n_gt - tp
            denom = tp + fp + fn
            precision = (tp / denom) if denom else 1.0
        per_threshold.append({
            "t": round(float(t), 2), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
        })

    return {
        "map": float(np.mean([d["precision"] for d in per_threshold])),
        "per_threshold": per_threshold,
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
    }


def dataset_map(
    pairs: Sequence[Any],
    thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> Dict[str, Any]:
    """Score a dataset: mean of per-IMAGE mAP, which is the competition's order.

    Parameters
    ----------
    pairs : sequence of (gt, pred) label-map tuples, or of (sample_id, gt, pred).

    Returns
    -------
    dict
        ``map``, ``n_images``, ``per_threshold`` (threshold-wise mean precision
        across images, which is what tells you whether the model is losing
        points on detection (low-t) or on boundary quality (high-t)),
        ``worst`` (the 5 lowest-scoring images, to inspect).

    Notes
    -----
    Averaging order matters and is NOT interchangeable: mean over images of
    (mean over thresholds) is the competition metric. Pooling all instances
    first would weight dense images far more heavily and report a different
    number.
    """
    scores = []
    ids = []
    acc = np.zeros(len(thresholds), dtype=np.float64)

    for item in pairs:
        if len(item) == 3:
            sid, g, p = item
        else:
            sid, (g, p) = None, item
        r = average_precision(g, p, thresholds=thresholds)
        scores.append(r["map"])
        ids.append(sid)
        acc += np.array([d["precision"] for d in r["per_threshold"]])

    n = len(scores)
    if n == 0:
        return {"map": 0.0, "n_images": 0, "per_threshold": [], "worst": []}

    order = np.argsort(scores)[:5]
    return {
        "map": float(np.mean(scores)),
        "n_images": n,
        "per_threshold": [
            {"t": round(float(t), 2), "mean_precision": float(acc[i] / n)}
            for i, t in enumerate(thresholds)
        ],
        "worst": [
            {"sample_id": ids[i], "map": round(float(scores[i]), 4)} for i in order
        ],
    }
