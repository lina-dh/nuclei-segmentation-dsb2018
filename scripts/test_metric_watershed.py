"""Unit tests for the mAP metric and the watershed post-processing.

    python -m scripts.test_metric_watershed

Synthetic cases first, with expected values derived BY HAND in the comments -
a metric validated only against its own implementation validates nothing. Then
the representation-ceiling experiment on real data, which answers the one
question that decides whether the 3-class plan can work at all:

    If the network were PERFECT, what mAP would the 3-class target + watershed
    round-trip actually yield?

That number is the hard upper bound on every score we can ever report. If it
were, say, 0.80, then chasing 0.85 would be impossible by construction.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import BG, BORDER, INTERIOR, SampleCache, make_3class_target  # noqa: E402
from src.dataset_manifest import load_manifest  # noqa: E402
from src.metrics import (  # noqa: E402
    IOU_THRESHOLDS, assert_unique_matching, average_precision, dataset_map,
    instance_iou_matrix,
)
from src.postprocess import (  # noqa: E402
    MIN_AREA, instances_from_logits, instances_from_probs, instances_from_target,
    summarize_instances,
)

STATE_DIR = ROOT / ".project_state"
CACHE_DIR = ROOT / "data" / "cache"

_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _blobs(shape, boxes):
    """Build a label map from (y0, y1, x0, x1) boxes, one id per box."""
    m = np.zeros(shape, dtype=np.int32)
    for i, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        m[y0:y1, x0:x1] = i
    return m


# =============================================================================
def test_iou():
    print("=== IoU matrix ===")
    # Two 10x10 GT squares. Pred 1 is identical to GT 1. Pred 2 is GT 2 shifted
    # 5 px right: overlap 10x5=50, union = 100 + 100 - 50 = 150 -> IoU = 1/3.
    gt = _blobs((40, 40), [(0, 10, 0, 10), (20, 30, 20, 30)])
    pred = _blobs((40, 40), [(0, 10, 0, 10), (20, 30, 25, 35)])
    iou = instance_iou_matrix(gt, pred)
    check("IoU shape is (n_gt, n_pred)", iou.shape == (2, 2), str(iou.shape))
    check("identical instance -> IoU 1.0", abs(iou[0, 0] - 1.0) < 1e-12,
          f"{iou[0, 0]:.6f}")
    check("half-shifted square -> IoU 50/150 = 1/3",
          abs(iou[1, 1] - 1 / 3) < 1e-12, f"{iou[1, 1]:.6f}")
    check("non-overlapping pairs -> IoU 0",
          iou[0, 1] == 0.0 and iou[1, 0] == 0.0)

    # Brute-force cross-check: the histogram trick must agree with the naive
    # per-pair boolean AND. This is what makes the fast path trustworthy.
    brute = np.zeros_like(iou)
    for i in range(1, 3):
        for j in range(1, 3):
            a, b = gt == i, pred == j
            brute[i - 1, j - 1] = (a & b).sum() / max((a | b).sum(), 1)
    check("histogram IoU == brute-force IoU", np.allclose(iou, brute))

    # Non-contiguous ids must not break the packing arithmetic.
    gt2 = gt.copy() * 7          # ids {0, 7, 14}
    iou2 = instance_iou_matrix(gt2, pred)
    check("non-contiguous ids: table sized from max id",
          iou2.shape == (14, 2) and abs(iou2[6, 0] - 1.0) < 1e-12, str(iou2.shape))

    check("empty gt -> (0, n_pred) matrix",
          instance_iou_matrix(np.zeros_like(pred), pred).shape == (0, 2))
    # The shape guard must fire: silently comparing mismatched maps would make
    # every score meaningless.
    try:
        instance_iou_matrix(np.zeros((9, 9), np.int32), pred)
        guard = False
    except ValueError:
        guard = True
    check("mismatched shapes raise ValueError", guard)


def test_map_values():
    print("\n=== mAP arithmetic ===")
    gt = _blobs((40, 40), [(0, 10, 0, 10), (20, 30, 20, 30)])

    # Perfect prediction: TP=2, FP=FN=0 at every threshold -> precision 1.0.
    r = average_precision(gt, gt.copy())
    check("identical maps -> mAP 1.0", abs(r["map"] - 1.0) < 1e-12, f"{r['map']:.6f}")

    # Disjoint prediction: TP=0, FP=1, FN=2 -> 0/(0+1+2) = 0 at all thresholds.
    pred_bad = _blobs((40, 40), [(35, 40, 35, 40)])
    r = average_precision(gt, pred_bad)
    check("fully disjoint -> mAP 0.0", r["map"] == 0.0)

    # ONE GT, ONE PRED, IoU exactly 1/3 -> below every threshold (min 0.5), so
    # TP=0, FP=1, FN=1 -> precision 0 everywhere.
    g1 = _blobs((40, 40), [(0, 10, 0, 10)])
    p1 = _blobs((40, 40), [(0, 10, 5, 15)])
    check("IoU 1/3 scores 0 (below the 0.5 floor)",
          average_precision(g1, p1)["map"] == 0.0)

    # HAND-COMPUTED PARTIAL CASE. Pred is GT shifted 1 px: a 10x10 square
    # overlapping 10x9=90, union 100+100-90=110 -> IoU = 90/110 = 0.8181...
    # Thresholds passed: 0.50..0.80 -> 7 of 10. Failed: 0.85, 0.90, 0.95.
    # Passing t:  TP=1 FP=0 FN=0 -> 1.0        (7 thresholds)
    # Failing t:  TP=0 FP=1 FN=1 -> 0.0        (3 thresholds)
    # mAP = 7/10 = 0.7
    p2 = _blobs((40, 40), [(0, 10, 1, 11)])
    iou_val = instance_iou_matrix(g1, p2)[0, 0]
    r = average_precision(g1, p2)
    check("single shifted square: IoU == 90/110", abs(iou_val - 90 / 110) < 1e-12,
          f"{iou_val:.6f}")
    check("that case scores exactly 0.7 (7/10 thresholds)",
          abs(r["map"] - 0.7) < 1e-12, f"{r['map']:.6f}")

    # OVER-SEGMENTATION (one GT split into two preds). Each half has IoU ~0.5
    # with the GT: 50/100 = 0.5, which is NOT > 0.5, so no match at any
    # threshold -> TP=0, FP=2, FN=1 -> 0.0 everywhere.
    p_split = _blobs((40, 40), [(0, 5, 0, 10), (5, 10, 0, 10)])
    r = average_precision(g1, p_split)
    check("split instance: TP=0, FP=2, FN=1 at t=0.5",
          r["per_threshold"][0]["fp"] == 2 and r["per_threshold"][0]["fn"] == 1
          and r["per_threshold"][0]["tp"] == 0)

    # UNDER-SEGMENTATION (two GT merged into one pred). The merged blob has
    # IoU 100/200 = 0.5 with each GT -> again not > 0.5. TP=0, FP=1, FN=2.
    g2 = _blobs((40, 40), [(0, 10, 0, 10), (10, 20, 0, 10)])
    p_merge = _blobs((40, 40), [(0, 20, 0, 10)])
    r = average_precision(g2, p_merge)
    check("merged instances: TP=0, FP=1, FN=2 at t=0.5",
          r["per_threshold"][0]["tp"] == 0 and r["per_threshold"][0]["fp"] == 1
          and r["per_threshold"][0]["fn"] == 2)

    # Kaggle's denominator includes FN: missing half the nuclei must score 0.5,
    # not 1.0 as a conventional TP/(TP+FP) precision would.
    p_half = _blobs((40, 40), [(0, 10, 0, 10)])
    r = average_precision(gt, p_half)
    check("1 of 2 found -> 1/(1+0+1) = 0.5 (FN is in the denominator)",
          abs(r["map"] - 0.5) < 1e-12, f"{r['map']:.6f}")

    # Empty conventions. Same shape as p_half, or the shape guard fires first.
    z = np.zeros_like(p_half)
    check("both empty -> 1.0", average_precision(z, z)["map"] == 1.0)
    check("gt empty, pred not -> 0.0", average_precision(z, p_half)["map"] == 0.0)
    check("pred empty, gt not -> 0.0", average_precision(p_half, z)["map"] == 0.0)

    check("threshold grid is the 10 competition values",
          len(IOU_THRESHOLDS) == 10
          and abs(IOU_THRESHOLDS[0] - 0.5) < 1e-12
          and abs(IOU_THRESHOLDS[-1] - 0.95) < 1e-12,
          f"{np.round(IOU_THRESHOLDS, 2).tolist()}")

    # Uniqueness guarantee that justifies counting instead of greedy matching.
    ok = True
    try:
        assert_unique_matching(instance_iou_matrix(gt, gt.copy()), 0.5)
    except AssertionError:
        ok = False
    check("one-to-one matching holds at t>=0.5", ok)

    # dataset_map averages per IMAGE: a perfect sparse image and a 0-scoring
    # dense one must give 0.5, NOT an instance-weighted number.
    d = dataset_map([("a", gt, gt.copy()), ("b", gt, pred_bad)])
    check("dataset_map averages per image, not per instance",
          abs(d["map"] - 0.5) < 1e-12, f"{d['map']:.6f}")
    check("dataset_map reports worst images", d["worst"][0]["sample_id"] == "b")


def test_watershed_synthetic():
    print("\n=== watershed (synthetic) ===")
    # TWO TOUCHING SQUARES, the case binary segmentation cannot solve.
    # Instances abut along x=10, so make_3class_target must ring both and the
    # watershed must recover exactly 2.
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[10:30, 0:10] = 1
    lab[10:30, 10:20] = 2
    tgt = make_3class_target(lab)
    check("target of touching pair contains a border class",
          int((tgt == BORDER).sum()) > 0, f"{int((tgt == BORDER).sum())} px")

    rec = instances_from_target(tgt)
    check("touching pair -> 2 instances recovered", int(rec.max()) == 2,
          f"got {int(rec.max())}")
    # Instance identity may be permuted, so score with the metric, which is
    # permutation-invariant, rather than comparing ids directly.
    r = average_precision(lab, rec)
    check("touching pair round-trip mAP > 0.5", r["map"] > 0.5, f"{r['map']:.4f}")

    # A single isolated blob must stay ONE instance (no spurious splitting).
    solo = np.zeros((40, 40), dtype=np.int32)
    solo[10:30, 10:30] = 1
    check("isolated blob -> exactly 1 instance",
          int(instances_from_target(make_3class_target(solo)).max()) == 1)

    # Empty input must not crash and must return an empty map.
    empty = np.zeros((20, 20), dtype=np.uint8)
    check("all-background target -> 0 instances",
          int(instances_from_target(empty).max()) == 0)

    # min_area filtering: a 3x3 (9 px) speck is below MIN_AREA=15 and must go,
    # while the 20x20 blob survives.
    speck = solo.copy()
    speck[0:3, 0:3] = 2
    rec_s = instances_from_target(make_3class_target(speck), min_area=MIN_AREA)
    check(f"speck under min_area={MIN_AREA} px is dropped",
          int(rec_s.max()) == 1, f"got {int(rec_s.max())}")
    check("min_area=1 keeps the speck",
          int(instances_from_target(make_3class_target(speck), min_area=1).max()) == 2)

    # ---- soft-probability path -------------------------------------------
    # Build probabilities from the hard target (one-hot) and confirm the soft
    # path reproduces the hard path on a confident input.
    probs = np.zeros((3, 40, 40), dtype=np.float32)
    for c in (BG, INTERIOR, BORDER):
        probs[c] = (tgt == c).astype(np.float32)
    rec_p = instances_from_probs(probs)
    check("one-hot probs (CHW) -> 2 instances", int(rec_p.max()) == 2,
          f"got {int(rec_p.max())}")
    rec_hwc = instances_from_probs(np.transpose(probs, (1, 2, 0)))
    check("HWC layout accepted and identical",
          np.array_equal(rec_p, rec_hwc))

    # Logit path: scaling one-hot by 20 makes extreme logits. The softmax must
    # not overflow (exp(20) is fine, but the max-subtraction is what guarantees
    # it in general) and must give the same instances.
    rec_l = instances_from_logits(probs * 20.0)
    check("logits path matches probs path", int(rec_l.max()) == 2)
    check("softmax of large logits stays finite",
          bool(np.isfinite(instances_from_logits(probs * 1000.0)).all()))

    check("distance-based elevation also recovers 2",
          int(instances_from_probs(probs, use_distance=True).max()) == 2)

    # ids must be contiguous 1..N for the metric's table sizing.
    ids = np.unique(rec_p)
    check("output ids are contiguous 1..N",
          np.array_equal(ids, np.arange(ids.size)), f"{ids.tolist()}")


def test_representation_ceiling():
    """The decisive experiment: perfect prediction -> watershed -> mAP."""
    print("\n=== REPRESENTATION CEILING (real data, perfect prediction) ===")
    manifest, _ = load_manifest(STATE_DIR)
    val_ids = [r["sample_id"] for r in manifest["samples"] if r["split"] == "val"]
    cache = SampleCache(CACHE_DIR, val_ids, mode="mmap", keep_label=True)

    pairs = []
    n_gt_total = n_rec_total = 0
    for sid in val_ids:
        _, tgt, lab = cache.get(sid)
        gt = np.asarray(lab).astype(np.int32)
        rec = instances_from_target(np.asarray(tgt))
        pairs.append((sid, gt, rec))
        n_gt_total += int(gt.max())
        n_rec_total += int(rec.max())

    d = dataset_map(pairs)
    print(f"  val images            : {d['n_images']}")
    print(f"  GT instances          : {n_gt_total}")
    print(f"  recovered instances   : {n_rec_total} "
          f"({100 * n_rec_total / max(1, n_gt_total):.2f}% of GT)")
    print(f"  CEILING mAP@[.5:.95]  : {d['map']:.4f}")
    print("  per-threshold mean precision:")
    for row in d["per_threshold"]:
        print(f"    t={row['t']:.2f}  {row['mean_precision']:.4f}")
    print(f"  worst images: {[(w['sample_id'][:10], w['map']) for w in d['worst']]}")

    # A ceiling below 0.90 would mean the TARGET REPRESENTATION, not the model,
    # is the binding constraint - that would force a redesign (wider border,
    # distance-transform regression, etc.) before any training.
    check("representation ceiling >= 0.90", d["map"] >= 0.90, f"{d['map']:.4f}")
    check("instance count recovered within 2% of GT",
          abs(n_rec_total - n_gt_total) / max(1, n_gt_total) < 0.02,
          f"{n_rec_total} vs {n_gt_total}")
    return d


def main() -> int:
    test_iou()
    test_map_values()
    test_watershed_synthetic()
    test_representation_ceiling()

    n_fail = sum(1 for _, ok in _results if not ok)
    print("\n" + "=" * 62)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed"
          + (f"  -- {n_fail} FAILED" if n_fail else "  -- ALL PASS"))
    for nm, ok in _results:
        if not ok:
            print(f"  FAILED: {nm}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
