"""Tests for the L6 sliding-window stitcher. Requires torch (run on the studio).

A stitching bug does not crash - it quietly depresses every val score, and it
looks exactly like a model that cannot learn boundaries. These tests pin the
properties that make the stitched output trustworthy.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import _hann_weight, _tile_positions, sliding_window_probs, tile_count  # noqa: E402

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


class ConstantModel(nn.Module):
    """Emits fixed logits regardless of input: makes the stitched result predictable."""

    def __init__(self, logits=(0.0, 2.0, 1.0)):
        super().__init__()
        self.register_buffer("v", torch.tensor(logits).view(1, 3, 1, 1))
        # A real parameter so .to(device)/.eval() behave like a true module.
        self.dummy = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.v.expand(x.shape[0], 3, x.shape[2], x.shape[3]).clone()


class XGradientModel(nn.Module):
    """Interior logit increases with the pixel's own red value.

    Lets us verify that the stitched output tracks a SPATIALLY VARYING function
    of the input, i.e. that tiles land at the right offsets. A constant model
    cannot catch a transposed or shifted paste.
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        # x is normalized; channel 0 carries the spatial pattern we planted.
        interior = x[:, 0:1] * 5.0
        return torch.cat([torch.zeros_like(interior), interior,
                          torch.zeros_like(interior)], dim=1)


def test_tiling():
    print("=== tiling geometry ===")
    check("exact fit -> single tile", _tile_positions(256, 256, 192) == [0])
    check("smaller than crop -> single tile", _tile_positions(100, 256, 192) == [0])
    # 1024 with crop 256 stride 192: 0,192,384,576,768 then clamp to 768 (=1024-256)
    pos = _tile_positions(1024, 256, 192)
    check("last tile is clamped to size-crop (full coverage, no overrun)",
          pos[-1] == 1024 - 256 and all(p + 256 <= 1024 for p in pos), f"{pos}")
    # Every pixel must be covered by at least one tile.
    cov = np.zeros(1024, dtype=int)
    for p in pos:
        cov[p:p + 256] += 1
    check("every row covered at least once", int(cov.min()) >= 1,
          f"min={int(cov.min())} max={int(cov.max())}")
    check("tile_count matches the val estimate for 1024x1024",
          tile_count(1024, 1024) == len(pos) ** 2, f"{tile_count(1024, 1024)}")

    w = _hann_weight(256)
    check("hann weight is strictly positive (no NaN at image borders)",
          float(w.min()) > 0, f"min={float(w.min()):.4f}")
    check("hann weight peaks at the tile centre",
          float(w[128, 128]) > float(w[0, 0]) * 5,
          f"centre={float(w[128, 128]):.3f} corner={float(w[0, 0]):.3f}")


def test_stitch(device):
    print(f"\n=== stitching (device={device}) ===")
    model = ConstantModel().to(device)

    # Softmax of (0, 2, 1) = exp/sum -> [0.0900, 0.6652, 0.2447].
    expect = torch.softmax(torch.tensor([0.0, 2.0, 1.0]), 0).numpy()

    for (H, W) in [(256, 256), (260, 347), (512, 640), (1024, 1024)]:
        img = np.random.randint(0, 255, (H, W, 3), np.uint8)
        probs = sliding_window_probs(model, img, device=device, amp=False)
        ok_shape = probs.shape == (3, H, W)
        # A constant model must stitch to that exact constant EVERYWHERE,
        # including seams: this is what proves the Hann weights normalize out.
        ok_val = np.allclose(probs, expect[:, None, None], atol=1e-5)
        ok_sum = np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)
        check(f"{H}x{W}: shape, constant value through seams, sums to 1",
              ok_shape and ok_val and ok_sum,
              f"shape={probs.shape} maxdev={float(np.abs(probs - expect[:, None, None]).max()):.2e}")

    # Sub-tile image: exercises the replicate-pad branch.
    small = np.random.randint(0, 255, (100, 180, 3), np.uint8)
    p = sliding_window_probs(model, small, device=device, amp=False)
    check("sub-256 image returns ORIGINAL size (padding undone)",
          p.shape == (3, 100, 180), str(p.shape))
    check("sub-256 image values still correct",
          np.allclose(p, expect[:, None, None], atol=1e-5))

    # ---- spatial fidelity -------------------------------------------------
    # Plant a horizontal ramp in the red channel; a correct stitch must produce
    # an interior probability that increases monotonically left-to-right.
    H = W = 512
    ramp = np.zeros((H, W, 3), np.uint8)
    ramp[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    gm = XGradientModel().to(device)
    pr = sliding_window_probs(gm, ramp, device=device, amp=False, norm=None)
    row = pr[1, H // 2]
    check("stitched output tracks a spatially varying input (monotonic ramp)",
          bool(np.all(np.diff(row) >= -1e-6)) and row[-1] > row[0] + 0.3,
          f"left={row[0]:.4f} right={row[-1]:.4f}")

    # Vertical ramp: catches an axis transposition that the horizontal test
    # alone would miss.
    ramp_v = np.zeros((H, W, 3), np.uint8)
    ramp_v[:, :, 0] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    pv = sliding_window_probs(gm, ramp_v, device=device, amp=False, norm=None)
    col = pv[1, :, W // 2]
    check("no axis transposition (vertical ramp is monotonic down)",
          bool(np.all(np.diff(col) >= -1e-6)) and col[-1] > col[0] + 0.3,
          f"top={col[0]:.4f} bottom={col[-1]:.4f}")

    # Uniform blending must agree with Hann for a constant model (both are
    # weighted means of identical values) - a sanity check on the blend switch.
    pu = sliding_window_probs(model, np.zeros((512, 640, 3), np.uint8),
                              device=device, amp=False, blend="uniform")
    check("uniform blending path also correct",
          np.allclose(pu, expect[:, None, None], atol=1e-5))

    if device == "cuda":
        # AMP must not change the stitched result materially: fp16 forward but
        # float32 softmax and float32 accumulators.
        pa = sliding_window_probs(model, np.zeros((512, 640, 3), np.uint8),
                                  device=device, amp=True)
        check("AMP stitch matches fp32 stitch",
              np.allclose(pa, expect[:, None, None], atol=1e-3),
              f"maxdev={float(np.abs(pa - expect[:, None, None]).max()):.2e}")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {device}\n")
    test_tiling()
    test_stitch(device)
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
