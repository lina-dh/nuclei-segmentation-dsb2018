"""Loss + AMP tests. REQUIRES TORCH (AMP checks need a CUDA GPU):

    python -m scripts.test_losses

Covers: weight schemes, CE+dice correctness on hand-constructed cases, the
ignore mask actually excluding pixels, behaviour when a class is absent,
gradient flow, and an end-to-end AMP (fp16 autocast + GradScaler) step - which
is where the fp16 underflow trap in soft_dice_loss would show up.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import (  # noqa: E402
    DEFAULT_CLASS_FREQS, CEDiceLoss, class_weights, make_loss, soft_dice_loss,
)

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def onehot_logits(target, n_cls=3, confidence=20.0):
    """Logits that predict `target` with near-certainty: a perfect prediction."""
    oh = F.one_hot(target.long(), n_cls).permute(0, 3, 1, 2).float()
    return oh * confidence


def test_weights():
    print("=== class weights ===")
    for scheme in ("none", "inv", "inv_sqrt"):
        w = class_weights(DEFAULT_CLASS_FREQS, scheme=scheme)
        check(f"{scheme}: 3 finite weights, mean 1.0",
              w.shape == (3,) and bool(torch.isfinite(w).all())
              and abs(float(w.mean()) - 1.0) < 1e-5,
              f"{np.round(w.numpy(), 3).tolist()}")
        # NON-DEGENERACY: every weighting scheme except "none" must actually
        # differentiate the classes. This is the check that exposed the
        # "effective number" scheme silently collapsing to (1,1,1).
        if scheme != "none":
            check(f"{scheme}: weights are not degenerate (all-equal)",
                  float(w.max() - w.min()) > 0.1,
                  f"spread={float(w.max() - w.min()):.3f}")

    try:
        class_weights(DEFAULT_CLASS_FREQS, scheme="effective")
        removed = False
    except ValueError:
        removed = True
    check("degenerate 'effective' scheme is rejected, not silently uniform",
          removed)

    w_inv = class_weights(DEFAULT_CLASS_FREQS, "inv")
    w_sqrt = class_weights(DEFAULT_CLASS_FREQS, "inv_sqrt")
    # Rarer class must get more weight: border > interior > background.
    check("inv_sqrt orders border > interior > bg",
          float(w_sqrt[2]) > float(w_sqrt[1]) > float(w_sqrt[0]))
    # And inv must be strictly more aggressive than inv_sqrt - the whole reason
    # inv_sqrt is the default.
    check("inv is more extreme than inv_sqrt (border/bg ratio)",
          float(w_inv[2] / w_inv[0]) > float(w_sqrt[2] / w_sqrt[0]),
          f"inv={float(w_inv[2] / w_inv[0]):.1f}x vs "
          f"inv_sqrt={float(w_sqrt[2] / w_sqrt[0]):.1f}x")
    # An absent class must not produce inf weights (the 1e-8 floor).
    check("absent class does not yield inf weight",
          bool(torch.isfinite(class_weights((0.9, 0.1, 0.0), "inv")).all()))


def test_loss_values(device):
    print(f"\n=== CE + dice values (device={device}) ===")
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    # A target containing all three classes: a square with a ring border.
    target = torch.zeros(B, H, W, dtype=torch.long)
    target[:, 16:48, 16:48] = 1
    target[:, 16:18, 16:48] = 2
    target[:, 46:48, 16:48] = 2
    target = target.to(device)

    loss_fn = make_loss(DEFAULT_CLASS_FREQS, scheme="inv_sqrt").to(device)

    # PERFECT prediction -> both terms near zero.
    perfect = onehot_logits(target).to(device)
    total, parts = loss_fn(perfect, target)
    check("perfect prediction: CE ~ 0", float(parts["ce"]) < 1e-5,
          f"ce={float(parts['ce']):.2e}")
    check("perfect prediction: dice ~ 0", float(parts["dice"]) < 1e-3,
          f"dice={float(parts['dice']):.2e}")

    # ALL-BACKGROUND prediction: the degenerate optimum of unweighted CE.
    allbg = torch.zeros(B, 3, H, W, device=device)
    allbg[:, 0] = 20.0
    total_bg, parts_bg = loss_fn(allbg, target)
    check("all-background prediction scores much worse than perfect",
          float(total_bg) > float(total) + 1.0,
          f"{float(total_bg):.3f} vs {float(total):.3f}")
    # Dice excludes bg, so predicting only bg must give dice ~ 1 (no overlap on
    # either foreground class).
    check("all-background: dice ~ 1.0 (fg classes have no overlap)",
          abs(float(parts_bg["dice"]) - 1.0) < 0.02, f"{float(parts_bg['dice']):.4f}")

    # WEIGHTING must matter: destroy ONLY the border class and confirm the
    # weighted loss rises more than the unweighted one.
    border_wrong = onehot_logits(target).to(device)
    is_border = (target == 2).unsqueeze(1)
    # Move border pixels' mass from class 2 to class 1 (a classic failure:
    # border absorbed into interior).
    border_wrong = torch.where(
        is_border,
        torch.stack([torch.zeros_like(target, dtype=torch.float)] * 1
                    + [torch.full_like(target, 20, dtype=torch.float)]
                    + [torch.zeros_like(target, dtype=torch.float)], dim=1),
        border_wrong,
    )
    unweighted = CEDiceLoss(weights=class_weights(DEFAULT_CLASS_FREQS, "none")).to(device)
    w_ce = float(loss_fn(border_wrong, target)[1]["ce"])
    u_ce = float(unweighted(border_wrong, target)[1]["ce"])
    check("losing the border costs MORE under weighting", w_ce > u_ce,
          f"weighted={w_ce:.4f} vs unweighted={u_ce:.4f}")

    # include_bg flips the dice magnitude, proving bg really is excluded.
    d_nobg = float(soft_dice_loss(allbg, target, include_bg=False))
    d_bg = float(soft_dice_loss(allbg, target, include_bg=True))
    check("include_bg changes dice (bg is excluded by default)",
          abs(d_nobg - d_bg) > 0.1, f"no_bg={d_nobg:.4f} with_bg={d_bg:.4f}")

    # ABSENT CLASS: a target with no border at all must not NaN.
    nob = torch.zeros(1, H, W, dtype=torch.long, device=device)
    nob[:, 10:30, 10:30] = 1
    t2, p2 = loss_fn(onehot_logits(nob).to(device), nob)
    check("absent class: loss stays finite (smooth term)",
          bool(torch.isfinite(t2)), f"total={float(t2):.2e}")


def test_ignore_mask(device):
    print("\n=== ignore mask ===")
    torch.manual_seed(0)
    B, H, W = 2, 32, 32
    target = torch.zeros(B, H, W, dtype=torch.long, device=device)
    target[:, 8:24, 8:24] = 1
    logits = torch.randn(B, 3, H, W, device=device)

    ignore = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    ignore[:, 24:, :] = True       # bottom strip is "padding"

    loss_fn = make_loss().to(device)
    base, _ = loss_fn(logits, target, ignore=ignore)

    # CORRUPT the ignored region's target. If ignoring works, the loss cannot
    # move at all - this is the strongest statement of the property.
    corrupted = target.clone()
    corrupted[:, 24:, :] = 2
    corr, _ = loss_fn(logits, corrupted, ignore=ignore)
    check("corrupting ignored-region TARGET does not change the loss",
          abs(float(base) - float(corr)) < 1e-6,
          f"{float(base):.8f} vs {float(corr):.8f}")

    # Corrupt the ignored region's LOGITS: likewise must not move the loss.
    logits2 = logits.clone()
    logits2[:, :, 24:, :] += 50.0
    corr2, _ = loss_fn(logits2, target, ignore=ignore)
    check("corrupting ignored-region LOGITS does not change the loss",
          abs(float(base) - float(corr2)) < 1e-6,
          f"{float(base):.8f} vs {float(corr2):.8f}")

    # Without the mask, the same corruption MUST change the loss, otherwise the
    # test above would be vacuous.
    nomask, _ = loss_fn(logits, target)
    nomask_corr, _ = loss_fn(logits, corrupted)
    check("without the mask, that corruption DOES change the loss",
          abs(float(nomask) - float(nomask_corr)) > 1e-4,
          f"{float(nomask):.6f} vs {float(nomask_corr):.6f}")

    # Fully-ignored input: CE over zero pixels. Must be 0, not NaN.
    allign = torch.ones(B, H, W, dtype=torch.bool, device=device)
    t, p = loss_fn(logits, target, ignore=allign)
    check("fully-ignored batch stays finite", bool(torch.isfinite(t)),
          f"total={float(t):.4f} ce={float(p['ce']):.4f} dice={float(p['dice']):.4f}")
    check("fully-ignored batch: CE is exactly 0", abs(float(p["ce"])) < 1e-9,
          f"ce={float(p['ce']):.2e}")
    # It must still be differentiable - a detached zero would break .backward().
    lg = logits.clone().requires_grad_(True)
    t2, _ = loss_fn(lg, target, ignore=allign)
    t2.backward()
    check("fully-ignored batch is still differentiable (finite grads)",
          lg.grad is not None and bool(torch.isfinite(lg.grad).all()))

    # SCALE REGRESSION: the manual weighted mean must reproduce PyTorch's own
    # reduction="mean" exactly, or the loss magnitude (and any tuned LR) shifts.
    w = class_weights(DEFAULT_CLASS_FREQS, "inv_sqrt").to(device)
    ce_target = target.masked_fill(ignore, -100)
    ref = F.cross_entropy(logits, ce_target, weight=w, ignore_index=-100)
    ours = loss_fn(logits, target, ignore=ignore)[1]["ce"]
    check("manual weighted mean == F.cross_entropy(reduction='mean')",
          abs(float(ref) - float(ours)) < 1e-6,
          f"ref={float(ref):.8f} ours={float(ours):.8f}")
    # Same check with no ignore mask at all.
    ref2 = F.cross_entropy(logits, target, weight=w)
    ours2 = loss_fn(logits, target)[1]["ce"]
    check("matches built-in mean with no ignore mask",
          abs(float(ref2) - float(ours2)) < 1e-6,
          f"ref={float(ref2):.8f} ours={float(ours2):.8f}")


def test_gradients_and_amp(device):
    print(f"\n=== gradients + AMP (device={device}) ===")
    torch.manual_seed(0)
    B, H, W = 4, 128, 128
    target = torch.zeros(B, H, W, dtype=torch.long, device=device)
    target[:, 32:96, 32:96] = 1
    target[:, 32:34, 32:96] = 2
    ignore = torch.zeros(B, H, W, dtype=torch.bool, device=device)

    # A trivial conv "model" so autograd has real parameters to populate.
    model = torch.nn.Conv2d(3, 3, 3, padding=1).to(device)
    x = torch.randn(B, 3, H, W, device=device)
    loss_fn = make_loss().to(device)

    # ---- fp32 reference ----
    model.zero_grad()
    total, parts = loss_fn(model(x), target, ignore=ignore)
    total.backward()
    g32 = model.weight.grad.detach().clone()
    check("fp32: gradients finite and non-zero",
          bool(torch.isfinite(g32).all()) and float(g32.abs().sum()) > 0,
          f"|g|={float(g32.abs().sum()):.4f}")
    fp32_dice = float(parts["dice"])

    if device != "cuda":
        check("AMP skipped (no CUDA)", True, "cpu run")
        return

    # ---- AMP: autocast fp16 forward + GradScaler backward ----
    scaler = torch.cuda.amp.GradScaler()
    model.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = model(x)
        amp_total, amp_parts = loss_fn(logits, target, ignore=ignore)
    check("autocast produced fp16 logits", logits.dtype == torch.float16,
          str(logits.dtype))
    scaler.scale(amp_total).backward()
    scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.1))
    gamp = model.weight.grad.detach().clone()
    check("AMP: gradients finite and non-zero",
          bool(torch.isfinite(gamp).all()) and float(gamp.abs().sum()) > 0,
          f"|g|={float(gamp.abs().sum()):.4f}")

    # THE fp16 TRAP: soft_dice_loss casts to float32 internally. Without that,
    # the squared-probability sums underflow and dice sticks at ~1.0. Compare
    # against the fp32 reference to prove the cast is doing its job.
    check("AMP dice matches fp32 dice (float32 cast works)",
          abs(float(amp_parts["dice"]) - fp32_dice) < 1e-3,
          f"amp={float(amp_parts['dice']):.6f} fp32={fp32_dice:.6f}")
    check("AMP total matches fp32 total within 1e-2",
          abs(float(amp_total) - float(total)) < 1e-2,
          f"amp={float(amp_total):.5f} fp32={float(total):.5f}")

    # A full scaler step must not produce inf/NaN params.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        t2, _ = loss_fn(model(x), target, ignore=ignore)
    scaler.scale(t2).backward()
    scaler.step(opt)
    scaler.update()
    check("full AMP optimizer step keeps params finite",
          bool(torch.isfinite(model.weight).all()))

    # Buffer must follow the module to the GPU, else CE would raise a
    # device-mismatch at the first weighted call.
    check("class weights live on the model device",
          loss_fn.weights.device.type == "cuda", str(loss_fn.weights.device))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | device {device} | "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}\n")
    test_weights()
    test_loss_values(device)
    test_ignore_mask(device)
    test_gradients_and_amp(device)

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
