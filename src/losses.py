"""Segmentation losses for the 3-class nuclei target. Requires torch.

THE PROBLEM THIS MODULE SOLVES: the class mix measured on the real crop stream
is roughly

    background 85%   interior 13%   border 2%

An unweighted cross-entropy reaches ~85% pixel accuracy by predicting
background everywhere, and the 2% border class - the ONLY thing that separates
the 78.1% of samples containing touching nuclei - contributes so little
gradient that it is effectively ignored. Collapse the border and watershed has
nothing to cut on, so instance mAP craters even while pixel accuracy looks fine.

TWO COMPLEMENTARY TERMS:

  WEIGHTED CROSS-ENTROPY - per-pixel, calibrated, and the term that actually
  learns the border. Weighting is what buys the border class gradient.

  SOFT DICE (foreground classes only) - region-level and inherently
  scale-invariant: it measures overlap ratios, so a 30 px nucleus and a 10,000
  px nucleus contribute comparably. CE alone is dominated by large nuclei
  because it sums over pixels; dice is what protects the 14.9% of instances
  under a 10 px equivalent diameter.

Neither alone is sufficient: dice alone is unstable early (near-zero
intersection gives near-zero gradient) and poorly calibrated; CE alone
under-weights small instances. The standard, and what we use, is their sum.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Class index order must match src.data_loader: {0=bg, 1=interior, 2=border}.
N_CLASSES = 3

# Measured on the real crop stream. Recomputed by
# ``src.data_loader.compute_class_frequencies`` (which lives there because it is
# numpy-only and needs the sampler, so it stays importable without torch);
# these are the fallback defaults, written into .project_state/class_stats.json.
DEFAULT_CLASS_FREQS = (0.852701, 0.126666, 0.020633)


def class_weights(
    freqs: Sequence[float] = DEFAULT_CLASS_FREQS,
    scheme: str = "inv_sqrt",
    normalize: bool = True,
) -> torch.Tensor:
    """Turn class frequencies into CE weights.

    Schemes, and why the default is not the obvious one:

    ``none``
        All ones. The baseline that ignores the border class.
    ``inv``
        ``1/f``. Mathematically the "correct" balance, but for our mix it yields
        roughly (1.2, 7.7, 48.8). A weight of ~49 on a 2% class makes the loss
        surface dominated by a handful of ring pixels per crop, which in practice
        produces loud gradients, thick smeared borders, and over-segmentation.
    ``inv_sqrt``  (DEFAULT)
        ``1/sqrt(f)``, giving (0.30, 0.78, 1.92) normalized - a ~6.4x border/bg
        ratio. Enough emphasis to learn the border without letting it dominate.
        The standard compromise for extreme imbalance.

    NOT offered: "effective number" class-balanced weighting,
    ``(1-beta)/(1-beta^n)``. It is defined over SAMPLE COUNTS, and with pixel
    counts in the millions ``beta^n`` underflows to 0 for every class, so the
    scheme collapses to uniform weights (it silently returned (1,1,1) here).
    Re-scaling n to make it non-degenerate just reproduces ``inv``. Removed
    rather than left as a trap.

    normalize=True rescales so the mean weight is 1.0, which keeps the loss
    magnitude - and therefore a tuned learning rate - comparable across schemes.
    """
    f = np.asarray(freqs, dtype=np.float64)
    # Guard a genuinely absent class: 1/0 would give inf weights and NaN loss.
    f = np.maximum(f, 1e-8)

    if scheme == "none":
        w = np.ones_like(f)
    elif scheme == "inv":
        w = 1.0 / f
    elif scheme == "inv_sqrt":
        w = 1.0 / np.sqrt(f)
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")

    if normalize:
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore: Optional[torch.Tensor] = None,
    include_bg: bool = False,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Macro soft-dice over classes. 1 - mean(dice).

    Parameters
    ----------
    logits : (B, C, H, W) float
    target : (B, H, W) long
    ignore : (B, H, W) bool, optional
        True = exclude from the loss (padded region).
    include_bg : bool
        False by default: background is 85% of pixels, so its dice is ~0.99
        regardless of model quality and simply dilutes the mean. Averaging only
        interior+border makes the number responsive to what we care about.

    Notes
    -----
    SOFT dice: it uses the softmax PROBABILITIES, not an argmax, because argmax
    is piecewise-constant and has zero gradient almost everywhere.

    The computation is forced to float32 even under autocast. In fp16 the
    squared-probability sums underflow (fp16's smallest normal is ~6e-5, and a
    256x256 sum of small probabilities drifts badly), which shows up as a dice
    term stuck at exactly 1.0.
    """
    # float() here is the AMP guard: autocast hands us fp16 logits.
    logits = logits.float()
    prob = F.softmax(logits, dim=1)

    # One-hot the target to match prob's layout. permute moves the new class
    # axis from last to position 1 -> (B, C, H, W).
    oh = F.one_hot(target.long(), num_classes=logits.shape[1])
    oh = oh.permute(0, 3, 1, 2).float()

    if ignore is not None:
        # Zero BOTH tensors on ignored pixels so they contribute to neither the
        # intersection nor the cardinality sums.
        keep = (~ignore).unsqueeze(1).float()
        prob = prob * keep
        oh = oh * keep

    cls = slice(0, logits.shape[1]) if include_bg else slice(1, logits.shape[1])
    prob, oh = prob[:, cls], oh[:, cls]

    # Sum over batch and space, keeping the class axis: a per-class dice over
    # the whole batch ("batch dice"), which is far more stable than per-image
    # dice when a crop happens to contain zero border pixels.
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)
    card = prob.sum(dims) + oh.sum(dims)

    # smooth in BOTH numerator and denominator: for an absent class (inter=0,
    # card=0) this yields dice=1 (nothing to find, nothing predicted = correct)
    # instead of 0/0.
    dice = (2.0 * inter + smooth) / (card + smooth)
    return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Weighted cross-entropy + soft dice, with an ignore mask.

    Parameters
    ----------
    weights : torch.Tensor, optional
        Per-class CE weights, e.g. from :func:`class_weights`.
    ce_weight, dice_weight : float
        Term coefficients. 1.0/1.0 is the standard starting point; raise
        dice_weight if small nuclei are being missed, raise ce_weight if the
        border is smearing.
    label_smoothing : float
        Small values (0.01-0.05) help here because EDA showed annotations are
        drawn ~1px LARGER than the true intensity edge, i.e. boundary labels are
        genuinely noisy. Smoothing stops the model from over-fitting that bias.

    Forward signature
    -----------------
    ``forward(logits, target, ignore=None) -> (loss, parts_dict)``
    Returns the parts so the training loop can log CE and dice separately -
    a diverging total is ambiguous, but a flat dice with a falling CE is not.
    """

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_bg_dice: bool = False,
        label_smoothing: float = 0.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        # register_buffer (not a plain attribute): makes the weights move with
        # .to(device) / .cuda() and get saved in the state_dict, so a reloaded
        # checkpoint cannot silently revert to unweighted CE.
        if weights is not None:
            self.register_buffer("weights", weights.float())
        else:
            self.weights = None
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.include_bg_dice = bool(include_bg_dice)
        self.label_smoothing = float(label_smoothing)
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore: Optional[torch.Tensor] = None,
    ):
        target = target.long()

        if ignore is not None:
            # IGNORE VIA SENTINEL: ignore_index drops these pixels from both the
            # numerator and the denominator. Writing -100 into a copy is cheaper
            # than masking and re-normalizing by hand.
            ce_target = target.masked_fill(ignore, -100)
        else:
            ce_target = target

        # MANUAL weighted mean instead of reduction="mean".
        #
        # Why: with every pixel ignored, reduction="mean" divides by a zero
        # denominator and returns NaN, which then poisons every parameter on the
        # first backward pass. Our pad masks never cover a whole crop, but any
        # future use of the ignore mask (uncertain boundaries, masked regions)
        # makes that reachable, and a silent NaN is the worst possible failure.
        #
        # reduction="none" already applies the class weight to each pixel, so
        # PyTorch's mean is sum(w_yi * l_i) / sum(w_yi). Reproducing that exact
        # denominator - rather than a plain pixel count - keeps the loss scale,
        # and therefore any tuned learning rate, unchanged.
        ce_map = F.cross_entropy(
            logits, ce_target,
            weight=self.weights,
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        valid = ce_target != -100
        if self.weights is not None:
            # clamp(min=0) only sanitizes the -100 sentinel for the gather; those
            # positions are zeroed by `valid` immediately after.
            w_px = self.weights[ce_target.clamp(min=0)] * valid
        else:
            w_px = valid.to(ce_map.dtype)
        # clamp_min keeps the division finite when nothing is valid; ce_map is
        # already 0 at ignored pixels, so the result is then exactly 0.
        ce = ce_map.sum() / w_px.sum().clamp_min(1e-8)

        dice = soft_dice_loss(
            logits, target, ignore=ignore,
            include_bg=self.include_bg_dice, smooth=self.smooth,
        )

        total = self.ce_weight * ce + self.dice_weight * dice
        # .detach() on the logged parts: keeping them attached would retain the
        # whole graph for as long as the log dict is alive.
        return total, {"ce": ce.detach(), "dice": dice.detach(), "total": total.detach()}


def make_loss(
    freqs: Sequence[float] = DEFAULT_CLASS_FREQS,
    scheme: str = "inv_sqrt",
    **kwargs: Any,
) -> CEDiceLoss:
    """Convenience constructor: frequencies -> weights -> loss module."""
    return CEDiceLoss(weights=class_weights(freqs, scheme=scheme), **kwargs)
