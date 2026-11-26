"""L6 - full-image inference by sliding window, with overlap blending.

WHY THIS EXISTS: validation images run up to 1024x1024, but the model consumes
256x256. Scoring on random crops is not a validation - instance mAP is defined
over whole images, and a nucleus cut by a crop edge is a different object than
the one the metric expects. So val and test must tile the full image and stitch.

THE SEAM PROBLEM: a tile's border pixels are predicted from truncated context,
so they are systematically worse than its centre pixels. Averaging tiles
uniformly spreads that error along every seam, which is exactly where the
border class lives, and the border class is what watershed cuts on. The fix is
a raised-cosine (Hann) weight per tile, so each pixel's prediction is dominated
by the tile that saw it with the most context.

Stride 192 on a 256 window = 64 px (25%) overlap. Cheap: the whole 135-image
val split is 806 tiles.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.data_loader import resolve_norm

# 25% overlap. Lower stride = better seams but more compute; 192 is the usual
# sweet spot and keeps the val pass under ~1k tiles.
DEFAULT_STRIDE = 192


def _tile_positions(size: int, crop: int, stride: int) -> List[int]:
    """Start offsets covering ``size`` with ``crop`` windows at ``stride``.

    The LAST position is clamped to ``size - crop`` rather than extending past
    the edge, so the final tile overlaps its neighbour more than `stride`. That
    guarantees full coverage with no padding and no out-of-range read.
    """
    if size <= crop:
        return [0]
    pos = list(range(0, size - crop + 1, stride))
    if pos[-1] != size - crop:
        pos.append(size - crop)
    return pos


def _hann_weight(crop: int, floor: float = 0.05) -> np.ndarray:
    """2-D raised-cosine tile weight, with a positive floor.

    np.hanning is exactly 0 at both ends. At the IMAGE border a pixel is covered
    by only one tile, so a zero weight there would make the accumulated weight 0
    and the normalized output NaN. The floor keeps every weight strictly
    positive while preserving the centre-favouring shape.

    The outer product of two 1-D windows gives the separable 2-D window.
    """
    w = np.hanning(crop + 2)[1:-1]          # drop the exact zeros at the ends
    w = floor + (1.0 - floor) * (w / w.max())
    return np.outer(w, w).astype(np.float32)


@torch.no_grad()
def sliding_window_probs(
    model: torch.nn.Module,
    image: np.ndarray,
    crop: int = 256,
    stride: int = DEFAULT_STRIDE,
    norm: Any = "imagenet",
    device: str = "cuda",
    amp: bool = True,
    batch_size: int = 8,
    blend: str = "hann",
) -> np.ndarray:
    """Predict per-pixel 3-class probabilities for ONE full image.

    Parameters
    ----------
    model : torch.nn.Module
        Eval-mode model emitting (B, 3, H, W) logits.
    image : np.ndarray
        ``(H, W, 3)`` uint8, ALREADY polarity-flipped (i.e. straight from the
        L2 cache). This function applies only the ``/255`` and standardization
        steps, matching training exactly.
    norm : see :func:`src.data_loader.resolve_norm`
        MUST match what training used, or every activation is shifted.

    Returns
    -------
    np.ndarray
        ``(3, H, W)`` float32, softmax probabilities summing to 1 per pixel.

    Notes
    -----
    Memory: one accumulator (3, H, W) float32 plus one weight plane (H, W).
    For the largest image that is ~17 MB, so full-resolution stitching is fine.
    """
    model.eval()
    H, W = image.shape[:2]
    stats = resolve_norm(norm)

    # Pad only if the image is smaller than one tile. BORDER_REPLICATE, not
    # zeros, for the same reason as in the sampler: a black frame mimics
    # fluorescence background and shifts the encoder's statistics.
    pad_h, pad_w = max(0, crop - H), max(0, crop - W)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    Hp, Wp = image.shape[:2]

    tops = _tile_positions(Hp, crop, stride)
    lefts = _tile_positions(Wp, crop, stride)

    wmap = _hann_weight(crop) if blend == "hann" else np.ones((crop, crop), np.float32)

    # float32 accumulators: fp16 sums over ~10 overlapping tiles lose precision
    # in the low-probability tail, which is where the border class often sits.
    acc = np.zeros((3, Hp, Wp), dtype=np.float32)
    wsum = np.zeros((Hp, Wp), dtype=np.float32)

    coords = [(t, l) for t in tops for l in lefts]

    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        # Build the batch on CPU in NHWC, then convert once - the same path the
        # training collate takes, so train and inference preprocessing cannot
        # drift apart.
        batch = np.stack([image[t:t + crop, l:l + crop] for t, l in chunk])
        batch = batch.astype(np.float32)
        batch /= 255.0
        if stats is not None:
            mean, std = stats
            batch -= mean
            batch /= std
        x = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(device)

        if amp and device == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x)
            # softmax in float32: fp16 softmax of confident logits saturates and
            # loses the small-probability detail the watershed elevation uses.
            probs = torch.softmax(logits.float(), dim=1)
        else:
            probs = torch.softmax(model(x).float(), dim=1)

        p = probs.cpu().numpy()
        for j, (t, l) in enumerate(chunk):
            # Broadcasting (3,crop,crop) * (crop,crop) applies the same spatial
            # weight to all 3 class planes.
            acc[:, t:t + crop, l:l + crop] += p[j] * wmap
            wsum[t:t + crop, l:l + crop] += wmap

    # wsum is strictly positive everywhere thanks to the Hann floor and full
    # coverage, so this division is safe without a clamp - but clamp anyway,
    # because a silent NaN here would poison the metric rather than crash.
    out = acc / np.maximum(wsum, 1e-8)[None]
    # Undo the padding so the result aligns with the original label map.
    out = out[:, :H, :W]
    # Re-normalize: the weighted average of per-tile distributions is already
    # near-normalized, but accumulated float error can leave it off by ~1e-6.
    return out / np.maximum(out.sum(axis=0, keepdims=True), 1e-8)


def tile_count(h: int, w: int, crop: int = 256, stride: int = DEFAULT_STRIDE) -> int:
    """Number of tiles a given image size produces. For cost estimates."""
    return (len(_tile_positions(max(h, crop), crop, stride))
            * len(_tile_positions(max(w, crop), crop, stride)))
