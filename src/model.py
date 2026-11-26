"""Model construction: U-Net with a pretrained ResNet34 encoder.

Why this architecture for this dataset:

  * U-NET SKIP CONNECTIONS are the point. Median nucleus diameter is 19.7 px and
    14.9% of instances are under a 10 px equivalent diameter, so the decoder
    must recover fine spatial detail that a plain encoder-decoder destroys. The
    skips carry pre-downsample resolution straight across.
  * RESNET34 is the right capacity: 24.4M params total, deep enough to benefit
    from ImageNet pretraining, small enough that a 16-crop fp16 batch fits a
    15 GB T4 with room for activations.
  * IMAGENET PRETRAINING matters even though microscopy looks nothing like
    ImageNet - early layers learn edge/blob detectors, which is exactly what
    nucleus boundaries are. This is also why the input MUST be normalized with
    NORM_IMAGENET: those filters were fitted under those statistics.

Output is 3 raw logit channels {bg, interior, border}; softmax lives in the loss
and in the inference path, never in the model, so AMP can keep the logits in
fp16 and the loss can cast where it needs float32.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

# Input side must be divisible by 32: a 5-stage ResNet encoder halves the
# resolution five times (2^5 = 32). Our 256 crops satisfy this, and the sliding
# window feeds 256 tiles at inference, so the constraint never binds.
ENCODER_STRIDE = 32


def build_model(
    encoder: str = "resnet34",
    classes: int = 3,
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    """Construct the segmentation model.

    Parameters
    ----------
    encoder : str
        Any smp-supported encoder name. resnet34 is the locked default.
    encoder_weights : str | None
        ``"imagenet"`` for pretrained, ``None`` for random init. If this is ever
        set to None, switch the sampler's ``norm`` to the dataset stats path -
        ImageNet normalization only makes sense with ImageNet weights.
    """
    # Imported lazily so that every other module in this package stays usable in
    # an environment without smp/torch installed.
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        # No activation: the head emits raw logits.
        activation=None,
    )
    return model


def param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    encoder_lr_mult: float = 1.0,
    no_decay_on_norm_bias: bool = True,
) -> List[Dict[str, Any]]:
    """Build AdamW parameter groups.

    Two standard refinements, both off-by-default-ish and both explained:

    ``encoder_lr_mult`` < 1 trains the pretrained encoder more slowly than the
    randomly-initialized decoder. The decoder starts from noise and needs large
    steps; the encoder starts from useful features that a large step destroys in
    the first few iterations. 0.1 is the usual value. Default is 1.0 (plain
    AdamW) so the baseline is the simple thing; try 0.1 as a tuning step.

    ``no_decay_on_norm_bias`` excludes BatchNorm weights/biases and all biases
    from weight decay. Decaying a norm layer's scale pulls it toward 0, which
    fights the normalization it is meant to perform; this is near-universal
    practice and costs nothing.
    """
    decay_enc, nodecay_enc, decay_dec, nodecay_dec = [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        # ndim <= 1 catches biases and all norm-layer affine params in one test,
        # since those are the only 1-D parameter tensors in a conv net.
        is_nodecay = no_decay_on_norm_bias and (p.ndim <= 1)
        if is_encoder:
            (nodecay_enc if is_nodecay else decay_enc).append(p)
        else:
            (nodecay_dec if is_nodecay else decay_dec).append(p)

    enc_lr = lr * encoder_lr_mult
    groups = [
        {"params": decay_dec, "lr": lr, "weight_decay": weight_decay, "name": "decoder"},
        {"params": nodecay_dec, "lr": lr, "weight_decay": 0.0, "name": "decoder_nodecay"},
        {"params": decay_enc, "lr": enc_lr, "weight_decay": weight_decay, "name": "encoder"},
        {"params": nodecay_enc, "lr": enc_lr, "weight_decay": 0.0, "name": "encoder_nodecay"},
    ]
    # Drop empty groups: an optimizer group with no params is legal but makes
    # the logged LR list confusing.
    return [g for g in groups if g["params"]]


def count_params(model: nn.Module) -> Dict[str, Any]:
    """Parameter counts, for the run log."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith("encoder."))
    return {
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "encoder_M": round(enc / 1e6, 2),
        "decoder_M": round((total - enc) / 1e6, 2),
    }


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed torch/cuda. The numpy and albumentations streams are seeded by the sampler.

    ``deterministic=True`` forces cuDNN to pick reproducible kernels, which
    costs throughput (it disables the autotuner). Left off by default: the data
    pipeline is already replayable, which is what debugging actually needs.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True lets cuDNN autotune for our fixed 256x256 input shape,
        # which is a real speedup precisely because the shape never varies.
        torch.backends.cudnn.benchmark = True
