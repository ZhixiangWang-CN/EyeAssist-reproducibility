"""Model factories and gaze-supervision losses used by the computational comparisons.

Torch is an optional dependency so the descriptive/statistical analyses remain
installable on machines without a GPU stack.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def available_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_classifier(*, pretrained: bool = True, n_classes: int = 2):
    """ResNet-50 classifier used in the auxiliary gaze-supervision comparison."""
    import torch.nn as nn
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def classifier_forward_with_cam(model, images, class_index):
    """Return logits and a class activation map for a ResNet GAP+linear head.

    For this architecture, the closed-form CAM uses the same channel weights as
    Grad-CAM after global average pooling. `class_index` has one class per image.
    """
    import torch.nn.functional as functional

    features = model.conv1(images)
    features = model.relu(model.bn1(features))
    features = model.maxpool(features)
    features = model.layer4(model.layer3(model.layer2(model.layer1(features))))
    logits = model.fc(features.mean(dim=(2, 3)))
    weights = model.fc.weight[class_index]
    cam = functional.relu((features * weights[:, :, None, None]).sum(dim=1))
    return logits, cam


def attention_kl(model_cam, human_target, epsilon: float = 1e-8):
    """KL(human target || model CAM), averaged over the batch."""
    import torch

    model = model_cam.flatten(1).clamp_min(0)
    target = human_target.flatten(1).clamp_min(0)
    model = model / (model.sum(dim=1, keepdim=True) + epsilon)
    target = target / (target.sum(dim=1, keepdim=True) + epsilon)
    return torch.mean(
        torch.sum(target * (torch.log(target + epsilon) - torch.log(model + epsilon)), dim=1)
    )


def make_saliency_model(*, pretrained_encoder: bool = True):
    """U-Net-style decoder with an ImageNet-pretrained ResNet-34 encoder."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision import models

    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_encoder else None
    encoder = models.resnet34(weights=weights)

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x, skip):
            x = functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(torch.cat([x, skip], dim=1))

    class ResNet34UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layer1 = encoder.layer1
            self.layer2 = encoder.layer2
            self.layer3 = encoder.layer3
            self.layer4 = encoder.layer4
            self.d3 = DecoderBlock(512, 256, 256)
            self.d2 = DecoderBlock(256, 128, 128)
            self.d1 = DecoderBlock(128, 64, 64)
            self.d0 = DecoderBlock(64, 64, 32)
            self.head = nn.Conv2d(32, 1, 1)

        def forward(self, image):
            s0 = self.stem(image)
            s1 = self.layer1(self.pool(s0))
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            bottleneck = self.layer4(s3)
            decoded = self.d3(bottleneck, s3)
            decoded = self.d2(decoded, s2)
            decoded = self.d1(decoded, s1)
            decoded = self.d0(decoded, s0)
            output = self.head(decoded)
            output = functional.interpolate(
                output, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            return functional.softplus(output)

    return ResNet34UNet()


def saliency_objective(
    prediction,
    target,
    *,
    kl_weight: float = 1.0,
    correlation_weight: float = 1.0,
    epsilon: float = 1e-8,
):
    """KL(target || prediction) plus a negative Pearson-correlation term."""
    import torch

    p = prediction.flatten(1).clamp_min(0)
    q = target.flatten(1).clamp_min(0)
    p = p / (p.sum(1, keepdim=True) + epsilon)
    q = q / (q.sum(1, keepdim=True) + epsilon)
    kl = torch.sum(q * (torch.log(q + epsilon) - torch.log(p + epsilon)), dim=1).mean()
    p_centered = p - p.mean(1, keepdim=True)
    q_centered = q - q.mean(1, keepdim=True)
    corr = torch.sum(p_centered * q_centered, dim=1) / (
        torch.sqrt(torch.sum(p_centered**2, dim=1) * torch.sum(q_centered**2, dim=1))
        + epsilon
    )
    return kl_weight * kl - correlation_weight * corr.mean()
