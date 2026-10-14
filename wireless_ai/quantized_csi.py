"""Locally implemented 4-bit CNN feedback experiment on centered two-channel CSI.

Each latent coordinate has 16 levels in training and evaluation; this
models payload size without implementing a packed transport codec.
"""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn

def four_bit_ste(z: torch.Tensor) -> torch.Tensor:
    """Fixed [-1,1] 16-level uniform quantizer with straight-through gradient."""
    clipped = torch.tanh(z)
    hard = torch.round((clipped + 1.0) * 7.5) / 7.5 - 1.0
    return clipped + (hard - clipped).detach()


class QuantizedCsiAutoencoder(nn.Module):
    def __init__(self, feedback_bits: int):
        super().__init__()
        if isinstance(feedback_bits, bool) or not isinstance(feedback_bits, Integral) or feedback_bits <= 0 or feedback_bits % 4:
            raise ValueError("feedback budget must be a positive multiple of four")
        self.feedback_bits = feedback_bits
        latent = feedback_bits // 4
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(0.1),
        )
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(2048, 256), nn.LeakyReLU(0.1), nn.Linear(256, latent))
        self.decoder_fc = nn.Sequential(nn.Linear(latent, 256), nn.LeakyReLU(0.1), nn.Linear(256, 2048))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GroupNorm(4, 16), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[0] == 0 or tuple(x.shape[1:]) != (2, 32, 32):
            raise ValueError("expected nonempty [sample,2,32,32] CSI")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("CSI must contain finite floating point values")
        return four_bit_ste(self.encoder_fc(self.encoder_conv(x)))

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if code.ndim != 2 or code.shape[0] == 0 or code.shape[1] != self.feedback_bits // 4:
            raise ValueError("code shape does not match the feedback budget")
        if not torch.is_floating_point(code) or not torch.isfinite(code).all():
            raise ValueError("code must contain finite floating point values")
        return self.decoder_conv(self.decoder_fc(code).reshape(-1, 32, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def batch_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or target.ndim < 2 or target.shape[0] == 0:
        raise ValueError("expected equal nonempty [sample,...] shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("NMSE requires finite values")
    numerator = (prediction - target).square().flatten(1).sum(dim=1)
    denominator = target.square().flatten(1).sum(dim=1)
    if torch.any(denominator <= 0):
        raise ValueError("zero-power CSI has undefined NMSE")
    return numerator / denominator
