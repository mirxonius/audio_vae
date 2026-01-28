import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


class Snake(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake(channels),
            WNConv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            Snake(channels),
            WNConv1d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
