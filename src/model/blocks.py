from typing import List
import torch
import torch.nn as nn

from src.model.layers import Snake, WNConv1d, ResidualUnit, WNConvTranspose1d


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        kernel_size: int = 7,
        dilations: List[int] = [1, 3, 9],
    ):
        super().__init__()
        self.res_units = nn.Sequential(
            *[ResidualUnit(in_channels, kernel_size, dilation=d) for d in dilations]
        )
        self.downsample = nn.Sequential(
            Snake(in_channels),
            WNConv1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.res_units(x))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        kernel_size: int = 7,
        dilations: List[int] = [1, 3, 9],
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            Snake(in_channels),
            WNConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
            ),
        )
        self.res_units = nn.Sequential(
            *[ResidualUnit(out_channels, kernel_size, dilation=d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_units(self.upsample(x))
