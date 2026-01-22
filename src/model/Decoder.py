from typing import List, Tuple
import torch
import torch.nn as nn
import math
from src.model.layers import Snake, WNConv1d, ResidualUnit
from src.model.blocks import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        strides: List[int] = [2, 4, 8, 8],
        latent_dim: int = 64,
        kernel_size: int = 7,
        dilations: List[int] = [1, 3, 9],
    ):
        super().__init__()
        channel_mults = list(reversed(channel_mults))
        strides = list(reversed(strides))

        self.input_conv = WNConv1d(
            latent_dim, base_channels * channel_mults[0], kernel_size=7, padding=3
        )

        self.blocks = nn.ModuleList()
        channels = base_channels * channel_mults[0]
        for i, (mult, stride) in enumerate(zip(channel_mults, strides)):
            out_ch = (
                base_channels * channel_mults[i + 1]
                if i < len(channel_mults) - 1
                else base_channels
            )
            self.blocks.append(
                DecoderBlock(channels, out_ch, stride, kernel_size, dilations)
            )
            channels = out_ch

        self.final_conv = nn.Sequential(
            Snake(channels),
            WNConv1d(channels, out_channels, kernel_size=7, padding=3),
            # nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(z)
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)
