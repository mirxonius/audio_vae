from typing import List, Tuple
import torch
import torch.nn as nn
import math
from src.model.layers import Snake, WNConv1d, ResidualUnit
from src.model.blocks import EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        strides: List[int] = [2, 4, 8, 8],
        latent_dim: int = 64,
        kernel_size: int = 7,
        dilations: List[int] = [1, 3, 9],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.strides = strides
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.hop_length = math.prod(strides)
        self.input_conv = WNConv1d(in_channels, base_channels, kernel_size=7, padding=3)

        self.blocks = nn.ModuleList()
        channels = base_channels
        for mult, stride in zip(channel_mults, strides):
            out_ch = base_channels * mult
            self.blocks.append(
                EncoderBlock(channels, out_ch, stride, kernel_size, dilations)
            )
            channels = out_ch

        self.final_conv = nn.Sequential(
            Snake(channels),
            WNConv1d(channels, latent_dim * 2, kernel_size=3, padding=1),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar
