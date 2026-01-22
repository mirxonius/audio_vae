"""
Multi-Period Discriminator (MPD) from HiFi-GAN.

Reshapes 1D waveform into 2D using fixed periods, then applies
2D convolutions to capture periodic/harmonic patterns.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


class PeriodDiscriminator(nn.Module):
    """
    Single period discriminator for MPD.

    Reshapes 1D waveform into 2D using a fixed period, then applies
    2D convolutions to capture periodic patterns.
    """

    def __init__(
        self,
        period: int,
        in_channels: int = 2,
        kernel_size: int = 5,
        stride: int = 3,
        channels: List[int] = [32, 128, 512, 512, 512],
    ):
        """
        Args:
            period: Period for reshaping (e.g., 2, 3, 5, 7, 11)
            in_channels: Number of audio channels
            kernel_size: Kernel size for 2D convolutions
            stride: Stride in frequency dimension
            channels: Channel progression through layers
        """
        super().__init__()
        self.period = period

        # Build conv layers with increasing channels
        self.convs = nn.ModuleList()

        in_ch = in_channels
        for i, out_ch in enumerate(channels):
            # Last layer uses stride 1
            s = 1 if i == len(channels) - 1 else stride
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(kernel_size, 1),
                        stride=(s, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
            )
            in_ch = out_ch

        # Final prediction layer
        self.conv_post = weight_norm(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, C, T] audio tensor
        Returns:
            logits: [B, 1, T', 1] discriminator output
            fmap: List of feature maps
        """
        B, C, T = x.shape

        # Pad to make T divisible by period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            T = T + pad_len

        # Reshape: [B, C, T] -> [B, C, T/period, period]
        x = rearrange(x, "b c (T p) -> b c T p", p=self.period)
        x = x.view(B, C, T // self.period, self.period)

        # Apply convolutions
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            fmap.append(x)

        # Final output
        logits = self.conv_post(x)

        return logits, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) from HiFi-GAN.

    Uses prime periods to capture different harmonic structures.
    Standard periods: [2, 3, 5, 7, 11] (primes to avoid overlap)
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        in_channels: int = 2,
        channels: List[int] = [32, 128, 512, 1024, 1024],
    ):
        """
        Args:
            periods: List of periods (typically prime numbers)
            in_channels: Number of audio channels
            channels: Channel progression for each sub-discriminator
        """
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(
                    period=p,
                    in_channels=in_channels,
                    channels=channels,
                )
                for p in periods
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, C, T] audio tensor
        Returns:
            logits_list: List of logits from each period discriminator
            fmaps_list: List of feature map lists from each discriminator
        """
        logits_list = []
        fmaps_list = []

        for disc in self.discriminators:
            logits, fmaps = disc(x)
            logits_list.append(logits)
            fmaps_list.append(fmaps)

        return logits_list, fmaps_list
