"""
Multi-Scale Discriminator (MSD) from MelGAN/HiFi-GAN.

Operates on raw waveforms at multiple scales via average pooling.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator for waveform."""

    def __init__(self, scale: int = 1, in_channels: int = 2):
        """
        Args:
            scale: Downsampling factor via average pooling (1 = no pooling)
            in_channels: Number of audio channels
        """
        super().__init__()
        self.scale = scale

        # Use stride to downsample and increase channels
        # Similar to HiFi-GAN MSD architecture
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(in_channels, 16, kernel_size=15, stride=1, padding=7)
                ),
                weight_norm(
                    nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)
                ),
                weight_norm(
                    nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)
                ),
                weight_norm(
                    nn.Conv1d(256, 256, kernel_size=41, stride=4, padding=20, groups=64)
                ),
                weight_norm(
                    nn.Conv1d(
                        256, 512, kernel_size=41, stride=4, padding=20, groups=256
                    )
                ),
                weight_norm(nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)),
            ]
        )

        # Final layer
        self.conv_post = weight_norm(
            nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1)
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, C, T] audio
        Returns:
            logits: [B, 1, T']
            features: List of intermediate features
        """
        # Downsample if needed
        if self.scale > 1:
            x = F.avg_pool1d(x, kernel_size=self.scale, stride=self.scale)

        # Extract features
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            fmap.append(x)

        # Final output
        logits = self.conv_post(x)

        return logits, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD) from MelGAN/HiFi-GAN.

    Operates on raw waveforms at multiple scales via average pooling.
    Commonly used alongside MS-STFT discriminator.
    """

    def __init__(self, scales: List[int] = [1, 2, 4], in_channels: int = 2):
        """
        Args:
            scales: List of pooling scales (1 = no pooling)
            in_channels: Number of audio channels
        """
        super().__init__()
        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(scale, in_channels) for scale in scales]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, C, T] audio tensor
        Returns:
            logits_list: List of logits for each scale
            fmaps_list: List of feature maps for each scale
        """
        logits_list, fmaps_list = [], []
        for disc in self.discriminators:
            logits, features = disc(x)
            logits_list.append(logits)
            fmaps_list.append(features)
        return logits_list, fmaps_list
