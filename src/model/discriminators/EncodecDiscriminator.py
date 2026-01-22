"""
Combined EnCodec Discriminator.

Combines MS-STFT, MSD, and MPD discriminators as used in the EnCodec paper,
extended with MPD from HiFi-GAN for improved harmonic fidelity.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .stft_discriminator import MultiScaleSTFTDiscriminator
from .scale_discriminator import MultiScaleDiscriminator
from .period_discriminator import MultiPeriodDiscriminator


class EnCodecDiscriminator(nn.Module):
    """
    Complete EnCodec discriminator combining MS-STFT, MSD, and MPD.

    This is the full discriminator setup used in the EnCodec paper,
    extended with MPD from HiFi-GAN for improved harmonic fidelity.
    """

    def __init__(
        self,
        # MS-STFT parameters
        stft_filters: int = 32,
        in_channels: int = 2,
        n_ffts: List[int] = [2048, 1024, 512, 256, 128],
        # MSD parameters
        msd_scales: List[int] = [1, 2, 4],
        use_msd: bool = True,
        # MPD parameters
        mpd_periods: List[int] = [2, 3, 5, 7, 11],
        mpd_channels: List[int] = [32, 128, 512, 1024, 1024],
        use_mpd: bool = True,
    ):
        """
        Args:
            stft_filters: Base filters for STFT discriminators
            in_channels: Number of audio channels (1=mono, 2=stereo)
            n_ffts: FFT sizes for MS-STFT discriminator
            msd_scales: Scales for multi-scale waveform discriminator
            use_msd: Whether to use MSD alongside MS-STFT
            mpd_periods: Periods for multi-period discriminator
            mpd_channels: Channel progression for MPD
            use_mpd: Whether to use MPD
        """
        super().__init__()

        self.use_msd = use_msd
        self.use_mpd = use_mpd

        # MS-STFT Discriminator
        self.msstft = MultiScaleSTFTDiscriminator(
            filters=stft_filters,
            in_channels=in_channels,
            n_ffts=n_ffts,
        )

        # Multi-Scale Discriminator (optional)
        if use_msd:
            self.msd = MultiScaleDiscriminator(
                scales=msd_scales,
                in_channels=in_channels,
            )

        # Multi-Period Discriminator (optional)
        if use_mpd:
            self.mpd = MultiPeriodDiscriminator(
                periods=mpd_periods,
                in_channels=in_channels,
                channels=mpd_channels,
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, C, T] audio tensor
        Returns:
            logits: Combined list of logits from all discriminators
            fmaps: Combined list of feature maps from all discriminators
        """
        # MS-STFT outputs
        logits, fmaps = self.msstft(x)

        # MSD outputs
        if self.use_msd:
            msd_logits, msd_fmaps = self.msd(x)
            logits = logits + msd_logits
            fmaps = fmaps + msd_fmaps

        # MPD outputs
        if self.use_mpd:
            mpd_logits, mpd_fmaps = self.mpd(x)
            logits = logits + mpd_logits
            fmaps = fmaps + mpd_fmaps

        return logits, fmaps
