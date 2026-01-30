"""
Combined EnCodec Discriminator.

Combines MS-STFT, MSD, MPD, and Mel-STFT discriminators as used in the EnCodec paper,
extended with MPD from HiFi-GAN for improved harmonic fidelity and optionally
Mel-STFT for perceptually-motivated discrimination.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from .stft_discriminator import MultiScaleSTFTDiscriminator
from .scale_discriminator import MultiScaleDiscriminator
from .period_discriminator import MultiPeriodDiscriminator
from .mel_stft_discriminator import MultiScaleMelSTFTDiscriminator


class EnCodecDiscriminator(nn.Module):
    """
    Complete EnCodec discriminator combining MS-STFT, MSD, MPD, and Mel-STFT.

    This is the full discriminator setup used in the EnCodec paper,
    extended with MPD from HiFi-GAN for improved harmonic fidelity
    and optionally Mel-STFT for perceptually-motivated discrimination.
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
        # Mel-STFT parameters
        use_mel_stft: bool = False,
        mel_stft_filters: int = 32,
        mel_stft_n_ffts: List[int] = [2048, 1024, 512, 256, 128],
        mel_stft_n_mels: List[int] = [128, 128, 80, 64, 32],
        mel_stft_sample_rate: int = 44100,
        mel_stft_f_min: float = 0.0,
        mel_stft_f_max: Optional[float] = None,
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
            use_mel_stft: Whether to use Mel-STFT discriminator
            mel_stft_filters: Base filters for Mel-STFT discriminators
            mel_stft_n_ffts: FFT sizes for Mel-STFT discriminator
            mel_stft_n_mels: Number of mel bins for each scale
            mel_stft_sample_rate: Audio sample rate for mel filterbank
            mel_stft_f_min: Minimum frequency for mel filterbank
            mel_stft_f_max: Maximum frequency for mel filterbank
        """
        super().__init__()

        self.use_msd = use_msd
        self.use_mpd = use_mpd
        self.use_mel_stft = use_mel_stft

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

        # Mel-STFT Discriminator (optional)
        if use_mel_stft:
            self.mel_stft = MultiScaleMelSTFTDiscriminator(
                filters=mel_stft_filters,
                in_channels=in_channels,
                n_ffts=mel_stft_n_ffts,
                n_mels=mel_stft_n_mels,
                sample_rate=mel_stft_sample_rate,
                f_min=mel_stft_f_min,
                f_max=mel_stft_f_max,
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

        # Mel-STFT outputs
        if self.use_mel_stft:
            mel_stft_logits, mel_stft_fmaps = self.mel_stft(x)
            logits = logits + mel_stft_logits
            fmaps = fmaps + mel_stft_fmaps

        return logits, fmaps
