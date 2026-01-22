"""
EnCodec-style Discriminators

Based on "High Fidelity Neural Audio Compression" (arxiv 2210.13438)

Includes:
1. MS-STFT Discriminator: Operates on complex-valued STFTs at multiple scales
2. Multi-Scale Discriminator (MSD): Operates on waveforms at multiple scales
3. Multi-Period Discriminator (MPD): Operates on waveforms reshaped by period
4. EnCodec Discriminator: Combined discriminator using all of the above
"""

from .utils import get_2d_padding

from .losses import (
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
)

from .stft_discriminator import (
    DiscriminatorSTFT,
    MultiScaleSTFTDiscriminator,
)

from .scale_discriminator import (
    ScaleDiscriminator,
    MultiScaleDiscriminator,
)

from .period_discriminator import (
    PeriodDiscriminator,
    MultiPeriodDiscriminator,
)

from .EncodecDiscriminator import EnCodecDiscriminator


__all__ = [
    # Utilities
    "get_2d_padding",
    # Loss functions
    "discriminator_loss",
    "generator_adversarial_loss",
    "feature_matching_loss",
    # STFT Discriminators
    "DiscriminatorSTFT",
    "MultiScaleSTFTDiscriminator",
    # Scale Discriminators
    "ScaleDiscriminator",
    "MultiScaleDiscriminator",
    # Period Discriminators
    "PeriodDiscriminator",
    "MultiPeriodDiscriminator",
    # Combined
    "EnCodecDiscriminator",
]
