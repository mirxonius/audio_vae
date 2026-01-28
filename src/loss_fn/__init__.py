# src/loss_fn/__init__.py
"""
Audio VAE Loss Functions Module.

This module provides a comprehensive set of loss functions for training
audio variational autoencoders, including:

- Time-domain losses (waveform L1/L2)
- Spectral losses (STFT, multi-resolution STFT)
- Perceptual losses (mel spectrogram, log spectral distance)
- Stereo losses (sum/difference, left/right)
- SDR losses (SI-SDR, SD-SDR)

Usage:
    # Direct import
    from src.loss_fn import MultiResolutionSTFTLoss, WaveformLoss

    # Using the registry
    from src.loss_fn import get_loss, list_losses

    loss = get_loss("waveform", weight=1.0)
    print(list_losses())  # Show all registered losses

Loss Registry:
    The module provides a registry system for dynamic loss instantiation:
    - register_loss(name): Decorator to register new losses
    - get_loss(name, **kwargs): Instantiate a loss by name
    - list_losses(): List all registered loss names
"""

# Base class and registry
from src.loss_fn.BaseLoss import (
    BaseLoss,
    register_loss,
    get_loss,
    list_losses,
)

# STFT components (low-level building blocks)
from src.loss_fn.stft_components import (
    STFTLoss,
    STFTMagnitudeLoss,
    SpectralConvergenceLoss,
    SumAndDifference,
    LeftRight,
    get_window,
    apply_reduction,
)

# Multi-resolution STFT losses
from src.loss_fn.MultiResolutionSTFTLoss import (
    MultiResolutionSTFTLoss,
    SumAndDifferenceSTFTLoss,
    LRMultiResolutionSTFTLoss,
    CombinedStereoSTFTLoss,
    MultiScaleMelLoss,
    LogSpectralDistanceLoss,
    SISDRLoss,
    SDSDRLoss,
)

# Waveform losses
from src.loss_fn.WaveformLoss import (
    WaveformLoss,
    WaveformL2Loss,
)

# Mel spectrogram losses (using auraloss)
from src.loss_fn.MelSpectrogramLoss import (
    MultiScaleMelSpectrogramLoss,
)

# Loss calculators
from src.loss_fn.VAELossCalculator import (
    VAELossCalculator,
    VAELossOutput,
)

from src.loss_fn.AdversarialLossCalculator import (
    AdversarialLossCalculator,
)

# Dual channel STFT loss (legacy)
from src.loss_fn.DualChannelSTFTLoss import (
    DualChannelSTFTLoss,
)


__all__ = [
    # Base and registry
    "BaseLoss",
    "register_loss",
    "get_loss",
    "list_losses",
    # STFT components
    "STFTLoss",
    "STFTMagnitudeLoss",
    "SpectralConvergenceLoss",
    "SumAndDifference",
    "LeftRight",
    "get_window",
    "apply_reduction",
    # Multi-resolution losses
    "MultiResolutionSTFTLoss",
    "SumAndDifferenceSTFTLoss",
    "LRMultiResolutionSTFTLoss",
    "CombinedStereoSTFTLoss",
    "MultiScaleMelLoss",
    "LogSpectralDistanceLoss",
    # SDR losses
    "SISDRLoss",
    "SDSDRLoss",
    # Waveform losses
    "WaveformLoss",
    "WaveformL2Loss",
    # Mel losses
    "MultiScaleMelSpectrogramLoss",
    # Loss calculators
    "VAELossCalculator",
    "VAELossOutput",
    "AdversarialLossCalculator",
    # Legacy
    "DualChannelSTFTLoss",
]
