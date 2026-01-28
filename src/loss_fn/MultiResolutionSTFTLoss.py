# src/loss_fn/MultiResolutionSTFTLoss.py
"""
Multi-resolution STFT losses for audio reconstruction.

This module provides various multi-resolution spectral losses including:
- MultiResolutionSTFTLoss: Standard multi-resolution STFT loss
- SumAndDifferenceSTFTLoss: Stereo loss using Mid/Side encoding
- LRMultiResolutionSTFTLoss: Stereo loss using Left/Right channels

References:
    - Yamamoto et al., 2019 (https://arxiv.org/abs/1910.11480)
    - Steinmetz et al., 2020 (https://arxiv.org/abs/2010.10291)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from src.loss_fn.BaseLoss import BaseLoss, register_loss
from src.loss_fn.stft_components import (
    STFTLoss,
    SumAndDifference,
    LeftRight,
    get_window,
    apply_reduction,
)


@register_loss("multi_resolution_stft")
class MultiResolutionSTFTLoss(BaseLoss):
    """
    Multi-resolution STFT loss.

    Computes STFT losses at multiple resolutions (FFT sizes) and averages them.
    This captures both fine-grained and coarse-grained spectral features.

    Reference: Yamamoto et al., 2019 (https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes: List of FFT sizes for each resolution
        hop_sizes: List of hop sizes (must match fft_sizes length)
        win_lengths: List of window lengths (must match fft_sizes length)
        window: Window function name (default: 'hann_window')
        w_sc: Spectral convergence weight (default: 1.0)
        w_log_mag: Log magnitude loss weight (default: 1.0)
        w_lin_mag: Linear magnitude loss weight (default: 0.0)
        w_phs: Phase loss weight (default: 0.0)
        sample_rate: Sample rate for mel/perceptual weighting
        scale: Frequency scaling ('mel', 'chroma', or None)
        n_bins: Number of frequency bins per resolution for scaling
        perceptual_weighting: Apply A-weighting (default: False)
        scale_invariance: Use scale-invariant loss (default: False)
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: Optional[float] = None,
        scale: Optional[str] = None,
        n_bins: Optional[List[int]] = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        weight: float = 1.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(weight=weight, name=name)

        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have equal length")

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = nn.ModuleList()
        for i, (fs, hs, wl) in enumerate(zip(fft_sizes, hop_sizes, win_lengths)):
            n_bin = n_bins[i] if (scale == "mel" and n_bins is not None) else None
            self.stft_losses.append(
                STFTLoss(
                    fft_size=fs,
                    hop_size=hs,
                    win_length=wl,
                    window=window,
                    w_sc=w_sc,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    w_phs=w_phs,
                    sample_rate=sample_rate,
                    scale=scale,
                    n_bins=n_bin,
                    perceptual_weighting=perceptual_weighting,
                    scale_invariance=scale_invariance,
                    **kwargs,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            x: Predicted audio (B, C, T)
            y: Target audio (B, C, T)

        Returns:
            Averaged loss across all resolutions
        """
        total_loss = 0.0
        for stft_loss in self.stft_losses:
            total_loss = total_loss + stft_loss(x, y)
        return total_loss / len(self.stft_losses)


@register_loss("sum_and_difference_stft")
class SumAndDifferenceSTFTLoss(BaseLoss):
    """
    Sum and Difference (Mid/Side) stereo STFT loss.

    Computes multi-resolution STFT loss on:
    - Sum signal (Mid): L + R
    - Difference signal (Side): L - R

    This is effective for stereo audio as it explicitly models the spatial
    information (difference) separately from the mono content (sum).

    Reference: Steinmetz et al., 2020 (https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes: List of FFT sizes for each resolution
        hop_sizes: List of hop sizes
        win_lengths: List of window lengths
        window: Window function name (default: 'hann_window')
        w_sum: Weight for sum (mid) loss (default: 1.0)
        w_diff: Weight for difference (side) loss (default: 1.0)
        w_log_mag: Log magnitude loss weight (default: 1.0)
        w_lin_mag: Linear magnitude loss weight (default: 0.0)
        sample_rate: Sample rate for perceptual weighting
        perceptual_weighting: Apply A-weighting (default: False)
        output: Return format, 'loss' or 'full' (default: 'loss')
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        w_sum: float = 1.0,
        w_diff: float = 1.0,
        output: str = "loss",
        weight: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        sample_rate: Optional[int] = None,
        perceptual_weighting: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(weight=weight, name=name)

        self.sum_and_diff = SumAndDifference()
        self.w_sum = w_sum
        self.w_diff = w_diff
        self.output = output

        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            w_lin_mag=w_lin_mag,
            w_log_mag=w_log_mag,
            sample_rate=sample_rate,
            perceptual_weighting=perceptual_weighting,
            **kwargs,
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sum/Difference STFT loss for stereo audio.

        Args:
            input: Predicted stereo audio (B, 2, T)
            target: Target stereo audio (B, 2, T)

        Returns:
            Combined loss (sum_loss * w_sum + diff_loss * w_diff) / 2
            If output='full': returns (loss, sum_loss, diff_loss)
        """
        if input.shape != target.shape:
            raise ValueError(f"Shape mismatch: input {input.shape} vs target {target.shape}")

        # Extract sum and difference signals
        input_sum, input_diff = self.sum_and_diff(input)
        target_sum, target_diff = self.sum_and_diff(target)

        # Compute STFT losses
        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = (self.w_sum * sum_loss + self.w_diff * diff_loss) / 2

        if self.output == "loss":
            return loss
        return loss, sum_loss, diff_loss


@register_loss("lr_multi_resolution_stft")
class LRMultiResolutionSTFTLoss(BaseLoss):
    """
    Left/Right channel multi-resolution STFT loss.

    Computes multi-resolution STFT loss on left and right channels independently,
    then combines them. This is complementary to SumAndDifferenceSTFTLoss.

    While Sum/Difference encodes spatial information explicitly, Left/Right
    processing can be useful when channel independence is important or when
    the stereo image should be preserved exactly.

    Args:
        fft_sizes: List of FFT sizes for each resolution
        hop_sizes: List of hop sizes
        win_lengths: List of window lengths
        window: Window function name (default: 'hann_window')
        w_left: Weight for left channel loss (default: 1.0)
        w_right: Weight for right channel loss (default: 1.0)
        w_log_mag: Log magnitude loss weight (default: 1.0)
        w_lin_mag: Linear magnitude loss weight (default: 0.0)
        sample_rate: Sample rate for perceptual weighting
        perceptual_weighting: Apply A-weighting (default: False)
        output: Return format, 'loss' or 'full' (default: 'loss')
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        w_left: float = 1.0,
        w_right: float = 1.0,
        output: str = "loss",
        weight: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        sample_rate: Optional[int] = None,
        perceptual_weighting: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(weight=weight, name=name)

        self.left_right = LeftRight()
        self.w_left = w_left
        self.w_right = w_right
        self.output = output

        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            w_lin_mag=w_lin_mag,
            w_log_mag=w_log_mag,
            sample_rate=sample_rate,
            perceptual_weighting=perceptual_weighting,
            **kwargs,
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Left/Right channel STFT loss for stereo audio.

        Args:
            input: Predicted stereo audio (B, 2, T)
            target: Target stereo audio (B, 2, T)

        Returns:
            Combined loss (left_loss * w_left + right_loss * w_right) / 2
            If output='full': returns (loss, left_loss, right_loss)
        """
        if input.shape != target.shape:
            raise ValueError(f"Shape mismatch: input {input.shape} vs target {target.shape}")

        # Extract left and right channels
        input_left, input_right = self.left_right(input)
        target_left, target_right = self.left_right(target)

        # Compute STFT losses
        left_loss = self.mrstft(input_left, target_left)
        right_loss = self.mrstft(input_right, target_right)
        loss = (self.w_left * left_loss + self.w_right * right_loss) / 2

        if self.output == "loss":
            return loss
        return loss, left_loss, right_loss


@register_loss("combined_stereo_stft")
class CombinedStereoSTFTLoss(BaseLoss):
    """
    Combined stereo STFT loss using both Mid/Side and Left/Right representations.

    This loss combines:
    - Sum/Difference (Mid/Side) loss for spatial encoding
    - Left/Right loss for channel-specific accuracy

    Using both representations provides a more complete stereo reconstruction
    signal than either alone.

    Args:
        fft_sizes: List of FFT sizes for each resolution
        hop_sizes: List of hop sizes
        win_lengths: List of window lengths
        window: Window function name (default: 'hann_window')
        w_ms: Weight for Mid/Side loss component (default: 1.0)
        w_lr: Weight for Left/Right loss component (default: 1.0)
        w_log_mag: Log magnitude loss weight (default: 1.0)
        w_lin_mag: Linear magnitude loss weight (default: 0.0)
        sample_rate: Sample rate for perceptual weighting
        perceptual_weighting: Apply A-weighting (default: False)
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        w_ms: float = 1.0,
        w_lr: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        sample_rate: Optional[int] = None,
        perceptual_weighting: bool = False,
        weight: float = 1.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(weight=weight, name=name)

        self.w_ms = w_ms
        self.w_lr = w_lr

        common_kwargs = dict(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            w_log_mag=w_log_mag,
            w_lin_mag=w_lin_mag,
            sample_rate=sample_rate,
            perceptual_weighting=perceptual_weighting,
            **kwargs,
        )

        self.ms_loss = SumAndDifferenceSTFTLoss(**common_kwargs)
        self.lr_loss = LRMultiResolutionSTFTLoss(**common_kwargs)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined stereo STFT loss.

        Args:
            input: Predicted stereo audio (B, 2, T)
            target: Target stereo audio (B, 2, T)

        Returns:
            Combined loss: (ms_loss * w_ms + lr_loss * w_lr) / (w_ms + w_lr)
        """
        ms_loss = self.ms_loss(input, target)
        lr_loss = self.lr_loss(input, target)
        return (self.w_ms * ms_loss + self.w_lr * lr_loss) / (self.w_ms + self.w_lr)


# =============================================================================
# SDR Losses
# =============================================================================

class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.

    Returns the negative SI-SDR (for minimization).

    Reference: Le Roux et al., 2018 (https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean: Remove DC offset (default: True)
        eps: Numerical stability epsilon (default: 1e-8)
        reduction: Reduction method (default: 'mean')
    """

    def __init__(
        self,
        zero_mean: bool = True,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if self.zero_mean:
            input = input - input.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)

        alpha = (input * target).sum(-1) / ((target ** 2).sum(-1) + self.eps)
        target_scaled = target * alpha.unsqueeze(-1)
        residual = input - target_scaled

        si_sdr = 10 * torch.log10(
            (target_scaled ** 2).sum(-1) / ((residual ** 2).sum(-1) + self.eps) + self.eps
        )
        return -apply_reduction(si_sdr, self.reduction)


class SDSDRLoss(nn.Module):
    """
    Scale-Dependent Signal-to-Distortion Ratio (SD-SDR) loss.

    Returns the negative SD-SDR (for minimization).

    Reference: Le Roux et al., 2018 (https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean: Remove DC offset (default: True)
        eps: Numerical stability epsilon (default: 1e-8)
        reduction: Reduction method (default: 'mean')
    """

    def __init__(
        self,
        zero_mean: bool = True,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if self.zero_mean:
            input = input - input.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)

        alpha = (input * target).sum(-1) / ((target ** 2).sum(-1) + self.eps)
        target_scaled = target * alpha.unsqueeze(-1)
        residual = input - target  # Note: uses unscaled target for SD-SDR

        sd_sdr = 10 * torch.log10(
            (target_scaled ** 2).sum(-1) / ((residual ** 2).sum(-1) + self.eps) + self.eps
        )
        return -apply_reduction(sd_sdr, self.reduction)


# =============================================================================
# Mel-scale Losses
# =============================================================================

@register_loss("multi_scale_mel")
class MultiScaleMelLoss(BaseLoss):
    """
    Multi-resolution Mel Spectrogram Loss.

    Uses multiple STFT configurations with Mel-scaled spectrograms.
    Effective for audio VAEs as it aligns with human perception.

    Args:
        fft_sizes: List of FFT sizes for each resolution
        hop_sizes: List of hop sizes
        win_lengths: List of window lengths
        n_mels: List of mel bin counts per resolution
        window: Window function name (default: 'hann_window')
        w_sc: Spectral convergence weight (default: 1.0)
        w_log_mag: Log magnitude loss weight (default: 1.0)
        w_lin_mag: Linear magnitude loss weight (default: 0.0)
        w_phs: Phase loss weight (default: 0.0)
        sample_rate: Sample rate (required)
        perceptual_weighting: Apply A-weighting (default: False)
        scale_invariance: Use scale-invariant loss (default: False)
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [256, 512, 128],
        win_lengths: List[int] = [1024, 2048, 512],
        n_mels: List[int] = [128, 256, 64],
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: Optional[float] = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        weight: float = 1.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(weight=weight, name=name)

        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths) == len(n_mels)):
            raise ValueError("All parameter lists must have the same length")
        if sample_rate is None:
            raise ValueError("sample_rate required for Mel spectrograms")

        self.mel_stft_losses = nn.ModuleList()
        for fs, hs, wl, nm in zip(fft_sizes, hop_sizes, win_lengths, n_mels):
            self.mel_stft_losses.append(
                STFTLoss(
                    fft_size=fs,
                    hop_size=hs,
                    win_length=wl,
                    window=window,
                    w_sc=w_sc,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    w_phs=w_phs,
                    sample_rate=sample_rate,
                    scale="mel",
                    n_bins=nm,
                    perceptual_weighting=perceptual_weighting,
                    scale_invariance=scale_invariance,
                    output="loss",
                    **kwargs,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn in self.mel_stft_losses:
            total_loss = total_loss + loss_fn(x, y)
        return total_loss / len(self.mel_stft_losses)


@register_loss("log_spectral_distance")
class LogSpectralDistanceLoss(BaseLoss):
    """
    Log-Spectral Distance (LSD) Loss.

    Measures the Euclidean distance between log-magnitude spectra.
    A common perceptual metric in speech and audio processing.

    LSD = sqrt( mean( |20*log10(|X|) - 20*log10(|Y|)|^2 ) )

    Args:
        fft_size: FFT size (default: 1024)
        hop_size: Hop size (default: 256)
        win_length: Window length (default: 1024)
        window: Window function name (default: 'hann_window')
        eps: Log stability epsilon (default: 1e-8)
        reduction: Reduction method (default: 'mean')
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        eps: float = 1e-8,
        reduction: str = "mean",
        weight: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(weight=weight, name=name)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.eps = eps
        self.reduction = reduction

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        self.window = self.window.to(x.device)
        x_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        return torch.abs(x_stft)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bs, chs, seq_len = input.shape
        input_flat = input.view(-1, seq_len)
        target_flat = target.view(-1, seq_len)

        input_mag = self._stft_magnitude(input_flat)
        target_mag = self._stft_magnitude(target_flat)

        input_log_mag = 20 * torch.log10(input_mag + self.eps)
        target_log_mag = 20 * torch.log10(target_mag + self.eps)

        squared_diff = (input_log_mag - target_log_mag) ** 2
        mean_squared = torch.mean(squared_diff, dim=[-2, -1])
        lsd = torch.sqrt(mean_squared).view(bs, chs)

        return apply_reduction(lsd, self.reduction)


# Backwards compatibility exports
__all__ = [
    # Main losses
    "MultiResolutionSTFTLoss",
    "SumAndDifferenceSTFTLoss",
    "LRMultiResolutionSTFTLoss",
    "CombinedStereoSTFTLoss",
    # SDR losses
    "SISDRLoss",
    "SDSDRLoss",
    # Mel losses
    "MultiScaleMelLoss",
    "LogSpectralDistanceLoss",
    # Components (for advanced usage)
    "STFTLoss",
    "SumAndDifference",
    "LeftRight",
]

# Re-export components for backwards compatibility
from src.loss_fn.stft_components import (
    STFTLoss,
    SumAndDifference,
    SpectralConvergenceLoss,
    STFTMagnitudeLoss,
)
