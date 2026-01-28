# src/loss_fn/stft_components.py
"""
Core STFT components for audio loss computation.

This module contains the building blocks used by multi-resolution STFT losses:
- Window functions
- Sum/Difference stereo transforms
- Spectral convergence and magnitude loss components
- Single-resolution STFT loss
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.signal
from typing import Any, Optional


def get_window(win_type: str, win_length: int) -> torch.Tensor:
    """
    Return a window function as a 1D tensor.

    Args:
        win_type: Window type. Can be a PyTorch window function
            ('hann_window', 'bartlett_window', 'blackman_window',
             'hamming_window', 'kaiser_window') or any SciPy window.
        win_length: Window length in samples

    Returns:
        Window as a 1D torch tensor
    """
    try:
        win = getattr(torch, win_type)(win_length)
    except AttributeError:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))
    return win


def apply_reduction(
    losses: torch.Tensor,
    reduction: str = "none",
    retain_batch_dim: bool = False
) -> torch.Tensor:
    """
    Apply reduction to loss tensor.

    Args:
        losses: Input loss tensor
        reduction: One of 'none', 'mean', 'sum'
        retain_batch_dim: If True and losses is 3D, reduce over last 2 dims only

    Returns:
        Reduced loss tensor
    """
    dim = [-1, -2] if retain_batch_dim and len(losses.shape) == 3 else None
    if reduction == "mean":
        losses = losses.mean(dim=dim)
    elif reduction == "sum":
        losses = losses.sum(dim=dim)
    return losses


def normalized_complex_distance_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Normalized complex distance loss.

    Args:
        x: Predicted complex spectrogram
        y: Target complex spectrogram
        eps: Small value for numerical stability

    Returns:
        Normalized distance loss
    """
    numerator = torch.nn.functional.l1_loss(x, y, reduction="none").abs()
    denominator = 0.5 * (x.abs() + y.abs()) + eps
    return numerator / denominator


class SumAndDifference(nn.Module):
    """
    Sum and difference signal extraction for stereo audio.

    Converts Left/Right channels to Mid/Side representation:
    - Sum (Mid): L + R
    - Difference (Side): L - R
    """

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Extract sum and difference signals.

        Args:
            x: Stereo audio tensor (B, 2, T)

        Returns:
            Tuple of (sum_signal, diff_signal), each (B, 1, T)

        Raises:
            ValueError: If input is not stereo (2 channels)
        """
        if x.size(1) != 2:
            raise ValueError(f"Input must be stereo (2 channels), got {x.size(1)}")

        sum_sig = (x[:, 0, :] + x[:, 1, :]).unsqueeze(1)
        diff_sig = (x[:, 0, :] - x[:, 1, :]).unsqueeze(1)
        return sum_sig, diff_sig

    @staticmethod
    def sum(x: torch.Tensor) -> torch.Tensor:
        """Extract sum (mid) signal: L + R"""
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x: torch.Tensor) -> torch.Tensor:
        """Extract difference (side) signal: L - R"""
        return x[:, 0, :] - x[:, 1, :]


class LeftRight(nn.Module):
    """
    Left and Right channel extraction for stereo audio.

    Simply separates the two channels for independent processing.
    Complementary to SumAndDifference for stereo loss computation.
    """

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Extract left and right channels.

        Args:
            x: Stereo audio tensor (B, 2, T)

        Returns:
            Tuple of (left_channel, right_channel), each (B, 1, T)

        Raises:
            ValueError: If input is not stereo (2 channels)
        """
        if x.size(1) != 2:
            raise ValueError(f"Input must be stereo (2 channels), got {x.size(1)}")

        left = x[:, 0:1, :]
        right = x[:, 1:2, :]
        return left, right


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral convergence loss.

    Measures the Frobenius norm of the difference between predicted and target
    magnitudes, normalized by the target magnitude.

    Reference: Arik et al., 2018 (https://arxiv.org/abs/1808.06719)
    """

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral convergence loss.

        Args:
            x_mag: Predicted magnitude spectrogram
            y_mag: Target magnitude spectrogram

        Returns:
            Spectral convergence loss
        """
        return (
            torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2])
            / (torch.norm(y_mag, p="fro", dim=[-1, -2]) + 1e-5)
        ).unsqueeze(-1).unsqueeze(-1)


class STFTMagnitudeLoss(nn.Module):
    """
    STFT magnitude loss with optional log compression.

    Computes L1 or L2 distance between magnitude spectrograms, optionally
    in log domain.

    References:
        - Arik et al., 2018 (https://arxiv.org/abs/1808.06719)
        - Engel et al., 2020 (https://arxiv.org/abs/2001.04643v1)

    Args:
        log: Use log-magnitude (default: True)
        log_eps: Epsilon for log stability (default: 1e-8)
        log_fac: Magnitude scaling factor before log (default: 1.0)
        distance: Distance function, 'L1' or 'L2' (default: 'L1')
        reduction: Reduction method (default: 'mean')
    """

    def __init__(
        self,
        log: bool = True,
        log_eps: float = 1e-8,
        log_fac: float = 1.0,
        distance: str = "L1",
        reduction: str = "mean",
    ):
        super().__init__()
        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'. Use 'L1' or 'L2'.")

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)


class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss.

    Combines spectral convergence, log-magnitude, linear magnitude, and phase
    losses at a single STFT resolution.

    Reference: Yamamoto et al. 2019 (https://arxiv.org/abs/1904.04472)

    Args:
        fft_size: FFT size in samples (default: 1024)
        hop_size: Hop size in samples (default: 256)
        win_length: Window length (default: 1024)
        window: Window function name (default: 'hann_window')
        w_sc: Spectral convergence weight (default: 1.0)
        w_log_mag: Log magnitude loss weight (default: 1.0)
        w_lin_mag: Linear magnitude loss weight (default: 0.0)
        w_phs: Phase loss weight (default: 0.0)
        sample_rate: Sample rate for mel/chroma scaling
        scale: Frequency scaling ('mel', 'chroma', or None)
        n_bins: Number of frequency bins for scaling
        perceptual_weighting: Apply A-weighting (default: False)
        scale_invariance: Scale-invariant loss (default: False)
        eps: Numerical stability epsilon (default: 1e-8)
        output: Return format, 'loss' or 'full' (default: 'loss')
        reduction: Reduction method (default: 'mean')
        mag_distance: Distance for magnitude losses (default: 'L1')
        retain_batch_dim: Keep batch dimension in output (default: False)
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: Optional[float] = None,
        scale: Optional[str] = None,
        n_bins: Optional[int] = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        device: Any = None,
        retain_batch_dim: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = get_window(window, win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.retain_batch_dim = retain_batch_dim
        self.phs_used = bool(self.w_phs)

        # Loss components
        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction if not retain_batch_dim else "none",
            distance=mag_distance,
            **kwargs,
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction if not retain_batch_dim else "none",
            distance=mag_distance,
            **kwargs,
        )

        # Setup filterbank for mel/chroma scaling
        if scale is not None:
            self._setup_filterbank(scale, sample_rate, fft_size, n_bins, device)

        # Setup perceptual weighting
        if perceptual_weighting:
            if sample_rate is None:
                raise ValueError("sample_rate required for perceptual_weighting")
            from auraloss.perceptual import FIRFilter
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)

    def _setup_filterbank(
        self,
        scale: str,
        sample_rate: float,
        fft_size: int,
        n_bins: int,
        device: Any
    ):
        """Setup mel or chroma filterbank."""
        try:
            import librosa.filters
        except ImportError:
            raise ImportError("librosa required for mel/chroma scaling. "
                              "Install with: pip install librosa")

        if sample_rate is None:
            raise ValueError("sample_rate required for frequency scaling")
        if n_bins is not None and n_bins > fft_size:
            raise ValueError("n_bins must be <= fft_size")

        if scale == "mel":
            fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
        elif scale == "chroma":
            fb = librosa.filters.chroma(sr=sample_rate, n_fft=fft_size, n_chroma=n_bins)
        else:
            raise ValueError(f"Invalid scale: {scale}. Use 'mel' or 'chroma'.")

        fb = torch.tensor(fb).unsqueeze(0)
        self.register_buffer("fb", fb)

        if device is not None:
            self.fb = self.fb.to(device)

    def stft(self, x: torch.Tensor) -> tuple:
        """
        Compute STFT magnitude and optionally phase.

        Args:
            x: Input signal (B*C, T)

        Returns:
            Tuple of (magnitude, phase) spectrograms
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps)
        )
        x_phs = x_stft if self.phs_used else None
        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()

        # Apply perceptual weighting
        if self.perceptual_weighting:
            input = input.view(bs * chs, 1, -1)
            target = target.view(bs * chs, 1, -1)
            self.prefilter.to(input.device)
            input, target = self.prefilter(input, target)
            input = input.view(bs, chs, -1)
            target = target.view(bs, chs, -1)

        # Compute STFT
        self.window = self.window.to(input.device)
        x_mag, x_phs = self.stft(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft(target.view(-1, target.size(-1)))

        # Apply frequency scaling
        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # Scale invariance normalization
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # Compute loss components
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = normalized_complex_distance_loss(x_phs, y_phs) if self.phs_used else 0.0

        # Combine losses
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )
        loss = apply_reduction(loss, self.reduction, self.retain_batch_dim)

        if self.output == "loss":
            return loss
        else:
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
