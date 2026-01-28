# src/loss_fn/MelSpectrogramLoss.py
"""Mel spectrogram losses using auraloss."""

import torch
from typing import List, Optional
from auraloss.freq import MelSTFTLoss
from src.loss_fn.BaseLoss import BaseLoss, register_loss


@register_loss("multi_scale_mel_spectrogram")
class MultiScaleMelSpectrogramLoss(BaseLoss):
    """
    Multi-scale Mel Spectrogram Loss using auraloss.

    Computes mel spectrogram losses at multiple resolutions and averages them.
    Uses the MelSTFTLoss from auraloss library.

    Args:
        sample_rate: Audio sample rate
        fft_sizes: List of FFT sizes for each scale (default: [512, 1024, 2048])
        hop_sizes: List of hop sizes (default: fft_size // 4)
        win_lengths: List of window lengths (default: fft_sizes)
        n_mels: List of mel bin counts per scale (default: [64, 128, 256])
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier (default: 'multi_scale_mel_loss')
    """

    def __init__(
        self,
        sample_rate: int,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        n_mels: List[int] = [64, 128, 256],
        weight: float = 1.0,
        name: str = "multi_scale_mel_loss",
    ):
        super().__init__(weight, name)

        # Defaults: hop = fft//4, win = fft
        if hop_sizes is None:
            hop_sizes = [f // 4 for f in fft_sizes]
        if win_lengths is None:
            win_lengths = fft_sizes

        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths) == len(n_mels)):
            raise ValueError("All parameter lists must have the same length")

        self.losses = torch.nn.ModuleList([
            MelSTFTLoss(
                sample_rate=sample_rate,
                fft_size=fft,
                hop_size=hop,
                win_length=win,
                n_mels=mels,
                w_log_mag=1.0,
            )
            for fft, hop, win, mels in zip(fft_sizes, hop_sizes, win_lengths, n_mels)
        ])

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale mel spectrogram loss.

        Args:
            x_pred: Predicted audio (B, C, T)
            x_target: Target audio (B, C, T)

        Returns:
            Averaged loss across all scales
        """
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss = total_loss + loss_fn(x_pred, x_target)
        return total_loss / len(self.losses)
