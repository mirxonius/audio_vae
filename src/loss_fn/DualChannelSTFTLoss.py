"""
Mid-Side and Left-Right weighted STFT loss for stereo audio
"""

from typing import Tuple
import torch
import torch.nn as nn

from src.loss_fn.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss


def lr_to_ms(audio: torch.Tensor) -> torch.Tensor:
    """
    Convert Left-Right stereo to Mid-Side representation

    Args:
        audio: [B, 2, T] where channels are [L, R]
    Returns:
        ms_audio: [B, 2, T] where channels are [M, S]
    """
    left = audio[:, 0:1, :]
    right = audio[:, 1:2, :]

    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    return torch.cat([mid, side], dim=1)


def ms_to_lr(audio: torch.Tensor) -> torch.Tensor:
    """
    Convert Mid-Side to Left-Right stereo representation

    Args:
        audio: [B, 2, T] where channels are [M, S]
    Returns:
        lr_audio: [B, 2, T] where channels are [L, R]
    """
    mid = audio[:, 0:1, :]
    side = audio[:, 1:2, :]

    left = mid + side
    right = mid - side

    return torch.cat([left, right], dim=1)


class DualChannelSTFTLoss(nn.Module):
    """
    Weighted combination of M/S and L/R STFT losses for stereo audio

    The M/S representation helps with overall stereo coherence and image,
    while the L/R component prevents ambiguity in left-right placement.
    """

    def __init__(
        self,
        ms_weight: float = 1.0,
        lr_weight: float = 0.5,
        fft_sizes: list = [2048, 1024, 512, 256, 128, 64],
        hop_sizes: list = [512, 256, 128, 64, 32, 16],
        win_sizes: list = [2048, 1024, 512, 256, 128, 64],
    ):
        """
        Args:
            ms_weight: Weight for mid-side loss (default: 1.0)
            lr_weight: Weight for left-right loss (default: 0.5)
            fft_sizes: FFT sizes for multi-resolution STFT
            hop_sizes: Hop sizes for multi-resolution STFT
            win_sizes: Window sizes for multi-resolution STFT
        """
        super().__init__()
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_sizes=win_sizes,
        )
        self.ms_weight = ms_weight
        self.lr_weight = lr_weight

    def forward(
        self, pred_lr: torch.Tensor, target_lr: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_lr: [B, 2, T] predicted audio in L/R format
            target_lr: [B, 2, T] target audio in L/R format
        Returns:
            total_loss: Weighted combination of MS and LR losses
            loss_dict: Dictionary with individual loss components
        """
        # Compute L/R loss
        lr_sc_loss, lr_mag_loss = self.stft_loss(pred_lr, target_lr)
        lr_loss = lr_sc_loss + lr_mag_loss

        # Convert to M/S and compute M/S loss
        pred_ms = lr_to_ms(pred_lr)
        target_ms = lr_to_ms(target_lr)
        ms_sc_loss, ms_mag_loss = self.stft_loss(pred_ms, target_ms)
        ms_loss = ms_sc_loss + ms_mag_loss

        # Weighted combination
        total_loss = self.ms_weight * ms_loss + self.lr_weight * lr_loss

        loss_dict = {
            "ms_loss": ms_loss,
            "lr_loss": lr_loss,
            "ms_sc_loss": ms_sc_loss,
            "ms_mag_loss": ms_mag_loss,
            "lr_sc_loss": lr_sc_loss,
            "lr_mag_loss": lr_mag_loss,
        }

        return total_loss, loss_dict
