# src/loss_fn/WaveformLoss.py
"""Simple waveform-domain losses."""

import torch
import torch.nn.functional as F
from src.loss_fn.BaseLoss import BaseLoss, register_loss


@register_loss("waveform")
class WaveformLoss(BaseLoss):
    """
    L1 loss on raw waveforms.

    Simple time-domain loss that measures absolute differences between
    predicted and target waveforms.

    Args:
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier (default: 'waveform_loss')
    """

    def __init__(self, weight: float = 1.0, name: str = "waveform_loss"):
        super().__init__(weight=weight, name=name)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 waveform loss.

        Args:
            x_pred: Predicted waveform (B, C, T)
            x_target: Target waveform (B, C, T)

        Returns:
            Scalar L1 loss
        """
        return F.l1_loss(x_pred, x_target)


@register_loss("waveform_l2")
class WaveformL2Loss(BaseLoss):
    """
    L2 (MSE) loss on raw waveforms.

    Time-domain loss that measures squared differences between
    predicted and target waveforms.

    Args:
        weight: Loss weighting factor (default: 1.0)
        name: Loss name identifier (default: 'waveform_l2_loss')
    """

    def __init__(self, weight: float = 1.0, name: str = "waveform_l2_loss"):
        super().__init__(weight=weight, name=name)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 (MSE) waveform loss.

        Args:
            x_pred: Predicted waveform (B, C, T)
            x_target: Target waveform (B, C, T)

        Returns:
            Scalar MSE loss
        """
        return F.mse_loss(x_pred, x_target)
