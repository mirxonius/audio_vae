# src/loss_fn/WaveformLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss_fn.BaseLoss import BaseLoss


class WaveformLoss(BaseLoss):
    """
    Simple L1 loss on waveforms.
    """

    def __init__(self, weight: float = 1.0, name: str = "waveform_loss"):
        super().__init__(weight=weight, name=name)

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_pred: [B, C, T] predicted waveform
            x_target: [B, C, T] target waveform
        """
        return F.l1_loss(x_pred, x_target)
