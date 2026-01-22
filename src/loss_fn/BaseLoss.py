# src/loss_fn/base.py
import torch.nn as nn
from abc import ABC, abstractmethod
import torch


class BaseLoss(nn.Module, ABC):
    """Base class for reconstruction losses."""

    def __init__(self, weight: float = 1.0, name: str = "loss"):
        super().__init__()
        self.weight = weight
        self.name = name

    @abstractmethod
    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute unweighted loss. Weighting is handled by VAELossCalculator."""
        pass
