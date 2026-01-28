# src/loss_fn/BaseLoss.py
"""Base loss class and loss registry for audio VAE losses."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any

# Global loss registry
_LOSS_REGISTRY: Dict[str, Type["BaseLoss"]] = {}


def register_loss(name: Optional[str] = None):
    """
    Decorator to register a loss class in the global registry.

    Usage:
        @register_loss("my_loss")
        class MyLoss(BaseLoss):
            ...

    Or without explicit name (uses class name):
        @register_loss()
        class MyLoss(BaseLoss):
            ...
    """
    def decorator(cls: Type["BaseLoss"]) -> Type["BaseLoss"]:
        loss_name = name if name is not None else cls.__name__
        if loss_name in _LOSS_REGISTRY:
            raise ValueError(f"Loss '{loss_name}' is already registered")
        _LOSS_REGISTRY[loss_name] = cls
        return cls
    return decorator


def get_loss(name: str, **kwargs) -> "BaseLoss":
    """
    Instantiate a loss by name from the registry.

    Args:
        name: Registered loss name
        **kwargs: Arguments to pass to the loss constructor

    Returns:
        Instantiated loss module
    """
    if name not in _LOSS_REGISTRY:
        available = list(_LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return _LOSS_REGISTRY[name](**kwargs)


def list_losses() -> list:
    """Return list of all registered loss names."""
    return list(_LOSS_REGISTRY.keys())


class BaseLoss(nn.Module, ABC):
    """
    Base class for all reconstruction losses.

    All losses should inherit from this class and implement the forward method.
    The `weight` parameter is used by the VAELossCalculator to scale the loss.

    Args:
        weight: Scaling factor for this loss component (default: 1.0)
        name: Identifier for this loss instance (default: class name in snake_case)
    """

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        super().__init__()
        self.weight = weight
        self.name = name if name is not None else self._default_name()

    def _default_name(self) -> str:
        """Convert class name to snake_case for default naming."""
        class_name = self.__class__.__name__
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    @abstractmethod
    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute the unweighted loss.

        Weighting is handled by VAELossCalculator - this method should return
        the raw loss value.

        Args:
            x_pred: Predicted/reconstructed audio [B, C, T]
            x_target: Target audio [B, C, T]

        Returns:
            Scalar loss tensor
        """
        pass

    def extra_repr(self) -> str:
        return f"weight={self.weight}, name='{self.name}'"
