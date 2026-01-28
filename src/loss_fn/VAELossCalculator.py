# src/loss_fn/VAELossCalculator.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass


@dataclass
class VAELossOutput:
    """Container for VAE loss outputs."""

    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor
    kl_weight: float
    loss_breakdown: Dict[str, torch.Tensor]


# Loss registry mapping loss names to classes
_LOSS_REGISTRY: Dict[str, type] = {}


def _populate_loss_registry():
    """Lazily populate the loss registry to avoid circular imports."""
    global _LOSS_REGISTRY
    if _LOSS_REGISTRY:
        return _LOSS_REGISTRY

    from src.loss_fn.MultiResolutionSTFTLoss import (
        MultiResolutionSTFTLoss,
        SumAndDifferenceSTFTLoss,
        MultiScaleMelLoss,
        LogSpectralDistanceLoss,
    )
    from src.loss_fn.MelSpectrogramLoss import MultiScaleMelSpectrogramLoss
    from src.loss_fn.WaveformLoss import WaveformLoss

    _LOSS_REGISTRY = {
        "multi_resolution_stft": MultiResolutionSTFTLoss,
        "sum_and_difference_stft": SumAndDifferenceSTFTLoss,
        "multi_scale_mel": MultiScaleMelLoss,
        "multi_scale_mel_spectrogram": MultiScaleMelSpectrogramLoss,
        "log_spectral_distance": LogSpectralDistanceLoss,
        "waveform": WaveformLoss,
    }
    return _LOSS_REGISTRY


def resolve_loss(loss_config: Union[nn.Module, Dict[str, Any], str]) -> nn.Module:
    """
    Resolve a loss from various input formats.

    Args:
        loss_config: Either:
            - An already instantiated nn.Module
            - A dict with 'type' key and optional kwargs
            - A string naming a registered loss type

    Returns:
        Instantiated loss module
    """
    if isinstance(loss_config, nn.Module):
        return loss_config

    registry = _populate_loss_registry()

    if isinstance(loss_config, str):
        if loss_config not in registry:
            raise ValueError(
                f"Unknown loss type: {loss_config}. Available: {list(registry.keys())}"
            )
        return registry[loss_config]()

    if isinstance(loss_config, dict):
        config = loss_config.copy()
        loss_type = config.pop("type", None)
        if loss_type is None:
            raise ValueError("Loss config dict must have 'type' key")
        if loss_type not in registry:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Available: {list(registry.keys())}"
            )
        return registry[loss_type](**config)

    raise TypeError(f"Cannot resolve loss from type: {type(loss_config)}")


class _LRMultiResolutionWrapper(nn.Module):
    """
    Wrapper that applies a multiresolution loss separately to left and right channels.

    For stereo audio [B, 2, T], computes the loss on:
    - Left channel [B, 1, T]
    - Right channel [B, 1, T]

    And returns the average of both.
    """

    def __init__(self, base_loss: nn.Module, weight: float = 1.0):
        super().__init__()
        self.base_loss = base_loss
        self.weight = weight

    def forward(
        self, x_pred: torch.Tensor, x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_pred: Predicted stereo audio [B, 2, T]
            x_target: Target stereo audio [B, 2, T]

        Returns:
            Average loss over left and right channels
        """
        if x_pred.size(1) != 2:
            raise ValueError(
                f"_LRMultiResolutionWrapper expects stereo input (2 channels), "
                f"got {x_pred.size(1)} channels"
            )

        # Extract left and right channels
        left_pred = x_pred[:, 0:1, :]  # [B, 1, T]
        right_pred = x_pred[:, 1:2, :]  # [B, 1, T]
        left_target = x_target[:, 0:1, :]  # [B, 1, T]
        right_target = x_target[:, 1:2, :]  # [B, 1, T]

        # Compute loss on each channel
        left_loss = self.base_loss(left_pred, left_target)
        right_loss = self.base_loss(right_pred, right_target)

        # Return average
        return (left_loss + right_loss) / 2.0


class VAELossCalculator(nn.Module):
    """
    VAE loss calculator with flexible reconstruction losses.

    Supports:
    - Direct loss module injection (for Hydra/config-based instantiation)
    - Loss name resolution from registry
    - Stereo-specific losses when num_channels=2:
      - Sum-and-difference STFT loss (mid/side decomposition)
      - Separate left/right channel multiresolution STFT loss

    Each loss can be:
    - An instantiated nn.Module
    - A dict with 'type' and kwargs for resolution
    - A string naming a registered loss
    """

    def __init__(
        self,
        # KL configuration
        kl_weight: float = 1e-4,
        kl_warmup_steps: int = 0,
        free_bits: float = 0.0,
        # Audio configuration
        num_channels: int = 1,
        # Individual reconstruction losses (all optional)
        # Can be nn.Module, dict with 'type' key, or None
        stft_loss: Optional[Union[nn.Module, Dict[str, Any]]] = None,
        mel_loss: Optional[Union[nn.Module, Dict[str, Any]]] = None,
        waveform_loss: Optional[Union[nn.Module, Dict[str, Any]]] = None,
        # Stereo-specific losses (only used when num_channels=2)
        use_sum_and_difference: bool = False,
        sum_and_difference_loss: Optional[Union[nn.Module, Dict[str, Any]]] = None,
        use_lr_multiresolution: bool = False,
        lr_multiresolution_loss: Optional[Union[nn.Module, Dict[str, Any]]] = None,
        # Extensibility: additional custom losses
        additional_losses: Optional[Dict[str, Union[nn.Module, Dict[str, Any]]]] = None,
        # Loss weights (for losses that don't have internal weights)
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.free_bits = free_bits
        self.num_channels = num_channels

        # Store loss weights
        self.loss_weights = loss_weights or {}

        # Collect all non-None losses into ModuleDict for proper registration
        self.reconstruction_losses = nn.ModuleDict()

        # Add standard losses
        if stft_loss is not None:
            self.reconstruction_losses["stft_loss"] = resolve_loss(stft_loss)
        if mel_loss is not None:
            self.reconstruction_losses["mel_loss"] = resolve_loss(mel_loss)
        if waveform_loss is not None:
            self.reconstruction_losses["waveform_loss"] = resolve_loss(waveform_loss)

        # Add stereo-specific losses when num_channels == 2
        if num_channels == 2:
            self._setup_stereo_losses(
                use_sum_and_difference=use_sum_and_difference,
                sum_and_difference_loss=sum_and_difference_loss,
                use_lr_multiresolution=use_lr_multiresolution,
                lr_multiresolution_loss=lr_multiresolution_loss,
            )

        # Add any additional custom losses
        if additional_losses is not None:
            for name, loss in additional_losses.items():
                self.reconstruction_losses[name] = resolve_loss(loss)

        if len(self.reconstruction_losses) == 0:
            raise ValueError("At least one reconstruction loss must be provided")

        # Track training step for KL warmup
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    def _setup_stereo_losses(
        self,
        use_sum_and_difference: bool,
        sum_and_difference_loss: Optional[Union[nn.Module, Dict[str, Any]]],
        use_lr_multiresolution: bool,
        lr_multiresolution_loss: Optional[Union[nn.Module, Dict[str, Any]]],
    ):
        """Setup stereo-specific losses for 2-channel audio."""
        from src.loss_fn.MultiResolutionSTFTLoss import (
            SumAndDifferenceSTFTLoss,
            MultiResolutionSTFTLoss,
        )

        # Sum and difference (mid/side) loss
        if use_sum_and_difference:
            if sum_and_difference_loss is not None:
                self.reconstruction_losses["sum_diff_loss"] = resolve_loss(
                    sum_and_difference_loss
                )
            else:
                # Default sum-and-difference loss configuration
                self.reconstruction_losses["sum_diff_loss"] = SumAndDifferenceSTFTLoss(
                    fft_sizes=[2048, 1024, 512, 256],
                    hop_sizes=[512, 256, 128, 64],
                    win_lengths=[2048, 1024, 512, 256],
                )

        # Left-Right multiresolution loss (applies STFT loss separately to each channel)
        if use_lr_multiresolution:
            if lr_multiresolution_loss is not None:
                self._lr_loss_module = resolve_loss(lr_multiresolution_loss)
            else:
                # Default LR multiresolution loss configuration
                self._lr_loss_module = MultiResolutionSTFTLoss(
                    fft_sizes=[2048, 1024, 512, 256],
                    hop_sizes=[512, 256, 128, 64],
                    win_lengths=[2048, 1024, 512, 256],
                )
            # Register module but handle forward separately
            self.reconstruction_losses["lr_stft_loss"] = _LRMultiResolutionWrapper(
                self._lr_loss_module
            )

    def _get_loss_weight(self, name: str, loss_fn: nn.Module) -> float:
        """
        Get weight for a loss function.

        Priority:
        1. Explicit weight in self.loss_weights dict
        2. loss_fn.weight attribute (if exists, for backward compatibility)
        3. Default to 1.0
        """
        if name in self.loss_weights:
            return self.loss_weights[name]
        if hasattr(loss_fn, "weight"):
            return loss_fn.weight
        return 1.0

    def get_kl_weight(self) -> float:
        """Get current KL weight with linear warmup."""
        if self.kl_warmup_steps <= 0:
            return self.kl_weight

        progress = min(1.0, self.global_step.item() / self.kl_warmup_steps)
        return self.kl_weight * progress

    def kl_divergence(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I).

        With optional free bits to prevent posterior collapse.
        """
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over latent dims, mean over batch
        return kl_per_dim.sum(dim=list(range(1, kl_per_dim.ndim))).mean()

    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        step: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total VAE loss.

        Args:
            x_recon: Reconstructed waveform [B, C, T]
            x_target: Target waveform [B, C, T]
            mu: Latent mean [B, latent_dim, T']
            log_var: Latent log variance [B, latent_dim, T']
            step: Optional global step for KL warmup
        """
        if step is not None:
            self.global_step.fill_(step)

        loss_breakdown = {}

        # Compute reconstruction losses
        total_recon_loss = torch.tensor(0.0, device=x_recon.device)

        for name, loss_fn in self.reconstruction_losses.items():
            loss_value = loss_fn(x_recon, x_target)
            weight = self._get_loss_weight(name, loss_fn)
            weighted_loss = loss_value * weight
            total_recon_loss = total_recon_loss + weighted_loss

            loss_breakdown[name] = loss_value
            loss_breakdown[f"{name}_weighted"] = weighted_loss

        # Compute KL loss
        kl_loss = self.kl_divergence(mu, log_var)
        current_kl_weight = self.get_kl_weight()
        weighted_kl = kl_loss * current_kl_weight

        loss_breakdown["kl_raw"] = kl_loss
        loss_breakdown["kl_weighted"] = weighted_kl
        loss_breakdown["kl_weight"] = torch.tensor(current_kl_weight)

        total_loss = total_recon_loss + weighted_kl

        return dict(
            total_loss=total_loss,
            reconstruction_loss=total_recon_loss,
            kl_loss=kl_loss,
            kl_weight=current_kl_weight,
        )

    def __repr__(self) -> str:
        losses = list(self.reconstruction_losses.keys())
        return (
            f"{self.__class__.__name__}(\n"
            f"  kl_weight={self.kl_weight},\n"
            f"  kl_warmup_steps={self.kl_warmup_steps},\n"
            f"  free_bits={self.free_bits},\n"
            f"  num_channels={self.num_channels},\n"
            f"  reconstruction_losses={losses},\n"
            f"  loss_weights={self.loss_weights}\n"
            f")"
        )
