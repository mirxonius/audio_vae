# src/loss_fn/VAELossCalculator.py
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
from auraloss.perceptual import FIRFilter


from src.loss_fn.BaseLoss import BaseLoss
from src.loss_fn.MultiResolutionSTFTLoss import (
    MultiResolutionSTFTLoss,
    SumAndDifferenceSTFTLoss,
    LogSpectralDistanceLoss,
    SISDRLoss,
)
from src.loss_fn.MelSpectrogramLoss import MultiScaleMelSpectrogramLoss
from src.loss_fn.WaveformLoss import WaveformLoss


@dataclass
class VAELossOutput:
    """Container for VAE loss outputs."""

    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor
    kl_weight: float
    loss_breakdown: Dict[str, torch.Tensor]


class VAELossCalculator(nn.Module):
    """
    VAE loss calculator with explicit, optional reconstruction losses.

    Each loss is a separate argument, making it easy to:
    - Enable/disable losses via config (set to null)
    - Override individual loss params
    - See exactly what losses are active
    """

    def __init__(
        self,
        # KL configuration
        kl_weight: float = 1e-4,
        kl_warmup_steps: int = 0,
        free_bits: float = 0.0,
        # Individual reconstruction losses (all optional)
        stft_loss: Optional[MultiResolutionSTFTLoss] = None,
        sum_and_diff_loss: Optional[SumAndDifferenceSTFTLoss] = None,
        mel_loss: Optional[MultiScaleMelSpectrogramLoss] = None,
        waveform_loss: Optional[WaveformLoss] = None,
        si_sdr_loss: Optional[SISDRLoss] = None,
        # Extensibility: additional custom losses
        additional_losses: Optional[Dict[str, BaseLoss]] = None,
        perceptual_weighting: bool = False,
    ):
        super().__init__()

        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.free_bits = free_bits

        # Collect all non-None losses into ModuleDict for proper registration
        self.reconstruction_losses = nn.ModuleDict()

        if stft_loss is not None:
            self.reconstruction_losses["stft_loss"] = stft_loss
        if mel_loss is not None:
            self.reconstruction_losses["mel_loss"] = mel_loss
        if waveform_loss is not None:
            self.reconstruction_losses["waveform_loss"] = waveform_loss
        if sum_and_diff_loss is not None:
            self.reconstruction_losses["sum_and_diff_stft"] = sum_and_diff_loss
        if si_sdr_loss:
            self.reconstruction_losses["si_sdr_loss"] = si_sdr_loss
        # Add any additional custom losses
        if additional_losses is not None:
            for name, loss in additional_losses.items():
                self.reconstruction_losses[name] = loss

        if len(self.reconstruction_losses) == 0:
            raise ValueError("At least one reconstruction loss must be provided")

        # Track training step for KL warmup
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.preceptual_weighting = perceptual_weighting
        if self.preceptual_weighting:
            pass

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
            total_recon_loss = total_recon_loss + loss_value

            loss_breakdown[name] = loss_value

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
            f"  reconstruction_losses={losses}\n"
            f")"
        )
