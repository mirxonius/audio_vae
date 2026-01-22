"""
Adversarial Loss Calculator for Audio VAE

Encapsulates all adversarial training loss computation including:
- Discriminator loss (real vs fake)
- Generator adversarial loss
- Feature matching loss
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from src.model.discriminators.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
)


class AdversarialLossCalculator(nn.Module):
    """
    Calculates all adversarial losses for GAN training.

    Supports:
    - Discriminator training (hinge loss)
    - Generator adversarial loss
    - Feature matching loss
    - Warmup scheduling for gradual adversarial training introduction
    """

    def __init__(
        self,
        adversarial_weight: float = 1.0,
        feature_matching_weight: float = 10.0,
        warmup_steps: int = 0,
    ):
        """
        Args:
            adversarial_weight: Weight for generator adversarial loss
            feature_matching_weight: Weight for feature matching loss
            warmup_steps: Number of steps before enabling adversarial training
        """
        super().__init__()
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight
        self.warmup_steps = warmup_steps

    def is_active(self, global_step: int) -> bool:
        """Check if adversarial training should be active."""
        return global_step >= self.warmup_steps

    def compute_discriminator_loss(
        self,
        discriminator: nn.Module,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss.

        Args:
            discriminator: The discriminator model
            real_audio: Real audio samples [B, C, T]
            fake_audio: Generated/fake audio samples [B, C, T] (should be detached)

        Returns:
            Dictionary with:
                - d_loss: Total discriminator loss
                - d_real_acc: Accuracy on real samples (percentage classified as real)
                - d_fake_acc: Accuracy on fake samples (percentage classified as fake)
        """
        # Get discriminator outputs
        real_logits, _ = discriminator(real_audio)
        fake_logits, _ = discriminator(fake_audio.detach())

        # Compute discriminator loss
        d_loss = discriminator_loss(real_logits, fake_logits)

        # Compute accuracies for monitoring
        with torch.no_grad():
            real_acc = sum(
                (logits > 0).float().mean() for logits in real_logits
            ) / len(real_logits)
            fake_acc = sum(
                (logits < 0).float().mean() for logits in fake_logits
            ) / len(fake_logits)

        return {
            "d_loss": d_loss,
            "d_real_acc": real_acc,
            "d_fake_acc": fake_acc,
        }

    def compute_generator_loss(
        self,
        discriminator: nn.Module,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
        global_step: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator adversarial and feature matching losses.

        Args:
            discriminator: The discriminator model
            real_audio: Real audio samples [B, C, T]
            fake_audio: Generated audio samples [B, C, T] (with gradients)
            global_step: Current training step (for warmup)

        Returns:
            Dictionary with:
                - g_adv_loss: Generator adversarial loss (weighted)
                - fm_loss: Feature matching loss (weighted)
                - total_adv_loss: Sum of weighted adversarial losses
                - is_active: Whether adversarial training is active
        """
        if not self.is_active(global_step):
            # Return zeros if warmup not complete
            return {
                "g_adv_loss": torch.tensor(0.0, device=fake_audio.device),
                "fm_loss": torch.tensor(0.0, device=fake_audio.device),
                "total_adv_loss": torch.tensor(0.0, device=fake_audio.device),
                "is_active": False,
            }

        # Get discriminator outputs for fake
        fake_logits, fake_fmaps = discriminator(fake_audio)

        # Generator adversarial loss
        g_adv_loss = generator_adversarial_loss(fake_logits)

        # Feature matching loss
        with torch.no_grad():
            _, real_fmaps = discriminator(real_audio)
        fm_loss = feature_matching_loss(fake_fmaps, real_fmaps)

        # Weighted total
        total_adv_loss = (
            self.adversarial_weight * g_adv_loss
            + self.feature_matching_weight * fm_loss
        )

        return {
            "g_adv_loss": g_adv_loss,
            "fm_loss": fm_loss,
            "total_adv_loss": total_adv_loss,
            "is_active": True,
        }

    def forward(
        self,
        discriminator: nn.Module,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
        global_step: int,
        mode: str = "generator",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses based on mode.

        Args:
            discriminator: The discriminator model
            real_audio: Real audio samples [B, C, T]
            fake_audio: Generated audio samples [B, C, T]
            global_step: Current training step
            mode: Either "generator" or "discriminator"

        Returns:
            Dictionary of losses
        """
        if mode == "generator":
            return self.compute_generator_loss(
                discriminator, real_audio, fake_audio, global_step
            )
        elif mode == "discriminator":
            return self.compute_discriminator_loss(
                discriminator, real_audio, fake_audio
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'generator' or 'discriminator'")
