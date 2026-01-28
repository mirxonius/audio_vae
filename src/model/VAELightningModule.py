"""
VAE Lightning Module with optional adversarial training
Supports warmup period before enabling discriminator

Refactored to use:
- Configurable discriminator via Hydra
- AdversarialLossCalculator for cleaner loss computation
- Improved modularity and maintainability
"""

import os
from typing import Tuple, Optional
import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import tempfile
import torchaudio

from src.model.AudioVAE import AudioVAE
from src.loss_fn.VAELossCalculator import VAELossCalculator
from src.loss_fn.AdversarialLossCalculator import AdversarialLossCalculator
from src.utils.model_io import save_model


class VAELightningModule(pl.LightningModule):
    def __init__(
        self,
        architecture: AudioVAE,
        loss_calculator: VAELossCalculator,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        log_audio_every_n_epochs: int = 5,
        sample_rate: int = 44100,
        num_audio_samples: int = 2,
        # Adversarial training parameters
        use_adversarial: bool = False,
        discriminator: Optional[torch.nn.Module] = None,
        adversarial_loss_calculator: Optional[AdversarialLossCalculator] = None,
        discriminator_lr: float = 1e-4,
        discriminator_update_every: int = 1,
        generator_update_every: int = 1,
        # Legacy parameters (for backward compatibility)
        adversarial_warmup_steps: Optional[int] = None,
        adversarial_weight: Optional[float] = None,
        feature_matching_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "architecture",
                "loss_calculator",
                "discriminator",
                "adversarial_loss_calculator",
            ]
        )

        # Core components
        self.model = architecture
        self.loss_calculator = loss_calculator

        # Adversarial components
        self.use_adversarial = use_adversarial
        self.discriminator_update_every = discriminator_update_every
        self.generator_update_every = generator_update_every

        if self.use_adversarial:
            # Discriminator
            if discriminator is None:
                raise ValueError(
                    "When use_adversarial=True, discriminator must be provided. "
                    "Add discriminator config to your model config."
                )
            self.discriminator = discriminator
            self.discriminator_lr = discriminator_lr

            # Adversarial loss calculator
            if adversarial_loss_calculator is None:
                # Backward compatibility: create from legacy parameters
                if (
                    adversarial_warmup_steps is not None
                    or adversarial_weight is not None
                ):
                    import warnings

                    warnings.warn(
                        "Using legacy adversarial parameters. "
                        "Please use adversarial_loss_calculator config instead.",
                        DeprecationWarning,
                    )
                    self.adversarial_loss_calculator = AdversarialLossCalculator(
                        adversarial_weight=adversarial_weight or 1.0,
                        feature_matching_weight=feature_matching_weight or 10.0,
                        warmup_steps=adversarial_warmup_steps or 0,
                    )
                else:
                    raise ValueError(
                        "When use_adversarial=True, adversarial_loss_calculator must be provided. "
                        "Add adversarial_loss_calculator config to your model config."
                    )
            else:
                self.adversarial_loss_calculator = adversarial_loss_calculator
        else:
            self.discriminator = None
            self.adversarial_loss_calculator = None

        # Optimizer/scheduler configs (partial, will be completed in configure_optimizers)
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        # Logging
        self.log_audio_every_n_epochs = log_audio_every_n_epochs
        self.sample_rate = sample_rate
        self.num_audio_samples = num_audio_samples

        # Automatic optimization disabled for adversarial training
        if self.use_adversarial:
            self.automatic_optimization = False

    def is_adversarial_active(self) -> bool:
        """Check if adversarial training should be active based on global_step"""
        if not self.use_adversarial or self.adversarial_loss_calculator is None:
            return False
        return self.adversarial_loss_calculator.is_active(self.global_step)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if self.use_adversarial:
            return self._adversarial_training_step(batch, batch_idx)
        else:
            return self._standard_training_step(batch, batch_idx)

    def _standard_training_step(self, x: torch.Tensor, batch_idx: int):
        """Standard VAE training without adversarial component"""

        # Forward pass
        recon, z, mean, logvar = self.forward(x)

        # Calculate losses
        losses = self.loss_calculator(recon, x, mean, logvar, self.global_step)

        # Log losses
        self.log("train/total_loss", losses["total_loss"], prog_bar=True)
        for loss_name, loss_value in losses.items():
            if loss_name != "total_loss":
                self.log(f"train/{loss_name}", loss_value)

        return losses["total_loss"]

    def _adversarial_training_step(self, x: torch.Tensor, batch_idx: int):
        """Training step with adversarial component and warmup"""
        opt_g, opt_d = self.optimizers()

        # =================
        # Train Generator (VAE)
        # =================
        if batch_idx % self.generator_update_every == 0:
            # Forward pass
            recon, z, mean, logvar = self.forward(x)

            # VAE reconstruction losses
            vae_losses = self.loss_calculator(recon, x, mean, logvar, self.global_step)

            # Adversarial losses (handled by AdversarialLossCalculator)
            adv_losses = self.adversarial_loss_calculator.compute_generator_loss(
                discriminator=self.discriminator,
                real_audio=x,
                fake_audio=recon,
                global_step=self.global_step,
            )

            # Combined generator loss
            total_g_loss = vae_losses["total_loss"] + adv_losses["total_adv_loss"]

            # Optimize generator
            opt_g.zero_grad()
            self.manual_backward(total_g_loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt_g.step()

            # Log generator losses
            self.log("train/total_g_loss", total_g_loss, prog_bar=True)
            for loss_name, loss_value in vae_losses.items():
                if loss_name != "total_loss":
                    self.log(f"train/{loss_name}", loss_value)

            # Log adversarial losses
            if adv_losses["is_active"]:
                self.log("train/g_adv_loss", adv_losses["g_adv_loss"], prog_bar=True)
                self.log("train/feature_matching_loss", adv_losses["fm_loss"])
            else:
                warmup_remaining = (
                    self.adversarial_loss_calculator.warmup_steps - self.global_step
                )
                self.log("train/warmup_steps_remaining", warmup_remaining)

        # =================
        # Train Discriminator
        # =================
        if (
            self.is_adversarial_active()
            and batch_idx % self.discriminator_update_every == 0
        ):
            # Get fresh outputs (detach generator)
            with torch.no_grad():
                recon, _, _, _ = self.forward(x)

            # Discriminator losses (handled by AdversarialLossCalculator)
            disc_losses = self.adversarial_loss_calculator.compute_discriminator_loss(
                discriminator=self.discriminator,
                real_audio=x,
                fake_audio=recon,
            )

            # Optimize discriminator
            opt_d.zero_grad()
            self.manual_backward(disc_losses["d_loss"])
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.0
            )
            opt_d.step()

            # Log discriminator metrics
            self.log("train/d_loss", disc_losses["d_loss"], prog_bar=True)
            self.log("train/d_real_acc", disc_losses["d_real_acc"])
            self.log("train/d_fake_acc", disc_losses["d_fake_acc"])

        # Step schedulers
        sch_g, sch_d = self.lr_schedulers()
        if self.trainer.is_last_batch and self.scheduler_interval == "epoch":
            sch_g.step()
            if self.is_adversarial_active():
                sch_d.step()
        elif self.scheduler_interval == "step":
            sch_g.step()
            if self.is_adversarial_active():
                sch_d.step()

    def validation_step(self, x: torch.Tensor, batch_idx: int):

        # Forward pass
        recon, z, mean, logvar = self.forward(x)

        # Calculate losses
        losses = self.loss_calculator(recon, x, mean, logvar, self.global_step)

        # Log losses
        self.log("val/total_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        for loss_name, loss_value in losses.items():
            if loss_name != "total_loss":
                self.log(f"val/{loss_name}", loss_value, sync_dist=True)

        # Adversarial validation metrics (if active)
        if self.is_adversarial_active():
            with torch.no_grad():
                # Generator adversarial losses
                adv_losses = self.adversarial_loss_calculator.compute_generator_loss(
                    discriminator=self.discriminator,
                    real_audio=x,
                    fake_audio=recon,
                    global_step=self.global_step,
                )
                self.log("val/g_adv_loss", adv_losses["g_adv_loss"], sync_dist=True)
                self.log("val/fm_loss", adv_losses["fm_loss"], sync_dist=True)

                # Discriminator loss
                disc_losses = (
                    self.adversarial_loss_calculator.compute_discriminator_loss(
                        discriminator=self.discriminator,
                        real_audio=x,
                        fake_audio=recon,
                    )
                )
                self.log("val/d_loss", disc_losses["d_loss"], sync_dist=True)

        # Log audio samples periodically
        if self.current_epoch % self.log_audio_every_n_epochs == 0 and batch_idx == 0:
            self._log_audio_samples(x, recon)

        return losses["total_loss"]

    def _log_audio_samples(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """Log audio samples - supports both MLflow and TensorBoard loggers"""

        num_samples = min(self.num_audio_samples, original.size(0))

        # MLflow: Save as artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(num_samples):
                # Save original
                original_path = os.path.join(
                    temp_dir, f"epoch_{self.current_epoch}_original_{i}.wav"
                )
                torchaudio.save(
                    original_path,
                    original[i].float().cpu(),
                    self.sample_rate,
                )

                # Save reconstruction
                recon_path = os.path.join(
                    temp_dir, f"epoch_{self.current_epoch}_reconstruction_{i}.wav"
                )
                torchaudio.save(
                    recon_path,
                    reconstructed[i].float().cpu(),
                    self.sample_rate,
                )

                # Log to MLflow as artifacts
                self.logger.experiment.log_artifact(
                    self.logger.run_id,
                    original_path,
                    artifact_path=f"audio_samples/epoch_{self.current_epoch}",
                )
                self.logger.experiment.log_artifact(
                    self.logger.run_id,
                    recon_path,
                    artifact_path=f"audio_samples/epoch_{self.current_epoch}",
                )

    def on_save_checkpoint(self, checkpoint):
        """
        Called when saving a checkpoint.
        Adds standalone AudioVAE model checkpoint for easy loading.
        """
        # Get optimizer and scheduler states if available
        optimizer_state = None
        scheduler_state = None

        if hasattr(self, "optimizers"):
            optimizers = self.optimizers()
            if optimizers is not None:
                if isinstance(optimizers, list):
                    optimizer_state = optimizers[0].state_dict()
                else:
                    optimizer_state = optimizers.state_dict()

        if hasattr(self, "lr_schedulers"):
            schedulers = self.lr_schedulers()
            if schedulers is not None:
                if isinstance(schedulers, list):
                    scheduler_state = schedulers[0].state_dict()
                else:
                    scheduler_state = schedulers.state_dict()

        # Store model config for easy reconstruction
        checkpoint["model_config"] = {
            "in_channels": self.model.encoder.in_channels,
            "base_channels": self.model.encoder.base_channels,
            "channel_mults": self.model.encoder.channel_mults,
            "strides": self.model.encoder.strides,
            "latent_dim": self.model.latent_dim,
            "kernel_size": self.model.encoder.kernel_size,
            "dilations": self.model.encoder.dilations,
        }

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        if not self.use_adversarial:
            # Standard single optimizer setup
            optimizer = self.optimizer_config(params=self.model.parameters())

            # Calculate total steps for scheduler
            if hasattr(self.trainer, "estimated_stepping_batches"):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = (
                    len(self.trainer.datamodule.train_dataloader())
                    * self.trainer.max_epochs
                )

            # Complete scheduler config with total steps
            if hasattr(self.scheduler_config.func, "__name__"):
                if "CosineAnnealing" in self.scheduler_config.func.__name__:
                    scheduler = self.scheduler_config(
                        optimizer=optimizer, T_max=total_steps
                    )
                else:
                    scheduler = self.scheduler_config(optimizer=optimizer)
            else:
                scheduler = self.scheduler_config(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_interval,
                    "frequency": self.scheduler_frequency,
                },
            }
        else:
            # Adversarial training: two optimizers
            opt_g = self.optimizer_config(params=self.model.parameters())
            opt_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.discriminator_lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
            )

            # Calculate total steps for schedulers
            if hasattr(self.trainer, "estimated_stepping_batches"):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = (
                    len(self.trainer.datamodule.train_dataloader())
                    * self.trainer.max_epochs
                )

            # Generator scheduler
            if hasattr(self.scheduler_config.func, "__name__"):
                if "CosineAnnealing" in self.scheduler_config.func.__name__:
                    sch_g = self.scheduler_config(optimizer=opt_g, T_max=total_steps)
                else:
                    sch_g = self.scheduler_config(optimizer=opt_g)
            else:
                sch_g = self.scheduler_config(optimizer=opt_g)

            # Discriminator scheduler (same as generator)
            sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_d, T_max=total_steps, eta_min=1e-6
            )

            return [opt_g, opt_d], [
                {
                    "scheduler": sch_g,
                    "interval": self.scheduler_interval,
                    "frequency": self.scheduler_frequency,
                },
                {
                    "scheduler": sch_d,
                    "interval": self.scheduler_interval,
                    "frequency": self.scheduler_frequency,
                },
            ]
