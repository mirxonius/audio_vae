"""
VAE Lightning Module with optional adversarial training
Supports warmup period before enabling discriminator
"""

import os
from typing import Tuple
import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import tempfile
import torchaudio

from src.model.AudioVAE import AudioVAE
from src.model.discriminators.EncodecDiscriminator import EnCodecDiscriminator
from src.model.discriminators.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
)
from src.loss_fn.VAELossCalculator import VAELossCalculator


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
        adversarial_warmup_steps: int = 10000,
        adversarial_weight: float = 0.1,
        feature_matching_weight: float = 10.0,
        discriminator_lr: float = 1e-4,
        discriminator_update_every: int = 1,
        generator_update_every: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["architecture", "loss_calculator"])

        # Core components
        self.model = architecture
        self.loss_calculator = loss_calculator

        # Adversarial components
        self.use_adversarial = use_adversarial
        self.adversarial_warmup_steps = adversarial_warmup_steps
        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight
        self.discriminator_update_every = discriminator_update_every
        self.generator_update_every = generator_update_every

        if self.use_adversarial:
            self.discriminator = EnCodecDiscriminator()
            self.discriminator_lr = discriminator_lr
        else:
            self.discriminator = None

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
        if not self.use_adversarial:
            return False
        return self.global_step >= self.adversarial_warmup_steps

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

            # Adversarial loss (if warmup complete)
            if self.is_adversarial_active():
                # Get discriminator outputs for fake
                fake_logits, fake_fmaps = self.discriminator(recon)

                # Generator adversarial loss
                g_adv_loss = generator_adversarial_loss(fake_logits)

                # Feature matching loss
                with torch.no_grad():
                    real_logits, real_fmaps = self.discriminator(x)
                fm_loss = feature_matching_loss(fake_fmaps, real_fmaps)

                # Combined generator loss
                total_g_loss = (
                    vae_losses["total_loss"]
                    + self.adversarial_weight * g_adv_loss
                    + self.feature_matching_weight * fm_loss
                )

                # Log adversarial losses
                self.log("train/g_adv_loss", g_adv_loss, prog_bar=True)
                self.log("train/feature_matching_loss", fm_loss)
            else:
                total_g_loss = vae_losses["total_loss"]
                self.log(
                    "train/warmup_steps_remaining",
                    self.adversarial_warmup_steps - self.global_step,
                )

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

            # Discriminator outputs
            real_logits, fmaps = self.discriminator(x)
            fake_logits, fake_fmaps = self.discriminator(recon.detach())

            # Discriminator loss
            d_loss = discriminator_loss(real_logits, fake_logits)

            # Optimize discriminator
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.0
            )
            opt_d.step()

            # Log discriminator loss
            self.log("train/d_loss", d_loss, prog_bar=True)

            # Log discriminator accuracy
            with torch.no_grad():
                real_acc = sum(
                    (logits > 0).float().mean() for logits in real_logits
                ) / len(real_logits)
                fake_acc = sum(
                    (logits < 0).float().mean() for logits in fake_logits
                ) / len(fake_logits)
                self.log("train/d_real_acc", real_acc)
                self.log("train/d_fake_acc", fake_acc)

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
                real_logits, real_fmaps = self.discriminator(x)
                fake_logits, fake_fmaps = self.discriminator(recon)

                # Discriminator loss
                d_loss = discriminator_loss(real_logits, fake_logits)
                self.log("val/d_loss", d_loss, sync_dist=True)

                # Generator adversarial loss
                g_adv_loss = generator_adversarial_loss(fake_logits)
                self.log("val/g_adv_loss", g_adv_loss, sync_dist=True)

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
