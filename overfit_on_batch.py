"""
Overfit on a single batch - debugging script to verify model can learn.

This script trains the model on a single fixed batch of data to verify that
the model architecture and training loop can successfully minimize the loss.
If the model cannot overfit to a single batch, there's likely a bug in the
implementation.

Usage:
    # Basic usage with defaults
    python overfit_on_batch.py

    # Override model/training settings
    python overfit_on_batch.py model.architecture.latent_dim=64

    # Use standard VAE (no discriminator)
    python overfit_on_batch.py model.use_adversarial=false

    # Change learning rate
    python overfit_on_batch.py model.optimizer.lr=1e-4

    # Change batch size and iterations
    python overfit_on_batch.py overfit.batch_size=4 overfit.num_iterations=2000
"""

import logging
from typing import Dict, Any, Optional

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

log = logging.getLogger(__name__)


class OverfitTrainer:
    """Trainer for overfitting on a single batch."""

    def __init__(
        self,
        model: pl.LightningModule,
        batch: torch.Tensor,
        device: str = "cuda",
        log_every: int = 100,
    ):
        self.model = model.to(device)
        self.batch = batch.to(device)
        self.device = device
        self.log_every = log_every
        self.global_step = 0

        # Setup optimizers
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Configure optimizers based on training mode."""
        if self.model.use_adversarial:
            # Adversarial training: two optimizers
            self.opt_g = self.model.optimizer_config(params=self.model.model.parameters())
            self.opt_d = torch.optim.AdamW(
                self.model.discriminator.parameters(),
                lr=self.model.discriminator_lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
            )
            self.optimizers = [self.opt_g, self.opt_d]
        else:
            # Standard VAE: single optimizer
            self.opt_g = self.model.optimizer_config(params=self.model.model.parameters())
            self.optimizers = [self.opt_g]

    def _standard_step(self) -> Dict[str, float]:
        """Standard VAE training step."""
        self.opt_g.zero_grad()

        # Forward pass
        recon, z, mean, logvar = self.model(self.batch)

        # Calculate losses
        losses = self.model.loss_calculator(
            recon, self.batch, mean, logvar, self.global_step
        )

        # Backward pass
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        self.opt_g.step()

        return {k: v.item() for k, v in losses.items()}

    def _adversarial_step(self) -> Dict[str, float]:
        """Adversarial training step."""
        losses = {}

        # =================
        # Train Generator (VAE)
        # =================
        self.opt_g.zero_grad()

        # Forward pass
        recon, z, mean, logvar = self.model(self.batch)

        # VAE reconstruction losses
        vae_losses = self.model.loss_calculator(
            recon, self.batch, mean, logvar, self.global_step
        )

        # Adversarial losses
        adv_losses = self.model.adversarial_loss_calculator.compute_generator_loss(
            discriminator=self.model.discriminator,
            real_audio=self.batch,
            fake_audio=recon,
            global_step=self.global_step,
        )

        # Combined generator loss
        total_g_loss = vae_losses["total_loss"] + adv_losses["total_adv_loss"]

        # Backward pass for generator
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        self.opt_g.step()

        # Collect generator losses
        losses["total_g_loss"] = total_g_loss.item()
        for k, v in vae_losses.items():
            losses[k] = v.item() if torch.is_tensor(v) else v
        if adv_losses["is_active"]:
            losses["g_adv_loss"] = adv_losses["g_adv_loss"].item()
            losses["fm_loss"] = adv_losses["fm_loss"].item()

        # =================
        # Train Discriminator
        # =================
        if self.model.adversarial_loss_calculator.is_active(self.global_step):
            self.opt_d.zero_grad()

            # Get fresh outputs (detach generator)
            with torch.no_grad():
                recon, _, _, _ = self.model(self.batch)

            # Discriminator losses
            disc_losses = self.model.adversarial_loss_calculator.compute_discriminator_loss(
                discriminator=self.model.discriminator,
                real_audio=self.batch,
                fake_audio=recon,
            )

            # Backward pass for discriminator
            disc_losses["d_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(), max_norm=1.0
            )
            self.opt_d.step()

            # Collect discriminator losses
            losses["d_loss"] = disc_losses["d_loss"].item()
            losses["d_real_acc"] = disc_losses["d_real_acc"]
            losses["d_fake_acc"] = disc_losses["d_fake_acc"]

        return losses

    def train(self, num_iterations: int) -> Dict[str, list]:
        """
        Train for specified number of iterations on the fixed batch.

        Args:
            num_iterations: Number of training iterations

        Returns:
            Dictionary of loss histories
        """
        self.model.train()
        history = {}

        pbar = tqdm(range(num_iterations), desc="Overfitting")

        for i in pbar:
            self.global_step = i

            # Update model's global_step for warmup schedules
            self.model._trainer = type('obj', (object,), {'global_step': i})()

            # Run training step
            if self.model.use_adversarial:
                losses = self._adversarial_step()
            else:
                losses = self._standard_step()

            # Store losses in history
            for k, v in losses.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            # Update progress bar
            if i % self.log_every == 0:
                # Get main loss for display
                main_loss = losses.get("total_g_loss", losses.get("total_loss", 0))
                pbar.set_postfix({"loss": f"{main_loss:.6f}"})

                # Log detailed metrics
                log.info(f"Step {i}: " + ", ".join(
                    f"{k}={v:.6f}" for k, v in losses.items()
                ))

        return history


def get_fixed_batch(
    cfg: DictConfig,
    batch_size: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Get a single fixed batch from the dataset.

    Args:
        cfg: Hydra configuration
        batch_size: Number of samples in the batch
        seed: Random seed for reproducibility

    Returns:
        Fixed batch tensor of shape (batch_size, channels, samples)
    """
    # Set seed for reproducibility
    pl.seed_everything(seed, workers=True)

    # Create a minimal datamodule config with our batch size
    datamodule_cfg = OmegaConf.to_container(cfg.datamodule, resolve=True)
    datamodule_cfg["batch_size"] = batch_size
    datamodule_cfg["train_samples"] = batch_size  # Only need one batch worth
    datamodule_cfg["num_workers"] = 0  # Avoid multiprocessing for reproducibility

    # Instantiate datamodule
    datamodule = instantiate(OmegaConf.create(datamodule_cfg))
    datamodule.setup("fit")

    # Get one batch
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))

    log.info(f"Fixed batch shape: {batch.shape}")
    log.info(f"Batch min: {batch.min():.4f}, max: {batch.max():.4f}")

    return batch


def overfit_on_batch(cfg: DictConfig) -> Dict[str, list]:
    """
    Main overfitting function.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary of loss histories
    """
    # Get overfit-specific settings
    overfit_cfg = cfg.get("overfit", {})
    batch_size = overfit_cfg.get("batch_size", 4)
    num_iterations = overfit_cfg.get("num_iterations", 1000)
    seed = overfit_cfg.get("seed", 42)
    log_every = overfit_cfg.get("log_every", 100)
    device = overfit_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_checkpoint = overfit_cfg.get("save_checkpoint", False)
    checkpoint_path = overfit_cfg.get("checkpoint_path", "overfit_checkpoint.pt")

    log.info(f"Overfitting settings:")
    log.info(f"  batch_size: {batch_size}")
    log.info(f"  num_iterations: {num_iterations}")
    log.info(f"  seed: {seed}")
    log.info(f"  device: {device}")

    # Set global seed
    pl.seed_everything(cfg.get("seed", seed), workers=True)

    # Get fixed batch
    log.info("Creating fixed batch...")
    batch = get_fixed_batch(cfg, batch_size, seed)

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if model.use_adversarial:
        disc_params = sum(p.numel() for p in model.discriminator.parameters())
        log.info(f"Discriminator parameters: {disc_params:,}")

    # Create trainer and run
    log.info("Starting overfitting...")
    trainer = OverfitTrainer(
        model=model,
        batch=batch,
        device=device,
        log_every=log_every,
    )

    history = trainer.train(num_iterations)

    # Log final results
    final_loss = history.get("total_g_loss", history.get("total_loss", [0]))[-1]
    initial_loss = history.get("total_g_loss", history.get("total_loss", [0]))[0]
    log.info(f"Training complete!")
    log.info(f"  Initial loss: {initial_loss:.6f}")
    log.info(f"  Final loss: {final_loss:.6f}")
    log.info(f"  Reduction: {(1 - final_loss/initial_loss) * 100:.2f}%")

    # Save checkpoint if requested
    if save_checkpoint:
        log.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save({
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": trainer.opt_g.state_dict(),
            "global_step": num_iterations,
            "final_loss": final_loss,
            "history": history,
        }, checkpoint_path)

    return history


@hydra.main(version_base="1.3", config_path="configs", config_name="overfit")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for overfitting script.

    Args:
        cfg: Configuration composed by Hydra
    """
    # Pretty print config
    if cfg.get("print_config", True):
        log.info("Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # Run overfitting
    history = overfit_on_batch(cfg)

    # Print summary
    print("\n" + "=" * 50)
    print("OVERFITTING COMPLETE")
    print("=" * 50)

    # Get final losses
    for key in ["total_loss", "total_g_loss", "recon_loss", "kl_loss"]:
        if key in history:
            initial = history[key][0]
            final = history[key][-1]
            print(f"{key}: {initial:.6f} -> {final:.6f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
