"""
Overfit on a single batch - debugging script to verify model can learn.

This script uses PyTorch Lightning's built-in `overfit_batches` feature to train
the model on a single fixed batch. This verifies that the model architecture
and training loop can successfully minimize the loss.

Uses the existing VAELightningModule - no custom training loop needed.

Usage:
    # Basic usage with defaults
    python overfit_on_batch.py

    # Disable adversarial training (faster)
    python overfit_on_batch.py model.use_adversarial=false

    # Change learning rate
    python overfit_on_batch.py model.optimizer.lr=1e-3

    # More iterations
    python overfit_on_batch.py trainer.max_steps=5000

    # Different latent dimension
    python overfit_on_batch.py model.architecture.latent_dim=64
"""

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="overfit")
def main(cfg: DictConfig) -> None:
    """
    Overfit on a single batch using PyTorch Lightning.

    Args:
        cfg: Configuration composed by Hydra
    """
    # Print config
    if cfg.get("print_config", True):
        log.info("Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # Set seed for reproducibility (ensures same batch every time)
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = instantiate(cfg.datamodule)

    # Instantiate model (uses existing VAELightningModule)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if model.use_adversarial:
        disc_params = sum(p.numel() for p in model.discriminator.parameters())
        log.info(f"Discriminator parameters: {disc_params:,}")

    # Instantiate trainer with overfit_batches
    log.info("Instantiating trainer with overfit_batches=1")
    trainer = instantiate(cfg.trainer)

    # Run training
    log.info("Starting overfitting on single batch...")
    trainer.fit(model, datamodule)

    # Print summary
    print("\n" + "=" * 50)
    print("OVERFITTING COMPLETE")
    print("=" * 50)
    print(f"Final step: {trainer.global_step}")
    print("=" * 50)


if __name__ == "__main__":
    main()
