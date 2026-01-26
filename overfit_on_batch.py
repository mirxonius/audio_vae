"""
Overfit on a single batch - debugging script to verify model can learn.

This script trains the model on a single fixed batch of data to verify that
the model architecture and training loop can successfully minimize the loss.

Uses the existing VAELightningModule with a dummy dataloader that always
returns the same cached batch.

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
from typing import Iterator, List

import hydra
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

log = logging.getLogger(__name__)


def instantiate_loggers(cfg: DictConfig) -> List:
    """
    Instantiate loggers from config.

    Args:
        cfg: Hydra configuration containing logger config

    Returns:
        List of instantiated loggers
    """
    loggers = []
    if cfg.get("logger"):
        for lg_name, lg_conf in cfg.logger.items():
            if lg_conf is not None and lg_conf.get("_target_"):
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(instantiate(lg_conf))
    return loggers


class SingleBatchDataset(Dataset):
    """Dataset that always returns the same cached batch."""

    def __init__(self, batch: torch.Tensor, num_samples: int):
        """
        Args:
            batch: Tensor of shape (batch_size, channels, samples)
        """
        self.batch = batch
        self.num_samples = num_samples
        self.batch_size = len(batch)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx = idx % self.batch_size
        return self.batch[idx]


class SingleBatchDataModule(pl.LightningDataModule):
    """DataModule that wraps a single batch for overfitting."""

    def __init__(self, batch: torch.Tensor, batch_size: int, num_samples: int):
        super().__init__()
        self.batch = batch
        self.batch_size = batch_size
        self.num_samples = num_samples

    def train_dataloader(self) -> DataLoader:
        dataset = SingleBatchDataset(self.batch, num_samples=self.num_samples)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep order consistent
            num_workers=0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        # Return same batch for validation
        return self.train_dataloader()


def get_fixed_batch(cfg: DictConfig) -> torch.Tensor:
    """
    Get a single fixed batch from the original datamodule.

    Args:
        cfg: Hydra configuration

    Returns:
        Fixed batch tensor of shape (batch_size, channels, samples)
    """
    # Instantiate the original datamodule
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup("fit")

    # Get one batch
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))

    log.info(f"Cached batch shape: {batch.shape}")
    log.info(
        f"Batch stats - min: {batch.min():.4f}, max: {batch.max():.4f}, mean: {batch.mean():.4f}"
    )

    return batch


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

    # Get and cache a single batch from the original datamodule
    log.info("Fetching and caching single batch...")
    batch = get_fixed_batch(cfg)

    # Create dummy datamodule with the cached batch
    datamodule = SingleBatchDataModule(
        batch=batch, batch_size=len(batch), num_samples=8000
    )

    # Instantiate model (uses existing VAELightningModule)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    if model.use_adversarial:
        disc_params = sum(p.numel() for p in model.discriminator.parameters())
        log.info(f"Discriminator parameters: {disc_params:,}")

    # Instantiate logger(s)
    loggers = instantiate_loggers(cfg)

    # Instantiate trainer with logger
    log.info("Instantiating trainer")
    trainer = instantiate(cfg.trainer, logger=loggers if loggers else None)

    # Log hyperparameters to all loggers
    if cfg.get("log_hyperparameters") and trainer.logger:
        log.info("Logging hyperparameters to MLflow")
        hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        # Add batch info to hyperparameters
        hparams["batch_shape"] = list(batch.shape)
        hparams["batch_stats"] = {
            "min": float(batch.min()),
            "max": float(batch.max()),
            "mean": float(batch.mean()),
        }
        trainer.logger.log_hyperparams(hparams)

    # Run training
    log.info("Starting overfitting on single batch...")
    trainer.fit(model, datamodule)

    # Print summary
    print("\n" + "=" * 50)
    print("OVERFITTING COMPLETE")
    print("=" * 50)
    print(f"Final step: {trainer.global_step}")

    # Log final metrics summary
    if trainer.callback_metrics:
        print("\nFinal metrics:")
        for name, value in trainer.callback_metrics.items():
            print(f"  {name}: {value:.6f}")

    # Print MLflow run info
    if trainer.logger:
        for lg in trainer.loggers:
            if hasattr(lg, "run_id"):
                print(f"\nMLflow run ID: {lg.run_id}")
                print(f"MLflow experiment: {lg.experiment_id}")

    print("=" * 50)


if __name__ == "__main__":
    main()
