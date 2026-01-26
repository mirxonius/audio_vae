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
import os
import tempfile
from typing import Iterator, List

import hydra
import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

log = logging.getLogger(__name__)


class BatchSavingCallback(Callback):
    """
    Callback that saves original and reconstructed batches after every epoch.
    Useful for debugging and visualizing overfitting progress.
    """

    def __init__(
        self,
        batch: torch.Tensor,
        sample_rate: int = 44100,
        output_dir: str = None,
    ):
        """
        Args:
            batch: The original batch tensor to reconstruct (batch_size, channels, samples)
            sample_rate: Audio sample rate for saving wav files
            output_dir: Directory to save audio files. If None, uses temp directory
                       and logs to MLflow if available.
        """
        super().__init__()
        self.batch = batch
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self._temp_dir = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Create output directory if needed."""
        if self.output_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="batch_reconstructions_")
            self.output_dir = self._temp_dir
            log.info(f"Saving batch reconstructions to: {self.output_dir}")
        else:
            os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save original and reconstructed batch after each epoch."""
        # Only save on rank 0 in distributed training
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)

        pl_module.eval()
        with torch.no_grad():
            # Move batch to device and reconstruct
            batch_device = self.batch.to(pl_module.device)
            reconstruction, _, _, _ = pl_module(batch_device)

            # Move back to CPU for saving
            original = self.batch.cpu()
            reconstruction = reconstruction.cpu()

            # Save each sample in the batch
            for i in range(len(original)):
                # Save original
                original_path = os.path.join(epoch_dir, f"sample_{i:02d}_original.wav")
                torchaudio.save(
                    original_path,
                    original[i].float(),
                    self.sample_rate,
                )

                # Save reconstruction
                recon_path = os.path.join(epoch_dir, f"sample_{i:02d}_reconstruction.wav")
                torchaudio.save(
                    recon_path,
                    reconstruction[i].float(),
                    self.sample_rate,
                )

            # Log to MLflow if available
            if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger, "run_id"):
                for filename in os.listdir(epoch_dir):
                    filepath = os.path.join(epoch_dir, filename)
                    trainer.logger.experiment.log_artifact(
                        trainer.logger.run_id,
                        filepath,
                        artifact_path=f"batch_reconstructions/epoch_{epoch:04d}",
                    )

        pl_module.train()
        log.info(f"Saved batch reconstructions for epoch {epoch} to {epoch_dir}")


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

    # Create batch saving callback
    batch_saving_callback = BatchSavingCallback(
        batch=batch,
        sample_rate=cfg.get("sample_rate", 44100),
    )

    # Instantiate trainer with logger and callback (no other callbacks)
    log.info("Instantiating trainer")
    trainer = instantiate(
        cfg.trainer,
        logger=loggers if loggers else None,
        callbacks=[batch_saving_callback],
    )

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
