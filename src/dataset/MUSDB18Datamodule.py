from typing import Optional

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.dataset.MUSDB18Dataset import MUSDB18Dataset

try:
    import musdb

    MUSDB_AVAILABLE = True
except ImportError:
    MUSDB_AVAILABLE = False
    print("Warning: musdb not installed. Install with: pip install musdb")


class MUSDB18DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MUSDB18 dataset.
    Uses explicit parameters instead of nested config.
    """

    def __init__(
        self,
        root: str,
        sample_rate: int = 44100,
        chunk_duration: float = 6.0,
        channels: int = 2,
        train_samples: int = 10000,
        val_samples: int = 1000,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            root: Path to MUSDB18 dataset root directory
            sample_rate: Audio sample rate (default: 44100)
            chunk_duration: Duration of audio chunks in seconds
            channels: Number of audio channels (1=mono, 2=stereo)
            train_samples: Number of training samples per epoch
            val_samples: Number of validation samples per epoch
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in DataLoader
            drop_last: Whether to drop last incomplete batch in training
        """
        super().__init__()
        self.save_hyperparameters()

        # Dataset parameters
        self.root = root
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels
        self.train_samples = train_samples
        self.val_samples = val_samples

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Computed parameters
        self.chunk_samples = int(chunk_duration * sample_rate)

        # Datasets (created in setup)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up train and validation datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = MUSDB18Dataset(
                root=self.root,
                split="train",
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                channels=self.channels,
                num_samples=self.train_samples,
            )

            self.val_dataset = MUSDB18Dataset(
                root=self.root,
                split="valid",
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                channels=self.channels,
                num_samples=self.val_samples,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch in validation
            persistent_workers=self.num_workers > 0,
        )
