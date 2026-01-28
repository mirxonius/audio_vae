from typing import Optional

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.dataset.GTZANDataset import GTZANDataset


class GTZANDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the GTZAN Music Genre Classification dataset.
    Provides audio chunks for autoencoder training (no genre labels).
    """

    def __init__(
        self,
        root: str,
        sample_rate: int = 22050,
        chunk_duration: float = 6.0,
        channels: int = 1,
        train_samples: int = 8000,
        val_samples: int = 800,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            root: Path to the GTZAN dataset root directory.
            sample_rate: Audio sample rate (default: 22050, GTZAN native).
            chunk_duration: Duration of audio chunks in seconds.
            channels: Number of audio channels (1=mono, 2=stereo).
            train_samples: Number of training samples per virtual epoch.
            val_samples: Number of validation samples per virtual epoch.
            batch_size: Batch size for dataloaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory in DataLoader.
            drop_last: Whether to drop last incomplete batch in training.
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
            self.train_dataset = GTZANDataset(
                root=self.root,
                split="train",
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                channels=self.channels,
                num_samples=self.train_samples,
            )

            self.val_dataset = GTZANDataset(
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
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
