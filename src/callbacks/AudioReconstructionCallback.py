import os
import tempfile
import torch
import torchaudio
import numpy as np
import musdb
import random
from typing import Optional, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class AudioReconstructionCallback(Callback):
    """
    Callback to log longer audio reconstructions during validation.
    Uses longer chunks than training to better evaluate reconstruction quality.
    """

    def __init__(
        self,
        musdb_root: str,
        sample_rate: int = 44100,
        chunk_duration: float = 5.0,  # Longer chunks for evaluation
        num_samples: int = 2,
        log_every_n_epochs: int = 5,
        stems: Optional[List[str]] = None,
        include_mixture: bool = True,
    ):
        """
        Args:
            musdb_root: Path to MUSDB18 dataset
            sample_rate: Audio sample rate
            chunk_duration: Duration of audio chunks to log (in seconds)
            num_samples: Number of audio samples to log per epoch
            log_every_n_epochs: Log audio every N epochs
            stems: List of stems to sample from (default: ['vocals', 'drums', 'bass', 'other'])
            include_mixture: Whether to include full mixture samples
            seed: Random seed for reproducible sample selection
        """
        super().__init__()
        self.musdb_root = musdb_root
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.stems = stems or ["vocal", "drums", "bass", "other"]
        self.include_mixture = include_mixture

        # Load validation set
        self.mus = musdb.DB(root=musdb_root, subsets="train", split="valid")

        # Pre-select tracks and configurations for consistency
        self._prepare_eval_samples()

    def _prepare_eval_samples(self):
        """Pre-select tracks and configurations for consistent evaluation"""
        rng = random.Random()

        self.eval_configs = []

        for i in range(self.num_samples):
            track = rng.choice(self.mus.tracks)

            # Decide if mixture or stem
            if self.include_mixture and rng.random() < 0.3:
                stem_name = "mixture"
            else:
                stem_name = rng.choice(self.stems)

            # Random start position (but consistent for this sample)
            if stem_name == "mixture":
                audio_length = len(track.audio)
            else:
                audio_length = len(track.targets[stem_name].audio)

            max_start = max(0, audio_length - self.chunk_samples)
            start_sample = rng.randint(0, max_start) if max_start > 0 else 0

            self.eval_configs.append(
                {
                    "track_name": track.name,
                    "track_idx": self.mus.tracks.index(track),
                    "stem": stem_name,
                    "start_sample": start_sample,
                }
            )

    def _load_audio_chunk(self, config: dict) -> torch.Tensor:
        """Load a specific audio chunk based on configuration"""
        track = self.mus.tracks[config["track_idx"]]

        # Get audio
        if config["stem"] == "mixture":
            audio = track.audio
        else:
            audio = track.targets[config["stem"]].audio

        # Extract chunk
        start = config["start_sample"]
        end = start + self.chunk_samples
        chunk = audio[start:end]

        # Pad if necessary
        if len(chunk) < self.chunk_samples:
            chunk = np.pad(chunk, ((0, self.chunk_samples - len(chunk)), (0, 0)))

        # Convert to tensor (Channels, Time)
        chunk = torch.from_numpy(chunk.T).float()

        # Peak normalization
        peak = chunk.abs().max()
        if peak > 0:
            chunk = chunk / peak

        return chunk

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Log audio reconstructions at the end of validation epoch"""

        # Only log every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Only log on rank 0 in distributed training
        if trainer.global_rank != 0:
            return

        pl_module.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            with torch.no_grad():
                for i, config in enumerate(self.eval_configs):
                    # Load audio chunk
                    audio = self._load_audio_chunk(config)
                    audio = audio.unsqueeze(0).to(
                        pl_module.device
                    )  # Add batch dimension

                    # Reconstruct
                    recon, _, _, _ = pl_module(audio)

                    # Move to CPU
                    audio = audio.squeeze(0).cpu()
                    recon = recon.squeeze(0).cpu()

                    # Create descriptive filename
                    stem_label = config["stem"]
                    track_name = (
                        config["track_name"].replace("/", "_").replace(" ", "_")
                    )
                    base_name = f"epoch{trainer.current_epoch}_{stem_label}"

                    # Save original
                    original_path = os.path.join(temp_dir, f"{base_name}_orig.wav")
                    torchaudio.save(
                        original_path,
                        audio.float(),
                        self.sample_rate,
                    )

                    # Save reconstruction
                    recon_path = os.path.join(temp_dir, f"{base_name}_recon.wav")
                    torchaudio.save(
                        recon_path,
                        recon.float(),
                        self.sample_rate,
                    )

                    # Log to MLflow
                    if hasattr(trainer.logger, "experiment"):
                        artifact_path = f"rec/epoch_{trainer.current_epoch}"

                        trainer.logger.experiment.log_artifact(
                            trainer.logger.run_id,
                            original_path,
                            artifact_path=artifact_path,
                        )
                        trainer.logger.experiment.log_artifact(
                            trainer.logger.run_id,
                            recon_path,
                            artifact_path=artifact_path,
                        )

        pl_module.train()
