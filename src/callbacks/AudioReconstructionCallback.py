import os
import tempfile
import torch
import torchaudio
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class AudioReconstructionCallback(Callback):
    """
    Callback to log audio reconstructions during validation.

    This callback is dataset-agnostic and uses the validation DataLoader
    from the trainer's datamodule to obtain samples for reconstruction.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        num_samples: int = 2,
        log_every_n_epochs: int = 5,
    ):
        """
        Args:
            sample_rate: Audio sample rate for saving WAV files.
                         Should match the datamodule's sample rate.
            num_samples: Number of audio samples to log per epoch.
            log_every_n_epochs: Log audio every N epochs.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs

        # Store fixed samples for consistent evaluation across epochs
        self._fixed_samples: Optional[torch.Tensor] = None
        self._samples_initialized = False

    def _initialize_fixed_samples(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        Initialize fixed samples from the validation dataloader.

        These samples are stored and reused across epochs for consistent
        comparison of reconstruction quality over training.
        """
        if self._samples_initialized:
            return

        # Get validation dataloader from trainer
        val_dataloader = trainer.datamodule.val_dataloader()

        # Collect samples from the validation set
        samples_collected = []
        samples_needed = self.num_samples

        for batch in val_dataloader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                # If batch is (data, labels) or similar
                audio_batch = batch[0]
            else:
                # If batch is just the audio tensor
                audio_batch = batch

            # audio_batch shape: [batch_size, channels, time]
            batch_size = audio_batch.shape[0]

            for i in range(min(batch_size, samples_needed - len(samples_collected))):
                samples_collected.append(audio_batch[i].clone())

            if len(samples_collected) >= samples_needed:
                break

        if len(samples_collected) > 0:
            self._fixed_samples = torch.stack(samples_collected)
            self._samples_initialized = True

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Log audio reconstructions at the end of validation epoch."""

        # Only log every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Only log on rank 0 in distributed training
        if trainer.global_rank != 0:
            return

        # Check if datamodule is available
        if trainer.datamodule is None:
            return

        # Initialize fixed samples on first call
        self._initialize_fixed_samples(trainer, pl_module)

        if self._fixed_samples is None or len(self._fixed_samples) == 0:
            return

        pl_module.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            with torch.no_grad():
                for i in range(min(self.num_samples, len(self._fixed_samples))):
                    # Get audio sample and move to device
                    audio = self._fixed_samples[i].unsqueeze(0).to(pl_module.device)

                    # Reconstruct
                    recon, _, _, _ = pl_module(audio)

                    # Move to CPU and remove batch dimension
                    audio = audio.squeeze(0).cpu()
                    recon = recon.squeeze(0).cpu()

                    # Create descriptive filename
                    base_name = f"epoch{trainer.current_epoch}_sample{i}"

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
