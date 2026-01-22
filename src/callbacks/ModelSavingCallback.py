"""
Model Saving Callback for Audio VAE with MLflow integration
Saves best model based on validation metric and logs to MLflow
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import mlflow
import pytorch_lightning as pl
import torch
import torchaudio
import yaml


class ModelSavingCallback(pl.Callback):
    """
    Callback for saving models to MLflow based on validation metrics

    Features:
    - Saves best model based on specified metric
    - Logs model to MLflow with metadata
    - Optional audio reconstruction logging
    - Optional model versioning in MLflow Model Registry
    - Saves source code for reproducibility
    """

    def __init__(
        self,
        save_dir: str,
        metric: str = "val/total_loss",
        mode: str = "min",
        temp_code_src_path: str = "./temp_src",
        log_reconstructions: bool = True,
        num_reconstruction_samples: int = 3,
        sample_rate: int = 44100,
        create_model_version: bool = False,
        model_name: Optional[str] = None,
        min_metric_threshold: Optional[float] = None,
    ):
        """
        Args:
            save_dir: Directory where config files are stored
            metric: Metric to monitor (e.g., "val/total_loss", "val/stft_loss")
            mode: "min" or "max" - whether lower or higher metric is better
            temp_code_src_path: Path to store temporary source code copy
            log_reconstructions: Whether to log audio reconstructions
            num_reconstruction_samples: Number of audio samples to log
            sample_rate: Audio sample rate
            create_model_version: Whether to create model version in MLflow registry
            model_name: Name for model in MLflow registry (required if create_model_version=True)
            min_metric_threshold: Minimum metric value required for model versioning
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'!")

        if create_model_version and model_name is None:
            raise ValueError("model_name must be provided if create_model_version=True")

        self.save_dir = Path(save_dir)
        self.best_metric = None
        self.metric = metric
        self.mode = mode
        self.log_reconstructions = log_reconstructions
        self.num_reconstruction_samples = num_reconstruction_samples
        self.sample_rate = sample_rate
        self.temp_code_src_path = Path(temp_code_src_path)
        self.create_model_version = create_model_version
        self.model_name = model_name
        self.min_metric_threshold = min_metric_threshold

        # Track if we've logged a model this run
        self.model_logged_this_run = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize temp source directory at training start"""
        if trainer.is_global_zero:
            client = mlflow.MlflowClient()
            run = client.get_run(trainer.logger.run_id)
            run_name = run.data.tags.get("mlflow.runName", "unnamed_run")
            self.create_temp_src(temp_src_name=run_name)

    def create_temp_src(self, temp_src_name: str):
        """Create temporary directory with source code for MLflow logging"""
        self.temp_code_src_path = self.temp_code_src_path / temp_src_name

        if not self.temp_code_src_path.exists():
            self.temp_code_src_path.mkdir(parents=True, exist_ok=True)

        # Copy source code (adjust paths based on your project structure)
        src_dirs = ["src"]  # Your source directories

        for src_dir in src_dirs:
            if os.path.exists(src_dir):
                shutil.copytree(
                    src_dir,
                    self.temp_code_src_path / src_dir,
                    dirs_exist_ok=True,
                    ignore=self._include_only_py_and_yaml,
                )

    def destroy_temp_src(self):
        """Clean up temporary source directory"""
        if self.temp_code_src_path.exists():
            shutil.rmtree(self.temp_code_src_path)

    def __del__(self):
        """Cleanup on deletion"""
        logging.info("ModelSavingCallback deleted")
        # Uncomment if you want automatic cleanup
        # self.destroy_temp_src()

    @staticmethod
    def _include_only_py_and_yaml(src, names):
        """Filter function to include only .py and .yaml files"""
        included = []

        for name in names:
            full_path = os.path.join(src, name)

            # Include directories
            if os.path.isdir(full_path):
                included.append(name)
            # Include .py and .yaml files
            elif name.endswith((".py", ".yaml", ".yml")):
                included.append(name)

        # Ignore everything else
        ignored = [name for name in names if name not in included]
        return ignored

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Enable MLflow system metrics logging"""
        if trainer.is_global_zero:
            os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    def _log_audio_reconstructions(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Generate and log audio reconstructions to MLflow"""
        import tempfile

        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        # Get a batch
        batch = next(iter(val_dataloader))
        audio = batch["audio"].to(pl_module.device)

        # Generate reconstructions
        pl_module.eval()
        with torch.no_grad():
            recon, z, mean, logvar = pl_module(audio)
        pl_module.train()

        # Log samples
        num_samples = min(self.num_reconstruction_samples, audio.size(0))

        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(num_samples):
                # Save original
                original_path = os.path.join(temp_dir, f"best_model_original_{i}.wav")
                torchaudio.save(
                    original_path,
                    audio[i].cpu(),
                    self.sample_rate,
                )

                # Save reconstruction
                recon_path = os.path.join(
                    temp_dir, f"best_model_reconstruction_{i}.wav"
                )
                torchaudio.save(
                    recon_path,
                    recon[i].cpu(),
                    self.sample_rate,
                )

                # Log to MLflow
                mlflow.log_artifact(
                    original_path,
                    artifact_path="best_model_audio",
                    run_id=trainer.logger.run_id,
                )
                mlflow.log_artifact(
                    recon_path,
                    artifact_path="best_model_audio",
                    run_id=trainer.logger.run_id,
                )

        logging.info(f"Logged {num_samples} audio reconstruction samples")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check if current model is best and save to MLflow"""
        current_metric = trainer.callback_metrics.get(self.metric, None)

        if current_metric is None:
            logging.warning(f"Metric '{self.metric}' not found in callback_metrics")
            return

        # Check if this is the best metric
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif self.mode == "min" and current_metric < self.best_metric:
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            is_best = True

        if not is_best:
            return

        # Update best metric
        self.best_metric = current_metric

        # Load config
        config_path = self.save_dir / "configs" / "config.yaml"
        if not config_path.exists():
            config_path = self.save_dir / "config.yaml"

        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    logging.error(f"Error loading config: {exc}")

        if trainer.is_global_zero:
            logging.info(
                f"New best model! {self.metric}={current_metric:.6f} "
                f"(epoch {trainer.current_epoch})"
            )

            # Prepare metadata
            metadata = {
                "config": config,
                "best_metric": float(current_metric),
                "metric_name": self.metric,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
            }

            # Add adversarial training info if applicable
            if hasattr(pl_module, "use_adversarial") and pl_module.use_adversarial:
                metadata["adversarial_training"] = True
                metadata["adversarial_active"] = pl_module.is_adversarial_active()

            # Log model to MLflow
            try:
                mlflow.pytorch.log_model(
                    pl_module.model,
                    "model",
                    run_id=trainer.logger.run_id,
                    metadata=metadata,
                    code_paths=(
                        [str(self.temp_code_src_path / "src")]
                        if (self.temp_code_src_path / "src").exists()
                        else None
                    ),
                )
                self.model_logged_this_run = True
                logging.info("Model logged to MLflow successfully")
            except Exception as e:
                logging.error(f"Error logging model to MLflow: {e}")

            # Log audio reconstructions if requested
            if self.log_reconstructions:
                try:
                    self._log_audio_reconstructions(trainer, pl_module)
                except Exception as e:
                    logging.error(f"Error logging audio reconstructions: {e}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create model version in MLflow registry if criteria met"""
        if not trainer.is_global_zero:
            return

        if not self.create_model_version:
            return

        if not self.model_logged_this_run:
            logging.info("No model was logged, skipping model versioning")
            return

        # Check if metric threshold is met
        if self.min_metric_threshold is not None:
            if self.best_metric is None:
                logging.warning("No best metric recorded, skipping model versioning")
                return

            if self.mode == "min" and self.best_metric > self.min_metric_threshold:
                logging.info(
                    f"Best metric {self.best_metric:.6f} does not meet threshold "
                    f"{self.min_metric_threshold} (mode=min), skipping versioning"
                )
                return
            elif self.mode == "max" and self.best_metric < self.min_metric_threshold:
                logging.info(
                    f"Best metric {self.best_metric:.6f} does not meet threshold "
                    f"{self.min_metric_threshold} (mode=max), skipping versioning"
                )
                return

        # Load config for tags
        config_path = self.save_dir / "configs" / "config.yaml"
        if not config_path.exists():
            config_path = self.save_dir / "config.yaml"

        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    logging.error(f"Error loading config: {exc}")

        # Prepare tags
        tags = {
            "best_metric": f"{self.best_metric:.6f}",
            "metric_name": self.metric,
        }

        # Add config-specific tags
        if "model" in config:
            model_config = config["model"]
            if "latent_dim" in model_config:
                tags["latent_dim"] = str(model_config["latent_dim"])
            if "strides" in model_config:
                tags["compression_factor"] = str(model_config["strides"])

        if "datamodule" in config:
            if "sample_rate" in config["datamodule"]:
                tags["sample_rate"] = str(config["datamodule"]["sample_rate"])

        # Add adversarial training info
        if hasattr(pl_module, "use_adversarial") and pl_module.use_adversarial:
            tags["adversarial_training"] = "true"
            tags["adversarial_weight"] = str(pl_module.adversarial_weight)

        # Create description
        metrics_str = ", ".join(
            f"{k}={v:.4f}"
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, (int, float))
        )
        description = (
            f"Best {self.metric}: {self.best_metric:.6f} | "
            f"Epoch: {trainer.current_epoch} | "
            f"Metrics: {metrics_str}"
        )

        # Create model version
        try:
            client = mlflow.MlflowClient()
            version = client.create_model_version(
                name=self.model_name,
                source=f"runs:/{trainer.logger.run_id}/model",
                run_id=trainer.logger.run_id,
                description=description,
                tags=tags,
            )
            logging.info(
                f"Created model version {version.version} for model '{self.model_name}'"
            )
        except Exception as e:
            logging.error(f"Error creating model version: {e}")

        # Cleanup environment variable
        if "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING" in os.environ:
            del os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"]


class EarlyStoppingWithPatience(pl.Callback):
    """
    Enhanced early stopping that works well with adversarial training
    Includes patience for metric improvements and minimum training time
    """

    def __init__(
        self,
        monitor: str = "val/total_loss",
        min_delta: float = 0.0001,
        patience: int = 10,
        mode: str = "min",
        min_epochs: int = 50,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement after which training stops
            mode: "min" or "max"
            min_epochs: Minimum number of epochs before early stopping can trigger
            verbose: Whether to print early stopping info
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.min_epochs = min_epochs
        self.verbose = verbose

        self.best_score = None
        self.wait_count = 0

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check if training should stop"""
        if trainer.current_epoch < self.min_epochs:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            return

        # Check for improvement
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.wait_count = 0
            if self.verbose:
                logging.info(
                    f"Metric improved to {current_score:.6f}, "
                    f"resetting patience counter"
                )
        else:
            self.wait_count += 1
            if self.verbose:
                logging.info(
                    f"No improvement for {self.wait_count}/{self.patience} epochs"
                )

            if self.wait_count >= self.patience:
                if self.verbose:
                    logging.info(
                        f"Early stopping triggered after {self.wait_count} epochs "
                        f"without improvement"
                    )
                trainer.should_stop = True
