import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow.pytorch
import logging
import torch


class MLflowVAETrackerCallback(pl.Callback):
    """
    Advanced callback to log the inner AudioVAE model to MLflow at multiple stages.
    """

    def __init__(
        self,
        monitor: str = "val/total_loss",
        mode: str = "min",
        every_n_epochs: int = 0,  # 0 to disable periodic logging
        save_best_only: bool = True,
        artifact_path: str = "vae_model",
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = every_n_epochs
        self.save_best_only = save_best_only
        self.base_path = artifact_path

        # Internal state for tracking best score
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.is_better = torch.lt if mode == "min" else torch.gt

    def _ensure_mlflow_run(self, trainer: "pl.Trainer") -> bool:
        """
        Ensures an active MLflow run exists before logging.
        Returns True if run is active, False otherwise.
        """
        if not isinstance(trainer.logger, MLFlowLogger):
            logging.warning(
                "Trainer logger is not MLFlowLogger. Skipping model logging."
            )
            return False

        mlflow_logger = trainer.logger

        # Check if run exists and is active
        if mlflow_logger.run_id is None:
            logging.warning("MLflow run_id is None. Skipping model logging.")
            return False

        # Set the active run to ensure context is correct
        try:
            if (
                mlflow.active_run() is None
                or mlflow.active_run().info.run_id != mlflow_logger.run_id
            ):
                mlflow.start_run(run_id=mlflow_logger.run_id)
            return True
        except Exception as e:
            logging.error(f"Failed to ensure MLflow run: {e}")
            return False

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        """Logs the model if it achieves a new 'best' score."""
        if self.monitor not in trainer.callback_metrics:
            return

        current_score = trainer.callback_metrics[self.monitor].item()

        if self.is_better(torch.tensor(current_score), torch.tensor(self.best_score)):
            self.best_score = current_score
            logging.info(
                f"New best {self.monitor}: {current_score:.4f}. Logging 'best' model."
            )
            self._log_model(trainer, pl_module, suffix="best")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        """Logs the model periodically every N epochs."""
        if (
            self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            logging.info(
                f"Periodic log at epoch {trainer.current_epoch}. Logging 'checkpoint' model."
            )
            self._log_model(trainer, pl_module, suffix=f"epoch_{trainer.current_epoch}")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Logs the final state of the model."""
        logging.info("Training complete. Logging 'final' model.")
        self._log_model(trainer, pl_module, suffix="final")

    def _log_model(self, trainer, pl_module, suffix: str):
        if not self._ensure_mlflow_run(trainer):
            return

        # Ensure we log the pure AudioVAE (submodule)
        vae_model = pl_module.model
        artifact_path = f"{self.base_path}_{suffix}"

        try:
            mlflow.pytorch.log_model(
                pytorch_model=vae_model,
                artifact_path=artifact_path,
                # We can also register the 'best' version in the model registry
                registered_model_name="AudioVAE_Best" if suffix == "best" else None,
            )
            logging.info(f"Successfully logged model to MLflow: {artifact_path}")
        except Exception as e:
            logging.error(f"Failed to log model to MLflow: {e}")
