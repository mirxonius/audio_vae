# train.py
import logging
from pathlib import Path
from typing import List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import mlflow
from hydra.utils import instantiate

log = logging.getLogger(__name__)


def get_mlflow_run_name() -> str:
    """Get the current MLflow run name."""
    run = mlflow.active_run()
    if run:
        return run.info.run_name
    return "default_run"


# Register the resolver before Hydra processes configs
OmegaConf.register_new_resolver("mlflow_run_name", get_mlflow_run_name, replace=True)


def train_model(cfg: DictConfig) -> Optional[float]:
    """
    Training function with full Hydra instantiation.

    Args:
        cfg: Hydra configuration

    Returns:
        Best validation metric (for hyperparameter optimization)
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        log.info(f"Set seed to {cfg.seed}")

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = instantiate(cfg.model)

    # Instantiate logger(s)
    logger = []
    if cfg.get("logger"):
        for lg_name, lg_conf in cfg.logger.items():
            if lg_conf is not None and lg_conf.get("_target_"):
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))

    # Instantiate callbacks
    callbacks: List[pl.Callback] = []
    if cfg.get("callbacks"):
        for cb_name, cb_conf in cfg.callbacks.items():
            if cb_conf is not None and cb_conf.get("_target_"):
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger if logger else None,
    )

    # Log hyperparameters to all loggers
    if cfg.get("log_hyperparameters") and trainer.logger:
        log.info("Logging hyperparameters")
        # Flatten config for logging
        hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        trainer.logger.log_hyperparams(hparams)

    # Train
    log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path"),
    )
    log.info("Training complete!")

    # Return best metric for optuna/hyperparameter sweeps
    best_metric = None
    if cfg.get("return_metric") and trainer.callback_metrics:
        best_metric = trainer.callback_metrics.get(cfg.return_metric)
        if best_metric is not None:
            best_metric = best_metric.item()
            log.info(f"Best {cfg.return_metric}: {best_metric}")

    # Optional testing
    if cfg.get("run_test") and trainer.checkpoint_callback:
        log.info("Running test on best checkpoint")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    return best_metric


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.

    Args:
        cfg: Configuration composed by Hydra

    Returns:
        Best metric value (for hyperparameter optimization)
    """
    # Pretty print config (but don't resolve mlflow_run_name yet)
    if cfg.get("print_config"):
        # Use resolve=False to avoid resolving mlflow_run_name before MLflow starts
        log.info("Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=False))

    return train_model(cfg)


if __name__ == "__main__":
    main()
