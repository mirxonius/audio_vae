"""
Model Save/Load Utilities for AudioVAE

Provides clean save/load functionality where models can be loaded
by only providing the AudioVAE class, with all hyperparameters
automatically restored from the checkpoint.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

from src.model.AudioVAE import AudioVAE


def save_model(
    model: AudioVAE,
    save_path: str,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save AudioVAE model with all hyperparameters and training state.

    Args:
        model: The AudioVAE model to save
        save_path: Path to save the checkpoint
        optimizer_state: Optional optimizer state dict
        scheduler_state: Optional scheduler state dict
        epoch: Optional current epoch number
        global_step: Optional current global step
        metadata: Optional additional metadata to save

    Example:
        >>> model = AudioVAE(latent_dim=64, base_channels=128)
        >>> save_model(model, "checkpoints/my_model.pt", epoch=10)
        >>> # Later, load with:
        >>> loaded_model = load_model("checkpoints/my_model.pt")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract model hyperparameters from __init__ signature
    # Store them explicitly to ensure reproducibility
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": model.encoder.in_channels,
            "base_channels": model.encoder.base_channels,
            "channel_mults": model.encoder.channel_mults,
            "strides": model.encoder.strides,
            "latent_dim": model.latent_dim,
            "kernel_size": model.encoder.kernel_size,
            "dilations": model.encoder.dilations,
        },
        "latent_statistics": {
            "latent_mean": model.latent_mean.data,
            "latent_std": model.latent_std.data,
        },
        "encoder_frozen": model.encoder_frozen,
    }

    # Add optional training state
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if global_step is not None:
        checkpoint["global_step"] = global_step
    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")


def load_model(
    checkpoint_path: str,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> AudioVAE:
    """
    Load AudioVAE model from checkpoint.

    The model is automatically reconstructed with the correct hyperparameters.
    Only requires the AudioVAE class - no need to specify architecture details.

    Args:
        checkpoint_path: Path to the checkpoint file
        map_location: Device to load the model on (e.g., 'cpu', 'cuda:0')
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Loaded AudioVAE model

    Example:
        >>> # Simple loading - architecture is inferred from checkpoint
        >>> model = load_model("checkpoints/my_model.pt")
        >>>
        >>> # Load on specific device
        >>> model = load_model("checkpoints/my_model.pt", map_location="cuda:0")
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Check if this is a new-style checkpoint with model_config
    if "model_config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'model_config'. "
            "This might be an old-style checkpoint. Please use the Lightning "
            "checkpoint loading mechanism or re-save the model using save_model()."
        )

    # Reconstruct model from config
    config = checkpoint["model_config"]
    model = AudioVAE(**config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Restore latent statistics if available
    if "latent_statistics" in checkpoint:
        model.latent_mean.data = checkpoint["latent_statistics"]["latent_mean"]
        model.latent_std.data = checkpoint["latent_statistics"]["latent_std"]

    # Restore encoder frozen state
    if checkpoint.get("encoder_frozen", False):
        model._freeze_encoder()

    print(f"✓ Model loaded from {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if "global_step" in checkpoint:
        print(f"  Global step: {checkpoint['global_step']}")

    return model


def load_training_state(
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load training state (optimizer, scheduler, epoch, etc.) from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        map_location: Device to load tensors on

    Returns:
        Dictionary containing training state information:
            - epoch: Training epoch
            - global_step: Global training step
            - metadata: Additional metadata if available

    Example:
        >>> model = load_model("checkpoint.pt")
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
        >>> state = load_training_state("checkpoint.pt", optimizer, scheduler)
        >>> print(f"Resuming from epoch {state['epoch']}")
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✓ Optimizer state loaded")
    elif optimizer is not None:
        warnings.warn("Optimizer provided but no optimizer state found in checkpoint")

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("✓ Scheduler state loaded")
    elif scheduler is not None:
        warnings.warn("Scheduler provided but no scheduler state found in checkpoint")

    # Return training state
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "metadata": checkpoint.get("metadata"),
    }


def load_from_lightning_checkpoint(
    checkpoint_path: str,
    map_location: Optional[str] = None,
) -> AudioVAE:
    """
    Load AudioVAE model from a PyTorch Lightning checkpoint.

    This is useful for extracting the AudioVAE model from a full
    VAELightningModule checkpoint.

    Args:
        checkpoint_path: Path to the Lightning checkpoint (.ckpt)
        map_location: Device to load the model on

    Returns:
        The AudioVAE model extracted from the Lightning checkpoint

    Example:
        >>> # Load model from Lightning checkpoint
        >>> model = load_from_lightning_checkpoint("lightning_logs/checkpoints/last.ckpt")
        >>> # Now you can use it for inference
        >>> model.eval()
        >>> with torch.no_grad():
        ...     recon, z, mean, logvar = model(audio)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Lightning checkpoints store state dict under "state_dict" key
    if "state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not appear to be a Lightning checkpoint"
        )

    # Extract model hyperparameters from Lightning checkpoint
    # These are stored in hyperparameters
    hparams = checkpoint.get("hyper_parameters", {})

    # Try to get architecture config
    if "architecture" in hparams:
        arch_config = hparams["architecture"]
        if hasattr(arch_config, "__dict__"):
            # It's an object, extract attributes
            config = {
                k: v for k, v in arch_config.__dict__.items() if not k.startswith("_")
            }
        elif isinstance(arch_config, dict):
            config = arch_config
        else:
            raise ValueError(
                "Unable to parse architecture config from Lightning checkpoint"
            )
    else:
        # Fallback: try to infer from state dict
        warnings.warn(
            "No architecture config found in checkpoint. "
            "Attempting to infer from state dict keys..."
        )
        state_dict = checkpoint["state_dict"]

        # Filter to only model.* keys
        model_state = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

        # Create a minimal model and load state
        # User might need to specify config manually
        raise NotImplementedError(
            "Cannot infer model config from Lightning checkpoint. "
            "Please use save_model() to create a standalone checkpoint first, "
            "or manually instantiate the model with the correct config."
        )

    # Create model
    model = AudioVAE(**config)

    # Extract and load model state dict
    model_state = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(model_state)

    print(f"✓ Model loaded from Lightning checkpoint: {checkpoint_path}")

    return model
