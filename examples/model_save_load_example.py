"""
Example: How to save and load AudioVAE models

This script demonstrates the new model save/load functionality
that makes it easy to save and load models with all hyperparameters.
"""

import torch
from src.model.AudioVAE import AudioVAE
from src.utils import save_model, load_model, load_training_state


def example_save_model():
    """Example: Save a model with hyperparameters"""

    # Create a model with specific hyperparameters
    model = AudioVAE(
        in_channels=2,
        base_channels=128,
        channel_mults=[1, 2, 4, 8],
        strides=[2, 4, 4, 8],
        latent_dim=64,
        kernel_size=7,
        dilations=[1, 3, 9],
    )

    # Save the model (all hyperparameters are automatically saved)
    save_model(
        model,
        save_path="checkpoints/my_model.pt",
        epoch=10,
        global_step=5000,
        metadata={"description": "My custom VAE model"}
    )

    print("✓ Model saved successfully!")


def example_load_model():
    """Example: Load a model (hyperparameters are automatically restored)"""

    # Load the model - no need to specify any hyperparameters!
    model = load_model("checkpoints/my_model.pt")

    # The model is ready to use
    model.eval()
    with torch.no_grad():
        # Example inference
        audio = torch.randn(1, 2, 44100)  # 1 second of stereo audio
        recon, z, mean, logvar = model(audio)
        print(f"Input shape: {audio.shape}")
        print(f"Reconstruction shape: {recon.shape}")
        print(f"Latent shape: {z.shape}")

    print("✓ Model loaded and used successfully!")


def example_resume_training():
    """Example: Resume training from a checkpoint"""

    # Load the model
    model = load_model("checkpoints/my_model.pt")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    # Load training state
    state = load_training_state(
        "checkpoints/my_model.pt",
        optimizer=optimizer,
        scheduler=scheduler
    )

    start_epoch = state.get("epoch", 0)
    print(f"✓ Resuming from epoch {start_epoch}")

    # Continue training...
    # for epoch in range(start_epoch, max_epochs):
    #     train_one_epoch(model, optimizer, scheduler)


def example_save_during_training():
    """Example: Save model during training"""

    model = AudioVAE(latent_dim=64, base_channels=128)
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(10):
        # ... training code ...

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_model(
                model,
                f"checkpoints/model_epoch_{epoch}.pt",
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                global_step=epoch * 1000,  # example
                metadata={"train_loss": 0.1234}  # example metrics
            )

    print("✓ Training completed with periodic saves!")


def example_convert_lightning_to_standalone():
    """Example: Convert a Lightning checkpoint to standalone format"""

    from src.utils import load_from_lightning_checkpoint

    # Load model from Lightning checkpoint
    model = load_from_lightning_checkpoint(
        "lightning_logs/version_0/checkpoints/last.ckpt"
    )

    # Save as standalone checkpoint
    save_model(model, "checkpoints/standalone_model.pt")

    # Now it can be loaded without Lightning
    loaded_model = load_model("checkpoints/standalone_model.pt")

    print("✓ Converted Lightning checkpoint to standalone!")


if __name__ == "__main__":
    print("Audio VAE Model Save/Load Examples")
    print("=" * 50)

    print("\n1. Saving a model:")
    print("-" * 50)
    example_save_model()

    print("\n2. Loading a model:")
    print("-" * 50)
    example_load_model()

    print("\n3. Resume training:")
    print("-" * 50)
    example_resume_training()

    print("\n4. Saving during training:")
    print("-" * 50)
    example_save_during_training()

    print("\n5. Convert Lightning checkpoint:")
    print("-" * 50)
    # example_convert_lightning_to_standalone()  # Requires actual checkpoint
    print("(Requires an actual Lightning checkpoint to run)")

    print("\n" + "=" * 50)
    print("All examples completed!")
