# Discriminator Configuration and Model Save/Load Guide

This document explains the improved discriminator configuration system and the new model save/load utilities.

## Table of Contents

1. [Overview of Improvements](#overview-of-improvements)
2. [Discriminator Configuration](#discriminator-configuration)
3. [Adversarial Loss Calculator](#adversarial-loss-calculator)
4. [Model Save/Load Utilities](#model-saveload-utilities)
5. [Migration Guide](#migration-guide)

---

## Overview of Improvements

### What Changed?

Previously, the discriminator was hardcoded in `VAELightningModule` with no way to configure it. The adversarial loss calculation was scattered across the training loop, making it hard to modify or experiment with different loss strategies.

**New Features:**

1. **Configurable Discriminator**: All discriminator parameters can now be configured via Hydra config files
2. **AdversarialLossCalculator**: Encapsulates all adversarial loss computation in a clean, modular class
3. **Model Save/Load**: Clean utilities to save and load `AudioVAE` models with all hyperparameters

### Benefits

- **Easier Experimentation**: Change discriminator architecture without modifying code
- **Cleaner Code**: Loss calculation logic is now centralized and easier to understand
- **Better Model Management**: Save and load models easily for inference or transfer learning
- **Backward Compatible**: Old checkpoints still work with legacy parameter support

---

## Discriminator Configuration

### Configuration File

The discriminator is now configured via `configs/model/discriminator/discriminator.yaml`:

```yaml
# Discriminator Configuration
_target_: src.model.discriminators.EncodecDiscriminator.EnCodecDiscriminator

# MS-STFT Discriminator parameters
stft_filters: 32
in_channels: 2  # Stereo audio
n_ffts: [2048, 1024, 512, 256, 128]  # Multi-scale FFT sizes

# Multi-Scale Discriminator (MSD) parameters
use_msd: true
msd_scales: [1, 2, 4]  # Waveform downsampling scales

# Multi-Period Discriminator (MPD) parameters
use_mpd: true
mpd_periods: [2, 3, 5, 7, 11]  # Periods for MPD
mpd_channels: [32, 128, 512, 1024, 1024]  # Channel progression
```

### Customizing the Discriminator

#### Disable Specific Discriminators

To disable MSD or MPD:

```yaml
use_msd: false  # Disable multi-scale discriminator
use_mpd: false  # Disable multi-period discriminator
```

#### Change FFT Sizes

```yaml
n_ffts: [4096, 2048, 1024, 512]  # Fewer, larger FFTs
```

#### Adjust Channel Capacity

```yaml
stft_filters: 64  # Increase discriminator capacity
```

### Using a Different Discriminator

You can easily swap in a different discriminator implementation:

1. Create your custom discriminator class
2. Update `_target_` in the config:

```yaml
_target_: src.model.discriminators.MyCustomDiscriminator
# ... your custom parameters ...
```

---

## Adversarial Loss Calculator

### Configuration File

Adversarial loss parameters are now in `configs/model/adversarial/adversarial_loss.yaml`:

```yaml
# Adversarial Loss Calculator Configuration
_target_: src.loss_fn.AdversarialLossCalculator.AdversarialLossCalculator

# Loss weights
adversarial_weight: 1.0        # Weight for generator adversarial loss
feature_matching_weight: 10.0  # Weight for feature matching loss

# Training schedule
warmup_steps: 0  # Steps before enabling adversarial training (0 = immediate)
```

### Key Parameters

- **adversarial_weight**: How much the generator cares about fooling the discriminator
  - Higher = stronger adversarial training
  - Typical range: 0.1 - 2.0

- **feature_matching_weight**: Weight for feature matching loss (perceptual loss)
  - Higher = generator matches discriminator features more closely
  - Typical range: 2.0 - 20.0

- **warmup_steps**: Number of training steps before enabling adversarial training
  - Useful for stable training (train VAE first, then add adversarial)
  - Set to 0 for immediate adversarial training
  - Typical values: 0, 5000, 10000

### Usage Examples

#### Gradual Adversarial Training

```yaml
# Train VAE for 10k steps, then enable adversarial
warmup_steps: 10000
adversarial_weight: 0.5  # Start with lower weight
```

#### Strong Adversarial Training

```yaml
warmup_steps: 0
adversarial_weight: 2.0  # Aggressive adversarial training
feature_matching_weight: 15.0
```

#### Focus on Perceptual Quality

```yaml
adversarial_weight: 0.5
feature_matching_weight: 20.0  # Emphasize feature matching
```

### API Usage

The `AdversarialLossCalculator` can also be used standalone:

```python
from src.loss_fn.AdversarialLossCalculator import AdversarialLossCalculator

loss_calc = AdversarialLossCalculator(
    adversarial_weight=1.0,
    feature_matching_weight=10.0,
    warmup_steps=5000
)

# Compute generator losses
gen_losses = loss_calc.compute_generator_loss(
    discriminator=disc,
    real_audio=real,
    fake_audio=fake,
    global_step=current_step
)

# Compute discriminator losses
disc_losses = loss_calc.compute_discriminator_loss(
    discriminator=disc,
    real_audio=real,
    fake_audio=fake
)
```

---

## Model Save/Load Utilities

### Quick Start

```python
from src.utils import save_model, load_model

# Save a model
model = AudioVAE(latent_dim=64, base_channels=128)
save_model(model, "my_model.pt")

# Load it back - no need to specify any parameters!
loaded_model = load_model("my_model.pt")
```

### API Reference

#### `save_model()`

```python
save_model(
    model: AudioVAE,
    save_path: str,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
)
```

**Example:**

```python
save_model(
    model,
    "checkpoints/model_epoch_50.pt",
    optimizer_state=optimizer.state_dict(),
    scheduler_state=scheduler.state_dict(),
    epoch=50,
    global_step=25000,
    metadata={"val_loss": 0.123, "notes": "Best model so far"}
)
```

#### `load_model()`

```python
load_model(
    checkpoint_path: str,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> AudioVAE
```

**Examples:**

```python
# Load on CPU
model = load_model("model.pt", map_location="cpu")

# Load on GPU
model = load_model("model.pt", map_location="cuda:0")

# Load with non-strict matching (useful for partial loading)
model = load_model("model.pt", strict=False)
```

#### `load_training_state()`

```python
load_training_state(
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]
```

**Example - Resume Training:**

```python
# Load model
model = load_model("checkpoint.pt")

# Create optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000)

# Load training state
state = load_training_state("checkpoint.pt", optimizer, scheduler)

# Resume from correct epoch
start_epoch = state["epoch"]
print(f"Resuming from epoch {start_epoch}")
```

### What Gets Saved?

The checkpoint includes:

1. **Model Architecture**: All hyperparameters needed to reconstruct the model
2. **Model Weights**: Complete state dict
3. **Latent Statistics**: Learned latent mean and std
4. **Training State** (optional):
   - Optimizer state
   - Scheduler state
   - Current epoch
   - Global step
5. **Metadata** (optional): Any custom information you want to save

### Checkpoint Structure

```python
{
    "model_state_dict": {...},  # PyTorch state dict
    "model_config": {
        "in_channels": 2,
        "base_channels": 128,
        "channel_mults": [1, 2, 4, 8],
        "strides": [2, 4, 4, 8],
        "latent_dim": 64,
        "kernel_size": 7,
        "dilations": [1, 3, 9]
    },
    "latent_statistics": {
        "latent_mean": Tensor(...),
        "latent_std": Tensor(...)
    },
    "encoder_frozen": False,
    "optimizer_state_dict": {...},  # Optional
    "scheduler_state_dict": {...},  # Optional
    "epoch": 50,                    # Optional
    "global_step": 25000,          # Optional
    "metadata": {...}              # Optional
}
```

### Converting Lightning Checkpoints

PyTorch Lightning saves checkpoints in its own format. You can convert them:

```python
from src.utils import load_from_lightning_checkpoint

# Load from Lightning checkpoint
model = load_from_lightning_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=99.ckpt"
)

# Save as standalone checkpoint
save_model(model, "standalone_model.pt")

# Now it can be loaded without Lightning
loaded = load_model("standalone_model.pt")
```

**Note:** This requires the Lightning checkpoint to have architecture config saved in hyperparameters.

---

## Migration Guide

### Updating Existing Code

#### 1. Update Config Files

**Old `audio_vae.yaml`:**

```yaml
use_adversarial: true
adversarial_warmup_steps: 0
adversarial_weight: 0.1
feature_matching_weight: 10.0
```

**New `audio_vae.yaml`:**

```yaml
defaults:
  - discriminator: discriminator/discriminator.yaml
  - adversarial_loss_calculator: adversarial/adversarial_loss.yaml

use_adversarial: true
```

#### 2. Legacy Parameter Support

The old parameters still work but will show a deprecation warning:

```python
# Still works, but deprecated
model = VAELightningModule(
    adversarial_warmup_steps=5000,  # DeprecationWarning
    adversarial_weight=0.1,          # DeprecationWarning
    feature_matching_weight=10.0,    # DeprecationWarning
)
```

**Recommended:** Use the new config-based approach instead.

#### 3. Saving Models from Lightning

Lightning checkpoints now automatically include model config:

```python
# This is done automatically in VAELightningModule.on_save_checkpoint()
# No code changes needed!
```

When you save a Lightning checkpoint, it now includes `model_config` which allows easy loading.

### Testing the Changes

```bash
# Test that training still works
python train.py trainer.max_epochs=1 trainer.limit_train_batches=10

# Test model save/load
python examples/model_save_load_example.py
```

---

## Examples

### Example 1: Training with Custom Discriminator

Create `configs/model/discriminator/lightweight.yaml`:

```yaml
_target_: src.model.discriminators.EncodecDiscriminator.EnCodecDiscriminator
stft_filters: 16  # Smaller discriminator
use_msd: false    # Disable MSD
use_mpd: true
mpd_periods: [2, 3, 5]  # Fewer periods
```

Train with it:

```bash
python train.py model.discriminator=discriminator/lightweight
```

### Example 2: Staged Adversarial Training

Create `configs/model/adversarial/staged.yaml`:

```yaml
_target_: src.loss_fn.AdversarialLossCalculator.AdversarialLossCalculator
adversarial_weight: 0.5
feature_matching_weight: 15.0
warmup_steps: 20000  # Train VAE for 20k steps first
```

Train with it:

```bash
python train.py model.adversarial_loss_calculator=adversarial/staged
```

### Example 3: Extract Model for Inference

```python
from src.utils import load_model
import torch

# Load trained model
model = load_model("lightning_logs/checkpoints/best.ckpt")
model.eval()
model = model.to("cuda")

# Use for inference
with torch.no_grad():
    audio = load_audio("input.wav")  # Your audio loading function
    recon, z, mean, logvar = model(audio)
    save_audio("output.wav", recon)  # Your audio saving function
```

---

## Troubleshooting

### Discriminator Not Found

**Error:** `discriminator must be provided when use_adversarial=True`

**Solution:** Add discriminator config to defaults:

```yaml
defaults:
  - discriminator: discriminator/discriminator.yaml
```

### Old Checkpoint Won't Load

**Error:** `Checkpoint does not contain 'model_config'`

**Solution:** This is an old-style checkpoint. Either:
1. Use the Lightning loading mechanism
2. Re-save using the new `save_model()` function

### Import Errors

**Error:** `Cannot import AdversarialLossCalculator`

**Solution:** Make sure you have the latest code:

```bash
git pull
# Check that these files exist:
ls src/loss_fn/AdversarialLossCalculator.py
ls src/utils/model_io.py
```

---

## Summary

The new system provides:

1. **Flexible Discriminator Configuration** - Easily experiment with different architectures
2. **Modular Loss Calculation** - Clean, maintainable adversarial loss logic
3. **Simple Model Management** - Save and load models with a single function call
4. **Backward Compatibility** - Existing code and checkpoints still work

For more examples, see:
- `examples/model_save_load_example.py` - Complete save/load examples
- `configs/model/discriminator/` - Discriminator configuration examples
- `configs/model/adversarial/` - Adversarial loss configuration examples
