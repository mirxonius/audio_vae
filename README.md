# Audio VAE

A PyTorch Lightning implementation of an Encodec-style Audio Variational Autoencoder (VAE) for high-quality audio reconstruction and compression. This project combines reconstruction losses, perceptual losses, and optional adversarial training for state-of-the-art audio generation.

## Features

- **Variational Autoencoder Architecture**: Encoder-decoder architecture with latent space compression
- **Multi-Resolution Reconstruction Losses**:
  - Multi-resolution STFT loss
  - Mel-spectrogram loss
  - Waveform L1 loss
  - Dual-channel (Mid-Side + Left-Right) weighted loss
- **Adversarial Training** (optional):
  - Multi-scale discriminators
  - STFT discriminators
  - Period discriminators
  - Band discriminators
  - Feature matching loss
- **KL Divergence Loss**: With warmup scheduling and free bits
- **MLflow Integration**: Experiment tracking and model versioning
- **PyTorch Lightning**: Modular training with callbacks and loggers
- **Hydra Configuration**: Flexible configuration management

## Requirements

```bash
# Core dependencies
pytorch>=2.0.0
pytorch-lightning>=2.0.0
hydra-core>=1.3.0
mlflow>=2.0.0
torchaudio>=2.0.0
librosa
musdb
streamlit  # For inference UI
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd audio_vae
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # If you have one
# Or install manually:
pip install torch pytorch-lightning hydra-core mlflow torchaudio librosa musdb streamlit
```

3. Download the MUSDB18 dataset:
```bash
# The dataset will be automatically downloaded by the musdb library
# Or manually download from: https://sigsep.github.io/datasets/musdb.html
```

4. Configure paths:
   - Edit `configs/datamodule/musdb18.yaml` to set your dataset path
   - Edit `configs/logger/mlflow.yaml` to set your MLflow tracking server (or use local tracking)

## Project Structure

```
audio_vae/
├── configs/              # Hydra configuration files
│   ├── train.yaml       # Main training configuration
│   ├── callbacks/       # Training callbacks (checkpointing, logging, etc.)
│   ├── datamodule/      # Dataset configurations
│   ├── logger/          # Logger configurations (MLflow)
│   ├── model/           # Model architecture and hyperparameters
│   │   ├── architecture/    # Encoder/decoder architecture
│   │   ├── loss_calculator/ # Loss function configuration
│   │   ├── optimizer/       # Optimizer settings
│   │   └── scheduler/       # Learning rate scheduler
│   ├── paths/           # Directory paths
│   └── trainer/         # PyTorch Lightning Trainer settings
├── src/
│   ├── model/           # VAE model implementation
│   │   ├── VAELightningModule.py  # Main Lightning module
│   │   ├── AudioVAE.py            # VAE architecture
│   │   ├── Encoder.py             # Audio encoder
│   │   ├── Decoder.py             # Audio decoder
│   │   └── discriminators/        # Adversarial discriminators
│   ├── loss_fn/         # Loss function implementations
│   ├── dataset/         # Dataset and datamodule
│   └── callbacks/       # Custom training callbacks
├── train.py             # Main training script
└── vae_inference.py     # Streamlit inference UI
```

## Quick Start

### Training

Basic training with default configuration:
```bash
python train.py
```

Override specific parameters:
```bash
# Change experiment and run names
python train.py experiment_name=my_experiment run_name=my_run

# Use different model configuration
python train.py model=audio_vae

# Change training parameters
python train.py trainer.max_epochs=100 trainer.devices=[0,1]

# Resume from checkpoint
python train.py ckpt_path=/path/to/checkpoint.ckpt
```

### Inference

Launch the Streamlit inference UI:
```bash
streamlit run vae_inference.py
```

The UI allows you to:
- Load trained model checkpoints
- Upload audio files for reconstruction
- Visualize original vs reconstructed audio
- Adjust latent space representations

## Configuration Guide

The project uses [Hydra](https://hydra.cc/) for configuration management. All configurations are in the `configs/` directory.

### Main Configuration (`configs/train.yaml`)

```yaml
experiment_name: audio_vae  # MLflow experiment name
run_name: big_model         # MLflow run name
seed: 2211                  # Random seed
sample_rate: 44100          # Audio sample rate
```

### Model Configuration (`configs/model/audio_vae.yaml`)

Key parameters:
- **Architecture**: Channel dimensions, strides, residual blocks
- **Adversarial training**: Enable/disable, warmup steps, loss weights
- **Discriminator**: Learning rate, update frequency
- **Logging**: Audio logging frequency, number of samples

### Dataset Configuration (`configs/datamodule/musdb18.yaml`)

```yaml
root: "/path/to/musdb18"    # Dataset path
sample_rate: 44100          # Audio sample rate
chunk_duration: 1.5         # Chunk length in seconds
batch_size: 8               # Training batch size
train_samples: 8000         # Samples per training epoch
val_samples: 800            # Samples per validation epoch
```

### Trainer Configuration (`configs/trainer/default.yaml`)

PyTorch Lightning Trainer settings:
- `max_epochs`: Maximum training epochs
- `devices`: GPU device IDs
- `precision`: Training precision (16-mixed, 32, bf16-mixed)
- `gradient_clip_val`: Gradient clipping threshold

### Callbacks Configuration (`configs/callbacks/`)

Available callbacks:
- **model_checkpoint**: Save best models
- **early_stopping**: Stop training when validation loss plateaus (commented out by default)
- **learning_rate_monitor**: Track learning rate
- **model_saving_callback**: Custom model saving logic
- **audio_reconstruction_logging**: Log audio samples to MLflow
- **mlflow_model_tracking**: Track models in MLflow Model Registry (commented out by default)

## Model Architecture

### Encoder
- Convolutional layers with residual blocks
- Progressive downsampling via strided convolutions
- Outputs mean and log-variance for latent distribution

### Latent Space
- Variational bottleneck with KL divergence regularization
- Dimensionality controlled by `latent_dim` parameter
- Reparameterization trick for gradient flow

### Decoder
- Transposed convolutional layers with residual blocks
- Progressive upsampling
- Reconstructs waveform from latent representation

### Discriminators (Optional)
When adversarial training is enabled:
- Multi-scale discriminators at different resolutions
- STFT-based discriminators for frequency domain
- Period discriminators for temporal patterns
- Band discriminators for frequency band specificity

## Loss Functions

1. **Reconstruction Losses**:
   - Multi-resolution STFT loss: Frequency domain reconstruction
   - Mel-spectrogram loss: Perceptual reconstruction
   - Waveform L1 loss: Time domain reconstruction

2. **KL Divergence Loss**:
   - Regularizes latent space to match prior distribution
   - Supports warmup scheduling
   - Free bits threshold to prevent posterior collapse

3. **Adversarial Losses** (optional):
   - Generator adversarial loss
   - Feature matching loss
   - Separate discriminator training

## Training Tips

1. **Start without adversarial training**: Train with reconstruction + KL loss first
2. **Use gradient clipping**: Prevents exploding gradients in audio models
3. **Monitor audio samples**: Use the audio reconstruction callback to listen to outputs
4. **Adjust KL weight**: Balance between reconstruction quality and latent regularization
5. **Use mixed precision**: 16-mixed precision significantly speeds up training

## Monitoring with MLflow

All experiments are tracked with MLflow:
- Metrics: Training/validation losses, KL divergence, reconstruction errors
- Parameters: All Hydra configuration parameters
- Artifacts: Audio samples, model checkpoints
- Models: Registered models for deployment

Access the MLflow UI:
```bash
mlflow ui --port 5000
# Visit http://localhost:5000
```

Or configure a remote MLflow tracking server in `configs/logger/mlflow.yaml`.

## Common Issues

### CUDA Out of Memory
- Reduce `batch_size` in `configs/datamodule/musdb18.yaml`
- Reduce `chunk_duration` to process shorter audio segments
- Use gradient accumulation: `trainer.accumulate_grad_batches=2`

### Posterior Collapse (KL → 0)
- Reduce `kl_weight` in model configuration
- Increase `kl_warmup_steps` for slower KL weight increase
- Use `free_bits` threshold

### Poor Reconstruction Quality
- Increase model capacity: larger `base_channels` or deeper network
- Adjust loss weights to emphasize reconstruction
- Train longer before enabling adversarial losses

## Citation

If you use this code, please cite the relevant papers:
- Encodec: [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## License

[Specify your license here]

## Acknowledgments

- Built with [PyTorch Lightning](https://lightning.ai/)
- Configuration management via [Hydra](https://hydra.cc/)
- Experiment tracking with [MLflow](https://mlflow.org/)
- Trained on [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset
