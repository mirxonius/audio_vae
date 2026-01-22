"""
Streamlit App for Audio VAE Inference and Visualization

This app allows you to:
- Load trained VAE checkpoints from a dropdown menu
- Upload audio files for reconstruction
- Compare original vs reconstructed audio
- Visualize spectrograms and waveforms
"""

import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from omegaconf import OmegaConf, DictConfig
import mlflow
from mlflow.tracking import MlflowClient

from src.model.VAELightningModule import VAELightningModule

# Register OmegaConf types as safe for torch.load in PyTorch 2.6+
try:
    from torch.serialization import add_safe_globals

    add_safe_globals([DictConfig])
except (ImportError, AttributeError):
    # Older PyTorch versions don't have this
    pass


# Page config
st.set_page_config(page_title="Audio VAE Inference", page_icon="🎵", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load VAE model from checkpoint with multiple fallback strategies"""

    # Strategy 1: Try with weights_only=False (PyTorch 2.6+ compatible)
    try:
        st.info("Loading checkpoint (strategy 1: weights_only=False)...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Try to get config from checkpoint
        if "hyper_parameters" in checkpoint:
            cfg = checkpoint["hyper_parameters"].get("cfg", None)
            if cfg is not None:
                # Convert to OmegaConf if it's a dict
                if isinstance(cfg, dict):
                    cfg = OmegaConf.create(cfg)
                model = VAELightningModule(cfg)

                # Load state dict
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                # Move to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                st.success("✅ Model loaded successfully!")
                return model, device
    except Exception as e:
        st.warning(f"Strategy 1 failed: {str(e)}")

    # Strategy 2: Try with safe_globals context manager
    try:
        st.info("Loading checkpoint (strategy 2: safe_globals context)...")
        from torch.serialization import safe_globals

        with safe_globals([DictConfig]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "hyper_parameters" in checkpoint:
            cfg = checkpoint["hyper_parameters"].get("cfg", None)
            if cfg is not None:
                if isinstance(cfg, dict):
                    cfg = OmegaConf.create(cfg)
                model = VAELightningModule(cfg)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                st.success("✅ Model loaded successfully!")
                return model, device
    except Exception as e:
        st.warning(f"Strategy 2 failed: {str(e)}")

    # Strategy 3: Load with Lightning's built-in method
    try:
        st.info("Loading checkpoint (strategy 3: Lightning load_from_checkpoint)...")
        model = VAELightningModule.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        st.success("✅ Model loaded successfully!")
        return model, device
    except Exception as e:
        st.warning(f"Strategy 3 failed: {str(e)}")

    # All strategies failed
    st.error("❌ All loading strategies failed. Please check your checkpoint format.")
    st.error("Checkpoint should contain 'state_dict' and 'hyper_parameters' keys.")

    with st.expander("🔍 Debug Information"):
        try:
            # Try to inspect checkpoint structure
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            st.write("**Checkpoint keys:**")
            st.write(list(checkpoint.keys()))

            if "hyper_parameters" in checkpoint:
                st.write("**Hyperparameters keys:**")
                st.write(list(checkpoint["hyper_parameters"].keys()))
        except Exception as e:
            st.write(f"Could not inspect checkpoint: {str(e)}")

    return None, None


def find_checkpoints(checkpoint_dir: str = "checkpoints"):
    """Find all checkpoint files in the specified directory"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []

    # Find all .ckpt files recursively
    checkpoints = list(checkpoint_path.rglob("*.ckpt"))
    return sorted([str(p) for p in checkpoints])


def get_mlflow_runs(tracking_uri: str, experiment_name: str = "audio_vae"):
    """Get all runs from MLflow experiment"""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1000,
        )

        return runs
    except Exception as e:
        st.error(f"Error connecting to MLflow: {str(e)}")
        return []


def get_mlflow_model_artifacts(tracking_uri: str, run_id: str):
    """Get model artifacts from a specific run"""
    try:
        client = MlflowClient(tracking_uri=tracking_uri)

        # List all artifacts for this run
        artifacts = client.list_artifacts(run_id)

        # Recursively search for checkpoint files
        checkpoints = []

        def search_artifacts(path=""):
            """Recursively search for .ckpt files in artifacts"""
            items = client.list_artifacts(run_id, path)
            for item in items:
                if item.is_dir:
                    # Recursively search subdirectories
                    search_artifacts(item.path)
                elif item.path.endswith(".ckpt"):
                    checkpoints.append(item.path)

        # Start recursive search
        search_artifacts()

        return checkpoints
    except Exception as e:
        st.error(f"Error fetching artifacts: {str(e)}")
        return []


def download_mlflow_model(tracking_uri: str, run_id: str, artifact_path: str):
    """Download model from MLflow to a temporary location"""
    try:
        client = MlflowClient(tracking_uri=tracking_uri)

        # Download artifact to temp directory
        local_path = client.download_artifacts(run_id, artifact_path)

        return local_path
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None


def load_and_preprocess_audio(audio_file, target_sr=44100, target_length=262144):
    """Load and preprocess audio file"""
    # Save uploaded file temporarily with proper extension
    file_extension = Path(audio_file.name).suffix
    if not file_extension:
        file_extension = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        # Load audio - torchaudio handles mp4/m4a/aac automatically via ffmpeg
        waveform, sr = torchaudio.load(tmp_path)
    except Exception as e:
        # Fallback: try with backend specification
        try:
            # Set backend to ffmpeg for better format support
            torchaudio.set_audio_backend("ffmpeg")
            waveform, sr = torchaudio.load(tmp_path)
        except:
            raise Exception(f"Could not load audio file. Error: {str(e)}")

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad or trim to target length
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]

    return waveform, target_sr


def reconstruct_audio(model, waveform, device):
    """Reconstruct audio using VAE"""
    with torch.no_grad():
        # Add batch dimension and move to device
        x = waveform.unsqueeze(0).to(device)

        # Reconstruct
        x_recon, mu, logvar = model(x)

        # Move back to CPU
        x_recon = x_recon.squeeze(0).cpu()

    return x_recon


def plot_waveform(waveform, sr, title="Waveform"):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 3))

    waveform_np = waveform.squeeze().numpy()
    time = np.arange(len(waveform_np)) / sr

    ax.plot(time, waveform_np, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrogram(waveform, sr, title="Spectrogram"):
    """Plot audio spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Compute spectrogram
    n_fft = 2048
    hop_length = 512

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    )

    spec = spec_transform(waveform)
    spec_db = 10 * torch.log10(spec + 1e-10)

    # Plot
    im = ax.imshow(
        spec_db.squeeze().numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, waveform.shape[-1] / sr, 0, sr / 2],
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="dB")

    plt.tight_layout()
    return fig


def compute_metrics(original, reconstructed):
    """Compute reconstruction metrics"""
    # MSE
    mse = torch.mean((original - reconstructed) ** 2).item()

    # SNR
    signal_power = torch.mean(original**2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / noise_power).item()

    # PSNR
    max_val = max(original.abs().max(), reconstructed.abs().max())
    psnr = 20 * torch.log10(max_val / torch.sqrt(noise_power)).item()

    return {"MSE": mse, "SNR (dB)": snr, "PSNR (dB)": psnr}


# Main app
def main():
    st.title("🎵 Audio VAE Inference & Visualization")
    st.markdown("---")

    # Sidebar - Model selection
    st.sidebar.header("Model Selection")

    # Source selection: Local or MLflow
    model_source = st.sidebar.radio(
        "Model Source",
        options=["Local Checkpoints", "MLflow Tracking"],
        help="Choose where to load models from",
    )

    selected_checkpoint = None

    if model_source == "Local Checkpoints":
        # Local checkpoint loading
        checkpoint_dir = st.sidebar.text_input(
            "Checkpoint Directory",
            value="checkpoints",
            help="Directory containing .ckpt files",
        )

        # Find checkpoints
        checkpoints = find_checkpoints(checkpoint_dir)

        if not checkpoints:
            st.sidebar.warning(f"No checkpoints found in '{checkpoint_dir}'")
        else:
            # Checkpoint selection
            selected_checkpoint = st.sidebar.selectbox(
                "Select Checkpoint",
                options=checkpoints,
                format_func=lambda x: Path(x).name,
            )

            # Display checkpoint info
            st.sidebar.info(f"**Selected:**\n{Path(selected_checkpoint).name}")

    else:
        # MLflow model loading
        tracking_uri = st.sidebar.text_input(
            "MLflow Tracking URI",
            value="http://10.100.111.208:30500",
            help="MLflow tracking server URI",
        )

        experiment_name = st.sidebar.text_input(
            "Experiment Name", value="audio_vae", help="Name of the MLflow experiment"
        )

        # Fetch runs button
        if st.sidebar.button("🔍 Fetch Runs"):
            with st.spinner("Fetching MLflow runs..."):
                runs = get_mlflow_runs(tracking_uri, experiment_name)
                st.session_state.mlflow_runs = runs
                if runs:
                    st.sidebar.success(f"✅ Found {len(runs)} runs")
                else:
                    st.sidebar.warning("No runs found")

        # Display runs if available
        if "mlflow_runs" in st.session_state and st.session_state.mlflow_runs:
            runs = st.session_state.mlflow_runs

            # Create display options for runs
            run_options = []
            run_map = {}

            for run in runs:
                # Get run info
                run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
                metrics = run.data.metrics
                val_loss = metrics.get("val/loss", metrics.get("val_loss", None))

                # Create display string
                if val_loss is not None:
                    display = f"{run_name} (val_loss: {val_loss:.4f})"
                else:
                    display = f"{run_name}"

                run_options.append(display)
                run_map[display] = run

            # Select run
            selected_run_display = st.sidebar.selectbox(
                "Select Run", options=run_options
            )

            if selected_run_display:
                selected_run = run_map[selected_run_display]

                # Display run info
                with st.sidebar.expander("📊 Run Info"):
                    st.write(f"**Run ID:** {selected_run.info.run_id}")
                    st.write(f"**Status:** {selected_run.info.status}")
                    st.write(f"**Start Time:** {selected_run.info.start_time}")

                    if selected_run.data.metrics:
                        st.write("**Metrics:**")
                        for key, value in selected_run.data.metrics.items():
                            st.write(f"  - {key}: {value:.4f}")

                # Fetch artifacts for this run
                if st.sidebar.button("📦 Fetch Checkpoints"):
                    with st.spinner("Fetching checkpoints..."):
                        artifacts = get_mlflow_model_artifacts(
                            tracking_uri, selected_run.info.run_id
                        )
                        st.session_state.mlflow_artifacts = artifacts
                        st.session_state.mlflow_run_id = selected_run.info.run_id
                        if artifacts:
                            st.sidebar.success(f"✅ Found {len(artifacts)} checkpoints")
                        else:
                            st.sidebar.warning("No checkpoints found in this run")

                # Display artifacts if available
                if (
                    "mlflow_artifacts" in st.session_state
                    and st.session_state.mlflow_artifacts
                ):
                    artifacts = st.session_state.mlflow_artifacts

                    selected_artifact = st.sidebar.selectbox(
                        "Select Checkpoint",
                        options=artifacts,
                        format_func=lambda x: Path(x).name,
                    )

                    if selected_artifact:
                        st.sidebar.info(
                            f"**Selected:**\n{Path(selected_artifact).name}"
                        )

                        # Download and Load button (combined)
                        if st.sidebar.button("⬇️ Download & Load Model", type="primary"):
                            with st.spinner("Downloading checkpoint from MLflow..."):
                                local_path = download_mlflow_model(
                                    tracking_uri,
                                    st.session_state.mlflow_run_id,
                                    selected_artifact,
                                )
                                if local_path:
                                    st.sidebar.success("✅ Downloaded!")
                                    # Immediately load the model
                                    with st.spinner("Loading model..."):
                                        result = load_model(local_path)
                                        if result and result[0] is not None:
                                            (
                                                st.session_state.model,
                                                st.session_state.device,
                                            ) = result
                                            st.session_state.model_path = local_path
                                            st.sidebar.success(
                                                "✅ Model loaded successfully!"
                                            )
                                            # Force a rerun to show the upload section
                                            st.rerun()
                                        else:
                                            st.sidebar.error("❌ Failed to load model")
                                else:
                                    st.sidebar.error("❌ Download failed")

    # Load model button (for local checkpoints only)
    if model_source == "Local Checkpoints" and selected_checkpoint:
        if st.sidebar.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                result = load_model(selected_checkpoint)
                if result and result[0] is not None:
                    st.session_state.model, st.session_state.device = result
                    st.session_state.model_path = selected_checkpoint
                    st.sidebar.success("✅ Model loaded successfully!")
                    # Force a rerun to show the upload section
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to load model")

    # Display model info if loaded
    if "model" in st.session_state:
        model = st.session_state.model
        device = st.session_state.device

        with st.sidebar.expander("ℹ️ Model Info"):
            num_params = sum(p.numel() for p in model.parameters())
            st.write(f"**Parameters:** {num_params:,} ({num_params/1e6:.2f}M)")
            st.write(f"**Device:** {device}")
            if hasattr(model, "vae") and hasattr(model.vae, "latent_dim"):
                st.write(f"**Latent Dim:** {model.vae.latent_dim}")
            if "model_path" in st.session_state:
                st.write(f"**Source:** {Path(st.session_state.model_path).name}")

    # Main content
    if "model" not in st.session_state:
        if model_source == "Local Checkpoints" and not find_checkpoints(checkpoint_dir):
            st.info("👈 Please specify a valid checkpoint directory in the sidebar")
        else:
            st.info("👈 Please load a model from the sidebar to begin")
        return

    # Audio upload
    st.header("📤 Upload Audio")
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg", "mp4", "m4a", "aac"],
        help="Upload an audio file to reconstruct",
    )

    if audio_file is None:
        st.info("Please upload an audio file to continue")
        return

    # Processing parameters
    col1, col2 = st.columns(2)
    with col1:
        target_length = st.number_input(
            "Target Length (samples)",
            min_value=16384,
            max_value=524288,
            value=262144,
            step=16384,
            help="Length to pad/trim audio to",
        )
    with col2:
        target_sr = st.number_input(
            "Sample Rate (Hz)",
            min_value=16000,
            max_value=48000,
            value=44100,
            step=100,
            help="Target sample rate",
        )

    # Process button
    if st.button("🎯 Reconstruct Audio", type="primary"):
        with st.spinner("Processing audio..."):
            try:
                # Load and preprocess
                waveform, sr = load_and_preprocess_audio(
                    audio_file, target_sr=target_sr, target_length=target_length
                )

                # Reconstruct
                reconstructed = reconstruct_audio(
                    st.session_state.model, waveform, st.session_state.device
                )

                # Store in session state
                st.session_state.original = waveform
                st.session_state.reconstructed = reconstructed
                st.session_state.sr = sr

                st.success("✅ Reconstruction complete!")

            except Exception as e:
                st.error(f"Error during reconstruction: {str(e)}")
                return

    # Display results
    if "original" in st.session_state and "reconstructed" in st.session_state:
        st.markdown("---")
        st.header("📊 Results")

        original = st.session_state.original
        reconstructed = st.session_state.reconstructed
        sr = st.session_state.sr

        # Metrics
        st.subheader("📈 Reconstruction Metrics")
        metrics = compute_metrics(original, reconstructed)

        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{metrics['MSE']:.6f}")
        col2.metric("SNR", f"{metrics['SNR (dB)']:.2f} dB")
        col3.metric("PSNR", f"{metrics['PSNR (dB)']:.2f} dB")

        st.markdown("---")

        # Audio players
        st.subheader("🔊 Audio Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Audio**")
            st.audio(original.squeeze().numpy(), sample_rate=sr, format="audio/wav")

        with col2:
            st.write("**Reconstructed Audio**")
            st.audio(
                reconstructed.squeeze().numpy(), sample_rate=sr, format="audio/wav"
            )

        st.markdown("---")

        # Waveform plots
        st.subheader("📉 Waveform Comparison")

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_waveform(original, sr, "Original Waveform")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig = plot_waveform(reconstructed, sr, "Reconstructed Waveform")
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # Spectrogram plots
        st.subheader("🎨 Spectrogram Comparison")

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_spectrogram(original, sr, "Original Spectrogram")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig = plot_spectrogram(reconstructed, sr, "Reconstructed Spectrogram")
            st.pyplot(fig)
            plt.close()

        # Difference plots
        st.markdown("---")
        st.subheader("🔍 Reconstruction Error")

        difference = original - reconstructed

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_waveform(difference, sr, "Difference (Original - Reconstructed)")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig = plot_spectrogram(difference, sr, "Difference Spectrogram")
            st.pyplot(fig)
            plt.close()

        # Download reconstructed audio
        st.markdown("---")
        st.subheader("💾 Download Reconstructed Audio")

        # Convert to numpy for download
        reconstructed_np = reconstructed.squeeze().numpy()

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            torchaudio.save(tmp_file.name, reconstructed.unsqueeze(0), sr, format="wav")

            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()

            st.download_button(
                label="📥 Download Reconstructed Audio",
                data=audio_bytes,
                file_name="reconstructed_audio.wav",
                mime="audio/wav",
            )


if __name__ == "__main__":
    main()
