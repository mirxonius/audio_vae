import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from typing import List, Tuple, Optional


def get_norm(layer, norm_type: str):
    match norm_type:
        case "weight_norm":
            return weight_norm(layer)
        case "spectral_norm":
            return spectral_norm(layer)
        case None:
            return layer
        case _:
            raise ValueError(f"Norm type {norm_type} not recognised!")


class ConvolutionalDiscriminator(nn.Module):
    """
    Single-band convolutional discriminator using 2D convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        norm_type: str = "weight_norm",
    ):
        super().__init__()

        # Layer configurations: (out_channels, kernel, stride, padding)
        layer_configs = [
            (base_channels, 7, 2, 3),
            (base_channels * 2, 5, 2, 2),
            (base_channels * 4, 5, 2, 2),
            (base_channels * 8, 5, 2, 2),
            (base_channels * 8, 3, 1, 1),
        ]

        self.convs = nn.ModuleList()
        curr_channels = in_channels

        for out_ch, k, s, p in layer_configs:
            self.convs.append(
                nn.Sequential(
                    get_norm(nn.Conv2d(curr_channels, out_ch, k, s, p), norm_type),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            curr_channels = out_ch

        self.final_conv = get_norm(
            nn.Conv2d(curr_channels, 1, kernel_size=3, padding=1), norm_type
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmaps = []
        for layer in self.convs:
            x = layer(x)
            fmaps.append(x)

        x = self.final_conv(x)
        fmaps.append(x)

        return torch.flatten(x, 1).mean(1, keepdim=True), fmaps


class BandSplitDiscriminator(nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        band_ranges: Optional[List[Tuple[int, int]]] = None,
        norm_type: str = "weight_norm",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Default bands: Sub-bass, Bass-mid, Mid-high, High
        self.band_ranges = band_ranges or [(0, 64), (64, 256), (256, 512), (512, 1025)]

        # Register Hann window as a buffer to handle device placement automatically
        self.register_buffer("window", torch.hann_window(n_fft))

        self.band_discriminators = nn.ModuleList(
            [
                ConvolutionalDiscriminator(1, norm_type=norm_type)
                for _ in range(len(self.band_ranges))
            ]
        )

    def compute_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Converts waveform to magnitude spectrogram."""
        if audio.dim() == 3:  # (B, 1, T) -> (B, T)
            audio = audio.squeeze(1)

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        return torch.abs(spec)

    def forward(
        self, audio: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        spec = self.compute_spectrogram(audio)  # (B, F, T)

        all_logits = []
        all_fmaps = []

        for i, (start, end) in enumerate(self.band_ranges):
            # Extract band and add channel dim: (B, 1, F_band, T)
            band_input = spec[:, start:end, :].unsqueeze(1)

            logit, fmaps = self.band_discriminators[i](band_input)
            all_logits.append(logit)
            all_fmaps.append(fmaps)

        return all_logits, all_fmaps


if __name__ == "__main__":
    print("=" * 80)
    print("Band Split Discriminator Test")
    print("=" * 80)

    # Configuration
    batch_size = 2
    sample_rate = 44100
    duration = 1.5  # seconds
    num_channels = 2

    # Create random stereo audio
    audio = torch.randn(batch_size, num_channels, int(sample_rate * duration))
    print(f"\n📊 Input Audio Shape: {audio.shape}")
    print(
        f"   Batch: {batch_size}, Channels: {num_channels}, Samples: {audio.shape[-1]}"
    )
    print(f"   Duration: {duration}s @ {sample_rate}Hz")

    # Initialize discriminator
    print("\n" + "=" * 80)
    print("Initializing Band Split Discriminator")
    print("=" * 80)

    discriminator = BandSplitDiscriminator(
        n_fft=2048, hop_length=512, norm_type="weight_norm"
    )

    print(f"\n🔧 Configuration:")
    print(f"   FFT Size: {discriminator.n_fft}")
    print(f"   Hop Length: {discriminator.hop_length}")
    print(f"   Number of Bands: {len(discriminator.band_ranges)}")
    print(f"\n📻 Band Ranges (frequency bins):")
    for i, (start, end) in enumerate(discriminator.band_ranges):
        freq_start = start * sample_rate / discriminator.n_fft
        freq_end = end * sample_rate / discriminator.n_fft
        print(
            f"   Band {i}: bins [{start:4d}, {end:4d}) → {freq_start:7.1f} - {freq_end:7.1f} Hz"
        )

    # Process each channel separately (as discriminator expects mono)
    print("\n" + "=" * 80)
    print("Forward Pass Analysis")
    print("=" * 80)

    for ch in range(num_channels):
        print(f"\n🎵 Processing Channel {ch}:")
        print("-" * 80)

        # Extract mono channel
        mono_audio = audio[:, ch : ch + 1, :]  # Keep channel dim: (B, 1, T)
        print(f"   Input shape: {mono_audio.shape}")

        # Forward pass
        with torch.no_grad():
            logits, feature_maps = discriminator(mono_audio)

        print(f"\n   📈 Logits (discrimination scores):")
        print(f"      Number of bands: {len(logits)}")
        for i, logit in enumerate(logits):
            print(
                f"      Band {i}: shape={logit.shape}, mean={logit.mean().item():.4f}, std={logit.std().item():.4f}"
            )

        print(f"\n   🗺️  Feature Maps:")
        print(f"      Number of discriminators: {len(feature_maps)}")
        for band_idx, fmaps in enumerate(feature_maps):
            print(f"\n      Band {band_idx} ({len(fmaps)} layers):")
            for layer_idx, fmap in enumerate(fmaps):
                print(f"         Layer {layer_idx}: {fmap.shape}")

    # Test spectrogram computation
    print("\n" + "=" * 80)
    print("Spectrogram Analysis")
    print("=" * 80)

    with torch.no_grad():
        spec = discriminator.compute_spectrogram(audio[:, 0:1, :])

    print(f"\n📊 Spectrogram Shape: {spec.shape}")
    print(f"   Format: (Batch, Frequency_bins, Time_frames)")
    print(
        f"   Frequency bins: {spec.shape[1]} (should be n_fft//2 + 1 = {discriminator.n_fft//2 + 1})"
    )
    print(f"   Time frames: {spec.shape[2]}")
    expected_frames = (
        1 + (audio.shape[-1] - discriminator.n_fft) // discriminator.hop_length
    )
    print(f"   Expected frames: ~{expected_frames}")
    print(f"\n   Magnitude stats:")
    print(f"      Min: {spec.min().item():.6f}")
    print(f"      Max: {spec.max().item():.6f}")
    print(f"      Mean: {spec.mean().item():.6f}")
    print(f"      Std: {spec.std().item():.6f}")

    # Model parameters
    print("\n" + "=" * 80)
    print("Model Statistics")
    print("=" * 80)

    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(
        p.numel() for p in discriminator.parameters() if p.requires_grad
    )

    print(f"\n📊 Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Per band: ~{total_params // len(discriminator.band_ranges):,}")

    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
