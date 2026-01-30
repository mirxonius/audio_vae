"""
Multi-Scale Mel-STFT Discriminator.

Operates on mel spectrograms at multiple scales for perceptually-motivated
audio discrimination. Mel scaling better matches human auditory perception
compared to linear-frequency STFT discriminators.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from einops import rearrange

from .utils import get_2d_padding


class DiscriminatorMelSTFT(nn.Module):
    """
    Single Mel-STFT discriminator operating at one scale.

    Computes mel spectrogram from audio and processes with 2D convolutions
    in time-frequency domain.
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        sample_rate: int = 44100,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        kernel_size: Tuple[int, int] = (3, 9),
        dilations: List[int] = [1, 2, 4],
        use_log: bool = True,
        log_eps: float = 1e-5,
    ):
        """
        Args:
            filters: Number of base filters
            in_channels: Number of audio input channels (1 for mono, 2 for stereo)
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            n_mels: Number of mel filterbank channels
            sample_rate: Audio sample rate for mel filterbank
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank (default: sample_rate / 2)
            kernel_size: Kernel size for 2D convolutions (freq, time)
            dilations: List of dilation rates in time dimension
            use_log: Whether to use log mel spectrogram
            log_eps: Epsilon for log computation
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.use_log = use_log
        self.log_eps = log_eps

        # Register window buffer
        self.register_buffer("window", torch.hann_window(win_length))

        # Create mel filterbank
        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_filterbank", mel_fb)

        # Convolution layers
        self.conv_layers = nn.ModuleList()

        # First layer: processes mel spectrogram (in_channels per audio channel)
        self.conv_layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    filters,
                    kernel_size=kernel_size,
                    padding=get_2d_padding(kernel_size),
                )
            )
        )

        # Dilated convolutions with stride in frequency dimension
        in_chs = filters
        for dilation in dilations:
            out_chs = min(filters * 4, 512)
            self.conv_layers.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_size,
                        stride=(2, 1),  # Downsample frequency, keep time
                        dilation=(1, dilation),  # Dilate in time dimension
                        padding=get_2d_padding(kernel_size, (1, dilation)),
                    )
                )
            )
            in_chs = out_chs

        # Final prediction layer
        self.conv_post = weight_norm(
            nn.Conv2d(in_chs, 1, kernel_size=(3, 3), padding=get_2d_padding((3, 3)))
        )

        self.activation = nn.LeakyReLU(0.1)

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Number of frequency bins in STFT
        n_freqs = self.n_fft // 2 + 1

        # Compute mel points
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Convert to FFT bin indices
        bin_points = torch.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).long()

        # Create filterbank
        filterbank = torch.zeros(self.n_mels, n_freqs)

        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    @staticmethod
    def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        """Convert Hz to mel scale."""
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        """Convert mel scale to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _compute_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from audio.

        Args:
            x: [B, C, T] audio tensor

        Returns:
            mel: [B, C, n_mels, n_frames] mel spectrogram
        """
        B, C, T = x.shape

        # Flatten batch and channels for STFT
        x_flat = x.reshape(B * C, T)

        # Compute STFT
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        # stft: [B*C, n_fft//2+1, n_frames]

        # Compute magnitude spectrogram
        magnitude = torch.abs(stft)

        # Apply mel filterbank: [n_mels, n_freqs] @ [B*C, n_freqs, n_frames]
        # Transpose to [B*C, n_frames, n_freqs], apply filterbank, transpose back
        mel = torch.matmul(self.mel_filterbank, magnitude)
        # mel: [B*C, n_mels, n_frames]

        # Apply log compression if enabled
        if self.use_log:
            mel = torch.log(mel + self.log_eps)

        # Reshape back to [B, C, n_mels, n_frames]
        n_frames = mel.shape[-1]
        mel = mel.reshape(B, C, self.n_mels, n_frames)

        return mel

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, C, T] audio tensor

        Returns:
            logits: [B, 1, F', T'] discriminator output
            fmap: List of feature maps from each layer
        """
        # Compute mel spectrogram: [B, C, n_mels, n_frames]
        mel = self._compute_mel_spectrogram(x)

        # Rearrange to [B, C, T, F] for processing (time-frequency like STFT disc)
        z = rearrange(mel, "b c f t -> b c t f")

        # Apply convolutions and collect features
        fmap = []
        for layer in self.conv_layers:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)

        # Final prediction
        logits = self.conv_post(z)

        return logits, fmap


class MultiScaleMelSTFTDiscriminator(nn.Module):
    """
    Multi-Scale Mel-STFT Discriminator.

    Consists of multiple Mel-STFT discriminators operating at different scales,
    providing perceptually-motivated multi-resolution discrimination.
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 2,
        n_ffts: List[int] = [2048, 1024, 512, 256, 128],
        hop_lengths: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        n_mels: List[int] = [128, 128, 80, 64, 32],
        sample_rate: int = 44100,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        use_log: bool = True,
    ):
        """
        Args:
            filters: Number of base filters for each discriminator
            in_channels: Number of audio channels (1=mono, 2=stereo)
            n_ffts: List of FFT sizes for each scale
            hop_lengths: List of hop lengths (default: n_fft // 4)
            win_lengths: List of window lengths (default: same as n_fft)
            n_mels: List of mel bins for each scale
            sample_rate: Audio sample rate
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank
            use_log: Whether to use log mel spectrogram
        """
        super().__init__()

        if hop_lengths is None:
            hop_lengths = [n // 4 for n in n_ffts]
        if win_lengths is None:
            win_lengths = n_ffts

        # Handle single n_mels value
        if isinstance(n_mels, int):
            n_mels = [n_mels] * len(n_ffts)

        assert len(n_ffts) == len(hop_lengths) == len(win_lengths) == len(n_mels)

        self.in_channels = in_channels

        # Create discriminators for each scale
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorMelSTFT(
                    filters=filters,
                    in_channels=in_channels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mels=n_mel,
                    sample_rate=sample_rate,
                    f_min=f_min,
                    f_max=f_max,
                    use_log=use_log,
                )
                for n_fft, hop_length, win_length, n_mel in zip(
                    n_ffts, hop_lengths, win_lengths, n_mels
                )
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, C, T] audio tensor

        Returns:
            logits_list: List of logits for each scale
            fmaps_list: List of feature maps for each scale
        """
        logits_list, fmaps_list = [], []

        for disc in self.discriminators:
            logits, features = disc(x)
            logits_list.append(logits)
            fmaps_list.append(features)

        return logits_list, fmaps_list
