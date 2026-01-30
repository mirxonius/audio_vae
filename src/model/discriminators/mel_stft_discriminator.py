"""
Multi-Scale Mel-STFT Discriminator.

Operates on mel spectrograms at multiple scales for perceptually-motivated
audio discrimination. Mel scaling better matches human auditory perception
compared to linear-frequency STFT discriminators.

Uses torchaudio.transforms.MelSpectrogram for robust mel spectrogram computation.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import MelSpectrogram
from einops import rearrange

from .utils import get_2d_padding


class DiscriminatorMelSTFT(nn.Module):
    """
    Single Mel-STFT discriminator operating at one scale.

    Computes mel spectrogram from audio using torchaudio and processes
    with 2D convolutions in time-frequency domain.
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        n_mels: int = 80,
        sample_rate: int = 44100,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        kernel_size: Tuple[int, int] = (3, 9),
        dilations: List[int] = [1, 2, 4],
        power: float = 1.0,
        normalized: bool = False,
        norm: Optional[str] = "slaney",
        mel_scale: str = "slaney",
        use_log: bool = True,
        log_eps: float = 1e-5,
    ):
        """
        Args:
            filters: Number of base filters
            in_channels: Number of audio input channels (1 for mono, 2 for stereo)
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT (default: n_fft)
            n_mels: Number of mel filterbank channels
            sample_rate: Audio sample rate for mel filterbank
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank (default: sample_rate / 2)
            kernel_size: Kernel size for 2D convolutions (freq, time)
            dilations: List of dilation rates in time dimension
            power: Exponent for the magnitude spectrogram (1.0 for energy, 2.0 for power)
            normalized: Whether to normalize mel filterbank weights
            norm: Normalization type for mel filterbank ('slaney' or None)
            mel_scale: Scale to use for mel filterbank ('htk' or 'slaney')
            use_log: Whether to use log mel spectrogram
            log_eps: Epsilon for log computation
        """
        super().__init__()
        self.in_channels = in_channels
        self.use_log = use_log
        self.log_eps = log_eps

        if win_length is None:
            win_length = n_fft

        # Use torchaudio's MelSpectrogram transform
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            norm=norm,
            mel_scale=mel_scale,
        )

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

    def _compute_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from audio using torchaudio.

        Args:
            x: [B, C, T] audio tensor

        Returns:
            mel: [B, C, n_mels, n_frames] mel spectrogram
        """
        B, C, T = x.shape

        # Flatten batch and channels for mel transform
        x_flat = x.reshape(B * C, T)

        # Compute mel spectrogram: [B*C, n_mels, n_frames]
        mel = self.mel_transform(x_flat)

        # Apply log compression if enabled
        if self.use_log:
            mel = torch.log(mel + self.log_eps)

        # Reshape back to [B, C, n_mels, n_frames]
        n_mels, n_frames = mel.shape[-2], mel.shape[-1]
        mel = mel.reshape(B, C, n_mels, n_frames)

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
        power: float = 1.0,
        normalized: bool = False,
        norm: Optional[str] = "slaney",
        mel_scale: str = "slaney",
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
            power: Exponent for the magnitude spectrogram
            normalized: Whether to normalize mel filterbank weights
            norm: Normalization type for mel filterbank
            mel_scale: Scale to use for mel filterbank
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
                    power=power,
                    normalized=normalized,
                    norm=norm,
                    mel_scale=mel_scale,
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
