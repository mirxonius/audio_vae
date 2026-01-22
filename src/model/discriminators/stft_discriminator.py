"""
Multi-Scale STFT Discriminator from EnCodec.

Based on "High Fidelity Neural Audio Compression" (arxiv 2210.13438)
Operates on complex-valued STFTs at multiple scales.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from einops import rearrange

from .utils import get_2d_padding


class DiscriminatorSTFT(nn.Module):
    """
    Single STFT-based discriminator operating at one scale.

    Processes complex-valued STFT (real and imaginary parts concatenated)
    with 2D convolutions in time-frequency domain.
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        kernel_size: Tuple[int, int] = (3, 9),
        dilations: List[int] = [1, 2, 4],
        normalized: bool = True,
    ):
        """
        Args:
            filters: Number of base filters
            in_channels: Number of audio input channels (1 for mono, 2 for stereo)
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            kernel_size: Kernel size for 2D convolutions (freq, time)
            dilations: List of dilation rates in time dimension
            normalized: Whether to normalize STFT by magnitude
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized

        # Register window buffer
        self.register_buffer("window", torch.hann_window(win_length))

        # Initial 2D conv: input has 2*in_channels (real + imag for each channel)
        self.conv_layers = nn.ModuleList()

        # First layer: kernel_size (3, 9) as in paper
        self.conv_layers.append(
            weight_norm(
                nn.Conv2d(
                    2 * in_channels,  # real + imaginary
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, C, T] audio tensor
        Returns:
            logits: [B, 1, F', T'] discriminator output
            fmap: List of feature maps from each layer
        """
        # Compute STFT for each channel
        # x: [B, C, T]
        B, C, T = x.shape

        # Reshape to process all channels together
        x_flat = x.reshape(B * C, T)

        # STFT computation
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=self.normalized,
            center=True,
        )
        # stft: [B*C, n_fft//2+1, n_frames]

        # Reshape back to separate channels
        stft = stft.reshape(B, C, self.n_fft // 2 + 1, -1)

        # Concatenate real and imaginary parts: [B, C, F, T] -> [B, 2*C, F, T]
        z = torch.cat([stft.real, stft.imag], dim=1)

        # Rearrange to [B, 2*C, T, F] for processing (time-frequency)
        z = rearrange(z, "b c f t -> b c t f")

        # Apply convolutions and collect features
        fmap = []
        for layer in self.conv_layers:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)

        # Final prediction
        logits = self.conv_post(z)

        return logits, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT Discriminator (MS-STFT) from EnCodec.

    Consists of multiple STFT discriminators operating at different scales.
    Paper uses window lengths: [2048, 1024, 512, 256, 128]
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 2,
        n_ffts: List[int] = [2048, 1024, 512, 256, 128],
        hop_lengths: List[int] = None,
        win_lengths: List[int] = None,
        normalized: bool = True,
    ):
        """
        Args:
            filters: Number of base filters for each discriminator
            in_channels: Number of audio channels (1=mono, 2=stereo)
            n_ffts: List of FFT sizes for each scale
            hop_lengths: List of hop lengths (default: n_fft // 4)
            win_lengths: List of window lengths (default: same as n_fft)
            normalized: Whether to normalize STFT
        """
        super().__init__()

        if hop_lengths is None:
            hop_lengths = [n // 4 for n in n_ffts]
        if win_lengths is None:
            win_lengths = n_ffts

        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)

        # For stereo, process L and R channels separately
        self.in_channels = in_channels

        # Create discriminators for each scale
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters=filters,
                    in_channels=1,  # Process each channel separately
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    normalized=normalized,
                )
                for n_fft, hop_length, win_length in zip(
                    n_ffts, hop_lengths, win_lengths
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
            logits_list: List of logits for each scale and channel
            fmaps_list: List of feature maps for each scale and channel
        """
        logits_list, fmaps_list = [], []

        # Process each channel separately (as in EnCodec paper)
        for channel_idx in range(self.in_channels):
            x_channel = x[:, channel_idx : channel_idx + 1, :]  # [B, 1, T]

            for disc in self.discriminators:
                logits, features = disc(x_channel)
                logits_list.append(logits)
                fmaps_list.append(features)

        return logits_list, fmaps_list
