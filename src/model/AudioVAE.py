from typing import List, Tuple
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from src.model.Encoder import Encoder
from src.model.Decoder import Decoder


class AudioVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        strides: List[int] = [2, 4, 8, 8],
        latent_dim: int = 64,
        kernel_size: int = 7,
        dilations: List[int] = [1, 3, 9],
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            base_channels,
            channel_mults,
            strides,
            latent_dim,
            kernel_size,
            dilations,
        )
        self.decoder = Decoder(
            in_channels,
            base_channels,
            channel_mults,
            strides,
            latent_dim,
            kernel_size,
            dilations,
        )

        self.latent_dim = latent_dim
        self.hop_length = math.prod(strides)
        self.latent_mean = nn.Parameter(torch.zeros(1, latent_dim, 1))
        self.latent_std = nn.Parameter(torch.ones(1, latent_dim, 1))
        self.encoder_frozen = False

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return (z - self.latent_mean) / (self.latent_std + 1e-8)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.encoder_frozen = True

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-20, max=20)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z * (self.latent_std + 1e-8) + self.latent_mean
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(not self.encoder_frozen):
            z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        if recon.shape[-1] != x.shape[-1]:
            recon = F.pad(recon, (0, x.shape[-1] - recon.shape[-1]))
        return recon, z, mean, logvar
