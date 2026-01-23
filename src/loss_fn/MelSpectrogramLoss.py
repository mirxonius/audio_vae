import torch
from typing import List, Tuple, Optional
from auraloss.freq import MelSTFTLoss
from src.loss_fn.BaseLoss import BaseLoss


class MultiScaleMelSpectrogramLoss(BaseLoss):
    def __init__(
        self,
        sample_rate: int,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        n_mels: List[int] = [64, 128, 256],
        weight: float = 1.0,
        name: str = "multi_scale_mel_loss",
    ):
        super().__init__(weight, name)

        # Defaults: hop = fft//4, win = fft
        if hop_sizes is None:
            hop_sizes = [f // 4 for f in fft_sizes]
        if win_lengths is None:
            win_lengths = fft_sizes

        assert (
            len(fft_sizes) == len(hop_sizes) == len(win_lengths) == len(n_mels)
        ), "All parameter lists must have the same length"

        self.losses = torch.nn.ModuleList(
            [
                MelSTFTLoss(
                    sample_rate=sample_rate,
                    fft_size=fft,
                    hop_size=hop,
                    win_length=win,
                    n_mels=mels,
                    w_log_mag=1.0,
                    # w_lin_mag=1.0,
                )
                for fft, hop, win, mels in zip(
                    fft_sizes, hop_sizes, win_lengths, n_mels
                )
            ]
        )

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss = total_loss + loss_fn(x_pred, x_true)
        return total_loss / len(self.losses)
