import torch
import numpy as np
import musdb
import random
import librosa
from torch.utils.data import Dataset


class MUSDB18Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 44100,
        chunk_duration: float = 6.0,
        channels: int = 2,
        num_samples: int = 1000,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.channels = channels
        self.num_samples = num_samples
        # Define the available stems in MUSDB18
        self.stems = ["vocals", "drums", "bass", "other"]

        # Load MUSDB18 based on split
        if split == "train":
            self.mus = musdb.DB(root=root, subsets="train", split="train")
        elif split == "valid":
            self.mus = musdb.DB(root=root, subsets="train", split="valid")
        elif split == "test":
            self.mus = musdb.DB(root=root, subsets="test")
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'valid', or 'test'"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Pick a random track regardless of idx
        track = random.choice(self.mus.tracks)

        # Randomly pick Full Mix or a Random Stem (25% full mix, 75% random stem)
        if random.random() < 0.25:
            # Take the full mixture
            audio = track.audio
        else:
            # Take a random stem
            target_stem = random.choice(self.stems)
            audio = track.targets[target_stem].audio

        if self.sample_rate != 44100:
            audio = librosa.resample(y=audio, orig_sr=44100, target_sr=self.sample_rate)
        # Random cropping

        max_start = max(0, len(audio) - self.chunk_samples)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        chunk = audio[start : start + self.chunk_samples]

        # Padding if the track is shorter than chunk_samples
        if len(chunk) < self.chunk_samples:
            chunk = np.pad(chunk, ((0, self.chunk_samples - len(chunk)), (0, 0)))

        # Convert to Tensor (Channels, Time)
        chunk = torch.from_numpy(chunk.T).float()

        # Mono conversion if requested
        if self.channels == 1:
            chunk = chunk.mean(dim=0, keepdim=True)

        # Peak normalization
        peak = chunk.abs().max()
        if peak > 0:
            chunk = chunk / peak

        return chunk
