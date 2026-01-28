import torch
import torchaudio
import random
from pathlib import Path
from torch.utils.data import Dataset


GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GTZANDataset(Dataset):
    """
    PyTorch Dataset for the GTZAN Music Genre Classification dataset,
    adapted for audio autoencoder training.

    The GTZAN dataset contains 1000 audio tracks (30 seconds each),
    split across 10 genres with 100 tracks per genre. Original sample
    rate is 22050 Hz, mono.

    Expected directory structure:
        root/
            genres/              (or genres_original/)
                blues/
                    blues.00000.wav
                    blues.00001.wav
                    ...
                classical/
                    classical.00000.wav
                    ...
                ...

    For autoencoder training, genre labels are not used — the dataset
    returns random audio chunks for reconstruction.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 22050,
        chunk_duration: float = 6.0,
        channels: int = 1,
        num_samples: int = 1000,
    ):
        """
        Args:
            root: Path to the GTZAN dataset root directory.
            split: One of 'train', 'valid', or 'test'.
                   Split is deterministic per genre (80/10/10).
            sample_rate: Target audio sample rate.
            chunk_duration: Duration of audio chunks in seconds.
            channels: Number of output audio channels (1=mono, 2=stereo).
            num_samples: Number of samples per virtual epoch.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.channels = channels
        self.num_samples = num_samples

        self.audio_files = self._discover_files(root, split)

        if len(self.audio_files) == 0:
            raise RuntimeError(
                f"No audio files found for split '{split}' in {root}. "
                "Expected GTZAN directory structure: "
                "root/genres/<genre>/<genre>.XXXXX.wav"
            )

    def _discover_files(self, root: str, split: str):
        """
        Discover audio files and split them deterministically.

        Files within each genre are sorted alphabetically and split as:
            - train: first 80%
            - valid: next 10%
            - test:  last 10%
        """
        root_path = Path(root)

        # Try common GTZAN directory structure variants
        genres_dir = None
        for candidate in ["genres", "genres_original", "."]:
            candidate_path = root_path / candidate
            if candidate_path.is_dir():
                subdirs = [d for d in candidate_path.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    genres_dir = candidate_path
                    break

        if genres_dir is None:
            raise RuntimeError(
                f"Could not find genre directories in {root}. "
                "Expected subdirectories like 'genres/' or 'genres_original/'."
            )

        all_files = []
        for genre_dir in sorted(genres_dir.iterdir()):
            if not genre_dir.is_dir():
                continue

            genre_files = sorted(
                str(f)
                for f in genre_dir.iterdir()
                if f.suffix.lower() in (".wav", ".au")
            )

            # Deterministic split: 80% train, 10% valid, 10% test
            n = len(genre_files)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)

            if split == "train":
                all_files.extend(genre_files[:train_end])
            elif split == "valid":
                all_files.extend(genre_files[train_end:val_end])
            elif split == "test":
                all_files.extend(genre_files[val_end:])
            else:
                raise ValueError(
                    f"Invalid split: {split}. Must be 'train', 'valid', or 'test'"
                )

        return all_files

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Pick a random track regardless of idx (virtual epoch pattern)
        max_retries = 10
        for attempt in range(max_retries):
            file_path = random.choice(self.audio_files)
            try:
                audio, orig_sr = torchaudio.load(file_path)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load audio after {max_retries} attempts. "
                        f"Last error: {e}"
                    )
                continue

        # audio shape: [channels, samples]

        # Resample if needed
        if orig_sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_sr, self.sample_rate
            )

        # Channel conversion
        if self.channels == 1 and audio.shape[0] > 1:
            # Convert to mono by averaging channels
            audio = audio.mean(dim=0, keepdim=True)
        elif self.channels == 2 and audio.shape[0] == 1:
            # Duplicate mono to stereo
            audio = audio.repeat(2, 1)

        # Random cropping
        num_frames = audio.shape[1]
        max_start = max(0, num_frames - self.chunk_samples)
        start = random.randint(0, max_start) if max_start > 0 else 0
        chunk = audio[:, start : start + self.chunk_samples]

        # Pad if the track is shorter than chunk_samples
        if chunk.shape[1] < self.chunk_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, self.chunk_samples - chunk.shape[1])
            )

        # Peak normalization
        peak = chunk.abs().max()
        if peak > 0:
            chunk = chunk / peak

        return chunk
