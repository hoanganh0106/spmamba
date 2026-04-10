###
# ThreeSpeakerDataModule
# Supports dataset structure:
#   data_dir/
#     mix/   - mixture audio (*.wav)
#     s1/    - speaker 1 audio (*.wav)
#     s2/    - speaker 2 audio (*.wav)
#     s3/    - speaker 3 audio (*.wav)
#
# If val_dir / test_dir not provided, automatically splits train by val_split ratio.
###
import os
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, Subset


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class ThreeSpeakerDataset(Dataset):
    """
    Dataset for 3-speaker separation.
    Reads WAV files directly from mix/, s1/, s2/, s3/ subdirectories.

    Args:
        data_dir:        Root folder containing mix/, s1/, s2/, s3/
        n_src:           Number of sources (3)
        sample_rate:     Target sample rate (will resample if needed via soundfile)
        segment:         Clip length in seconds. None = full audio (for test)
        normalize_audio: Apply per-utterance RMS normalization
        indices:         Optional list of indices to use (for subset splits)
    """

    def __init__(
        self,
        data_dir: str,
        n_src: int = 3,
        sample_rate: int = 16000,
        segment: float = 3.0,
        normalize_audio: bool = False,
        indices=None,
    ):
        super().__init__()
        self.EPS = 1e-8
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.seg_len = int(segment * sample_rate) if segment is not None else None
        self.is_test = self.seg_len is None

        mix_dir = os.path.join(data_dir, "mix")
        src_dirs = [os.path.join(data_dir, f"s{i+1}") for i in range(n_src)]

        # Gather filenames from mix/ folder
        all_files = sorted([
            f for f in os.listdir(mix_dir) if f.endswith(".wav")
        ])

        if len(all_files) == 0:
            raise ValueError(f"No .wav files found in {mix_dir}")

        # Verify source folders exist
        for src_dir in src_dirs:
            if not os.path.isdir(src_dir):
                raise ValueError(f"Source directory not found: {src_dir}")

        # Apply index subset if given
        if indices is not None:
            all_files = [all_files[i] for i in indices]

        # Filter out clips shorter than segment length
        valid_files = []
        dropped = 0
        for fname in all_files:
            mix_path = os.path.join(mix_dir, fname)
            info = sf.info(mix_path)
            n_samples = int(info.frames * sample_rate / info.samplerate)
            if self.seg_len is None or n_samples >= self.seg_len:
                valid_files.append(fname)
            else:
                dropped += 1

        if dropped > 0:
            print(f"[ThreeSpeakerDataset] Dropped {dropped} files shorter than {segment}s")

        self.mix_paths = [os.path.join(mix_dir, f) for f in valid_files]
        self.src_paths = [
            [os.path.join(src_dir, f) for f in valid_files]
            for src_dir in src_dirs
        ]
        self.filenames = valid_files
        print(f"[ThreeSpeakerDataset] Loaded {len(self.mix_paths)} files from {data_dir}")

    def __len__(self):
        return len(self.mix_paths)

    def __getitem__(self, idx):
        mix_path = self.mix_paths[idx]

        if self.is_test:
            # Full audio for test/val
            mixture, sr = sf.read(mix_path, dtype="float32")
            sources = []
            for src_list in self.src_paths:
                s, _ = sf.read(src_list[idx], dtype="float32")
                sources.append(s)
        else:
            # Read audio info for random chunk selection
            info = sf.info(mix_path)
            total_samples = info.frames
            # Adjust for sample rate difference
            total_samples_target = int(total_samples * self.sample_rate / info.samplerate)

            if total_samples_target <= self.seg_len:
                start = 0
            else:
                start = random.randint(0, total_samples_target - self.seg_len)

            # Convert start/stop to original file sample rate
            orig_start = int(start * info.samplerate / self.sample_rate)
            orig_stop = int((start + self.seg_len) * info.samplerate / self.sample_rate)

            mixture, sr = sf.read(mix_path, start=orig_start, stop=orig_stop, dtype="float32")
            sources = []
            for src_list in self.src_paths:
                s, _ = sf.read(src_list[idx], start=orig_start, stop=orig_stop, dtype="float32")
                sources.append(s)

        # Convert to tensors
        mixture = torch.from_numpy(mixture.copy())        # [T]
        sources = torch.from_numpy(np.stack(sources, axis=0).copy())  # [n_src, T]

        # Pad or trim to exact seg_len
        if not self.is_test and self.seg_len is not None:
            if mixture.shape[-1] < self.seg_len:
                pad_len = self.seg_len - mixture.shape[-1]
                mixture = torch.nn.functional.pad(mixture, (0, pad_len))
                sources = torch.nn.functional.pad(sources, (0, pad_len))
            else:
                mixture = mixture[:self.seg_len]
                sources = sources[:, :self.seg_len]

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

        return mixture, sources, self.filenames[idx]


class ThreeSpeakerDataModule:
    """
    DataModule for 3-speaker separation dataset.

    Dataset structure expected:
        train_dir/
            mix/  s1/  s2/  s3/

    If valid_dir or test_dir is None or does not exist,
    the training set is automatically split using val_split ratio (default 10%).

    Config example (spmamba-3speakers.yml):
        datamodule:
          data_name: ThreeSpeakerDataModule
          data_config:
            train_dir: /content/data/train
            valid_dir: null        # auto-split from train
            test_dir: null         # auto-split from train
            n_src: 3
            sample_rate: 16000
            segment: 3.0
            normalize_audio: false
            batch_size: 4
            num_workers: 4
            pin_memory: true
            persistent_workers: false
            val_split: 0.1
    """

    def __init__(
        self,
        train_dir: str,
        valid_dir: str = None,
        test_dir: str = None,
        n_src: int = 3,
        sample_rate: int = 16000,
        segment: float = 3.0,
        normalize_audio: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        val_split: float = 0.1,
    ):
        self.train_dir = train_dir
        self.valid_dir = valid_dir if valid_dir and os.path.isdir(str(valid_dir)) else None
        self.test_dir = test_dir if test_dir and os.path.isdir(str(test_dir)) else None
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.val_split = val_split

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self):
        if self.valid_dir and self.test_dir:
            # Use separate directories
            self.data_train = ThreeSpeakerDataset(
                self.train_dir, self.n_src, self.sample_rate, self.segment, self.normalize_audio
            )
            self.data_val = ThreeSpeakerDataset(
                self.valid_dir, self.n_src, self.sample_rate, None, self.normalize_audio
            )
            self.data_test = ThreeSpeakerDataset(
                self.test_dir, self.n_src, self.sample_rate, None, self.normalize_audio
            )
        else:
            # Auto-split train → train / val / test
            full_dataset = ThreeSpeakerDataset(
                self.train_dir, self.n_src, self.sample_rate, self.segment, self.normalize_audio
            )
            n_total = len(full_dataset)
            n_val = max(1, int(n_total * self.val_split))
            n_test = max(1, int(n_total * self.val_split))
            n_train = n_total - n_val - n_test

            indices = list(range(n_total))
            random.seed(42)
            random.shuffle(indices)

            train_idx = indices[:n_train]
            val_idx   = indices[n_train:n_train + n_val]
            test_idx  = indices[n_train + n_val:]

            print(f"[ThreeSpeakerDataModule] Auto-split: train={n_train}, val={n_val}, test={n_test}")

            self.data_train = Subset(full_dataset, train_idx)
            # For val/test, reload without segment limit (full audio)
            full_test_ds = ThreeSpeakerDataset(
                self.train_dir, self.n_src, self.sample_rate, None, self.normalize_audio
            )
            self.data_val  = Subset(full_test_ds, val_idx)
            self.data_test = Subset(full_test_ds, test_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,           # full-length clips, batch=1
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
