###
# H5DataModule - Drop-in replacement for ThreeSpeakerDataModule
# Reads from a single HDF5 file instead of 108,000 WAV files.
# 10-50x faster I/O, single file deployment.
#
# HDF5 structure expected:
#   /mixtures/{mixture_id}/
#       ├── mix   (float32, shape=[64000])
#       ├── s1    (float32, shape=[64000])
#       ├── s2    (float32, shape=[64000])
#       └── s3    (float32, shape=[64000])
#   /mixture_ids  (string array for fast indexing)
#   Root attrs: sample_rate, n_src, n_samples, target_length
###
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class H5Dataset(Dataset):
    """
    PyTorch Dataset reading directly from HDF5.

    Features:
    - Lazy loading: only reads data when __getitem__ is called
    - Per-worker file handle (thread-safe for DataLoader)
    - Random crop segment for training
    - Compatible with SPMamba's (mixture, sources, name) tuple format

    Parameters:
        h5_path (str): Path to .h5 file
        segment (float): Segment length in seconds. None = full length (test)
        sample_rate (int): Sample rate (read from file attrs if not set)
        n_src (int): Number of speakers (read from file attrs if not set)
        normalize_audio (bool): Zero-mean unit-variance normalization
        subset_indices (list): Only use these indices (for train/val/test split)
    """

    def __init__(
        self,
        h5_path: str,
        segment: float = 4.0,
        sample_rate: int = None,
        n_src: int = None,
        normalize_audio: bool = False,
        subset_indices: list = None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.normalize_audio = normalize_audio
        self.EPS = 1e-8

        # Open once to read metadata, then close
        with h5py.File(h5_path, 'r') as f:
            self.sample_rate = sample_rate or int(f.attrs['sample_rate'])
            self.n_src = n_src or int(f.attrs['n_src'])
            self.target_length = int(f.attrs.get('target_length', 64000))

            # Read mixture_ids for fast indexing
            if 'mixture_ids' in f:
                self.mixture_ids = list(f['mixture_ids'][:])
                if isinstance(self.mixture_ids[0], bytes):
                    self.mixture_ids = [mid.decode('utf-8') for mid in self.mixture_ids]
            else:
                self.mixture_ids = list(f['mixtures'].keys())

        # Segment length
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * self.sample_rate)

        self.test = self.seg_len is None

        # Subset filtering
        if subset_indices is not None:
            self.mixture_ids = [self.mixture_ids[i] for i in subset_indices]

        self.length = len(self.mixture_ids)

        # File handle (lazy-open per worker)
        self._h5file = None

        print(f"[H5Dataset] Loaded {self.length} samples from {h5_path}")
        print(f"  Sample rate: {self.sample_rate} Hz, Speakers: {self.n_src}")
        if segment:
            print(f"  Segment: {segment}s ({self.seg_len} samples)")
        else:
            print(f"  Segment: full length (test mode)")

    def _get_h5file(self):
        """Lazy-open. Each DataLoader worker needs its own handle."""
        if self._h5file is None:
            self._h5file = h5py.File(self.h5_path, 'r')
        return self._h5file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_h5file()
        mid = self.mixture_ids[idx]
        grp = f['mixtures'][mid]

        # Read full-length arrays
        mix_data = grp['mix']
        total_len = len(mix_data)

        # Random crop for training, full length for testing
        if self.test or self.seg_len is None or total_len <= self.seg_len:
            rand_start = 0
            stop = total_len
        else:
            rand_start = np.random.randint(0, total_len - self.seg_len)
            stop = rand_start + self.seg_len

        # Read data slice (HDF5 supports efficient slicing)
        mixture = torch.from_numpy(mix_data[rand_start:stop].astype(np.float32))

        source_arrays = []
        for i in range(1, self.n_src + 1):
            s = grp[f's{i}'][rand_start:stop].astype(np.float32)
            source_arrays.append(s)
        sources = torch.from_numpy(np.stack(source_arrays))

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

        return mixture, sources, mid

    def __del__(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass


class H5DataModule:
    """
    DataModule for SPMamba, reading from a single HDF5 file.
    Drop-in replacement for ThreeSpeakerDataModule.

    Auto-splits train/val/test from 1 file using deterministic seed.

    Config example (spmamba-h5-5080.yml):
        datamodule:
          data_name: H5DataModule
          data_config:
            h5_path: mixdata/storage/data_30h.h5
            n_src: 3
            sample_rate: 16000
            segment: 3.0
            batch_size: 6
            num_workers: 4
            pin_memory: true
    """

    def __init__(
        self,
        h5_path: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.05,
        segment: float = 3.0,
        normalize_audio: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        seed: int = 42,
        sample_rate: int = 16000,
        n_src: int = 3,
        # Unused params for config compat
        **kwargs,
    ):
        self.h5_path = h5_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and (num_workers > 0)
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.seed = seed
        self.sample_rate = sample_rate
        self.n_src = n_src

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self):
        """Split dataset into train/val/test with deterministic seed."""
        with h5py.File(self.h5_path, 'r') as f:
            total = int(f.attrs['n_samples'])

        # Deterministic split
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(total)

        n_test = int(total * self.test_ratio)
        n_val = int(total * self.val_ratio)
        n_train = total - n_val - n_test

        train_idx = sorted(indices[:n_train].tolist())
        val_idx = sorted(indices[n_train:n_train + n_val].tolist())
        test_idx = sorted(indices[n_train + n_val:].tolist())

        print(f"[H5DataModule] Split: train={n_train}, val={n_val}, test={n_test}")

        self.data_train = H5Dataset(
            h5_path=self.h5_path,
            segment=self.segment,
            sample_rate=self.sample_rate,
            n_src=self.n_src,
            normalize_audio=self.normalize_audio,
            subset_indices=train_idx,
        )
        self.data_val = H5Dataset(
            h5_path=self.h5_path,
            segment=self.segment,
            sample_rate=self.sample_rate,
            n_src=self.n_src,
            normalize_audio=self.normalize_audio,
            subset_indices=val_idx,
        )
        self.data_test = H5Dataset(
            h5_path=self.h5_path,
            segment=None,  # Full-length for testing
            sample_rate=self.sample_rate,
            n_src=self.n_src,
            normalize_audio=self.normalize_audio,
            subset_indices=test_idx,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
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
