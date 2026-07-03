"""Build an HDF5 training file for the HLS-CMDS heart/lung dataset.

The output schema matches ``look2hear.datas.h5_datamodule.H5Dataset``:

    /mixtures/{M####}/mix
    /mixtures/{M####}/s1
    /mixtures/{M####}/s2
    /mixture_ids
    attrs: sample_rate, n_src, n_samples, target_length
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np


DEFAULT_SRC = (
    "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+"
    "using+digital+stethoscope/HLS-CMDS/Mix/Mix"
)


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in [-1, 1]."""
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except ImportError:
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(path)
        data = _wavfile_to_float32(data)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        if data.shape[1] == 1:
            data = data[:, 0]
        else:
            print(f"[warn] {path.name}: {data.shape[1]} channels, averaging to mono")
            data = data.mean(axis=1, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError(f"{path} has unsupported shape {data.shape}")
    return data.astype(np.float32, copy=False), int(sample_rate)


def _wavfile_to_float32(data: np.ndarray) -> np.ndarray:
    """Scale scipy.io.wavfile output to float32."""
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)

    if np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        centered = data.astype(np.float32) - (info.max + 1) / 2.0
        return centered / ((info.max + 1) / 2.0)

    if np.issubdtype(data.dtype, np.signedinteger):
        info = np.iinfo(data.dtype)
        return data.astype(np.float32) / float(info.max)

    raise TypeError(f"Unsupported WAV dtype: {data.dtype}")


def _fit_length(data: np.ndarray, target_length: int, label: str) -> np.ndarray:
    """Pad or crop a waveform to the expected fixed length."""
    if len(data) == target_length:
        return data.astype(np.float32, copy=False)

    if len(data) > target_length:
        print(f"[warn] {label}: cropped {len(data)} -> {target_length} samples")
        return data[:target_length].astype(np.float32, copy=False)

    print(f"[warn] {label}: padded {len(data)} -> {target_length} samples")
    return np.pad(data, (0, target_length - len(data))).astype(np.float32, copy=False)


def _mixture_ids(src: Path) -> Iterable[str]:
    pattern = re.compile(r"^M(\d+)\.wav$", re.IGNORECASE)
    ids = []
    for path in src.glob("M*.wav"):
        match = pattern.match(path.name)
        if match:
            ids.append(match.group(1))
    return sorted(ids)


def build_hls_h5(
    src: Path,
    out: Path,
    sample_rate: int,
    target_length: int,
) -> None:
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    ids = list(_mixture_ids(src))
    if not ids:
        raise RuntimeError(f"No M####.wav files found under {src}")

    missing = []
    triplets = []
    for item_id in ids:
        mix_path = src / f"M{item_id}.wav"
        heart_path = src / f"H{item_id}.wav"
        lung_path = src / f"L{item_id}.wav"
        absent = [p.name for p in (mix_path, heart_path, lung_path) if not p.is_file()]
        if absent:
            missing.append((item_id, absent))
            continue
        triplets.append((f"M{item_id}", mix_path, heart_path, lung_path))

    if missing:
        details = "; ".join(f"{item_id}: {', '.join(names)}" for item_id, names in missing[:10])
        if len(missing) > 10:
            details += f"; ... {len(missing) - 10} more"
        raise RuntimeError(f"Missing HLS-CMDS triplet files: {details}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as h5:
        h5.attrs["sample_rate"] = sample_rate
        h5.attrs["n_src"] = 2
        h5.attrs["n_samples"] = len(triplets)
        h5.attrs["target_length"] = target_length

        h5.create_dataset(
            "mixture_ids",
            data=np.asarray([mid for mid, *_ in triplets], dtype=object),
            dtype=string_dtype,
        )
        mixtures = h5.create_group("mixtures")

        for mid, mix_path, heart_path, lung_path in triplets:
            mix, mix_sr = _read_wav(mix_path)
            heart, heart_sr = _read_wav(heart_path)
            lung, lung_sr = _read_wav(lung_path)

            rates = {mix_sr, heart_sr, lung_sr}
            if rates != {sample_rate}:
                raise ValueError(
                    f"{mid}: expected {sample_rate} Hz, got "
                    f"mix={mix_sr}, heart={heart_sr}, lung={lung_sr}"
                )

            raw_lengths = (len(mix), len(heart), len(lung))
            if len(set(raw_lengths)) != 1:
                print(f"[warn] {mid}: raw lengths differ {raw_lengths}")

            group = mixtures.create_group(mid)
            group.create_dataset("mix", data=_fit_length(mix, target_length, f"{mid}/mix"))
            group.create_dataset("s1", data=_fit_length(heart, target_length, f"{mid}/s1"))
            group.create_dataset("s2", data=_fit_length(lung, target_length, f"{mid}/s2"))

    mib = out.stat().st_size / (1024 * 1024)
    print(f"[done] wrote {out}")
    print(f"[done] n_samples={len(triplets)}, sample_rate={sample_rate}, n_src=2")
    print(f"[done] target_length={target_length}, size={mib:.1f} MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HLS-CMDS Mix/Mix WAV triplets to an H5DataModule-compatible file.",
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="Directory containing M####/H####/L#### WAV files")
    parser.add_argument("--out", default="data_hls.h5", help="Output HDF5 file")
    parser.add_argument("--sample-rate", type=int, default=4000, help="Expected sample rate")
    parser.add_argument("--target-length", type=int, default=60000, help="Samples per waveform")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_hls_h5(
        src=Path(args.src),
        out=Path(args.out),
        sample_rate=args.sample_rate,
        target_length=args.target_length,
    )


if __name__ == "__main__":
    main()
