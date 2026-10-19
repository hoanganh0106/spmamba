"""Build an HDF5 training file for the HLS-CMDS heart/lung dataset.

The output schema matches ``look2hear.datas.h5_datamodule.H5Dataset``:

    /mixtures/{id}/mix
    /mixtures/{id}/s1
    /mixtures/{id}/s2
    /mixture_ids
    attrs: sample_rate, n_src, n_samples, target_length

Two build modes are supported:

``dynamic`` (default, recommended)
    Synthesise mixtures from the *pure* single-source recordings in
    ``HS/HS`` (50 heart) and ``LS/LS`` (50 lung). Each mixture is built as
    ``mix = s1 + s2`` with a random heart-to-lung level ratio, so the
    sources are, by construction, the exact additive components of the
    mixture. This is the only setup for which supervised waveform
    separation (SI-SDR / SNR + PIT) is well-posed. 50x50 = 2500 possible
    pairs give far more data than the 145 recorded mixtures and reduce
    overfit.

    By default the sources are partitioned into disjoint train/val/test
    pools so the heart/lung recordings behind any eval mixture never appear
    in a train mixture (no source leakage). The split is written to the H5
    as a ``/split`` dataset and ``H5DataModule`` honours it instead of its
    random index split. Pass ``--no-source-split`` to fall back to a flat
    file (random split by the datamodule).

``triplet`` (legacy)
    Copy the recorded ``M####``/``H####``/``L####`` triplets from
    ``Mix/Mix`` verbatim. NOTE: in the recorded dataset the heart, lung and
    mixed signals are *independent takes* (``mix != s1 + s2``, correlation
    ~0), so this mode is NOT valid for supervised separation training. It is
    kept only for inspection / comparison.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np


DATASET_ROOT = (
    "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+"
    "using+digital+stethoscope/HLS-CMDS"
)
DEFAULT_HEART_SRC = f"{DATASET_ROOT}/HS/HS"
DEFAULT_LUNG_SRC = f"{DATASET_ROOT}/LS/LS"
DEFAULT_TRIPLET_SRC = f"{DATASET_ROOT}/Mix/Mix"


# --------------------------------------------------------------------------- #
# WAV I/O helpers
# --------------------------------------------------------------------------- #
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


def _list_wavs(src: Path) -> List[Path]:
    """Return sorted WAV files in a directory."""
    files = sorted(p for p in src.glob("*.wav") if p.is_file())
    return files


# --------------------------------------------------------------------------- #
# Dynamic mixing
# --------------------------------------------------------------------------- #
def _rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + eps)


def _partition_sources(
    n: int, val_frac: float, test_frac: float, rng: np.random.Generator
) -> Tuple[List[int], List[int], List[int]]:
    """Split ``n`` source indices into disjoint train/val/test pools."""
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n_test + n_val >= n:
        raise ValueError(
            f"val_frac+test_frac too large for {n} sources "
            f"(need >=1 left for train; got n_val={n_val}, n_test={n_test})"
        )
    test = sorted(idx[:n_test])
    val = sorted(idx[n_test : n_test + n_val])
    train = sorted(idx[n_test + n_val :])
    return train, val, test


def _sample_pairs(
    h_idx: List[int],
    l_idx: List[int],
    count: int,
    rng: np.random.Generator,
    allow_repeat: bool,
) -> List[Tuple[int, int]]:
    """Draw ``count`` (heart, lung) index pairs from the given pools.

    Without replacement while unique pairs remain; caps at the number of
    unique pairs when ``allow_repeat`` is False, otherwise refills with
    random repeats (each still gets a fresh SNR at mix time).
    """
    combos = [(h, l) for h in h_idx for l in l_idx]
    rng.shuffle(combos)
    if count <= len(combos):
        return combos[:count]
    if not allow_repeat:
        return combos
    extra = [combos[i] for i in rng.integers(0, len(combos), size=count - len(combos))]
    return combos + extra


def _make_mixture(
    heart: np.ndarray,
    lung: np.ndarray,
    snr_db: float,
    peak: float,
    rms_norm: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine one heart and one lung waveform into an additive mixture.

    Steps:
      1. (optional) normalise each source to unit RMS,
      2. boost the heart by ``snr_db`` dB so the heart-to-lung RMS ratio
         equals ``snr_db``,
      3. sum to form the mixture,
      4. rescale mixture *and* both sources by a single factor so the
         mixture peak equals ``peak`` (avoids clipping, keeps
         ``mix == s1 + s2`` exact and preserves the source ratio).

    Returns ``(mix, s1_heart, s2_lung)`` as float32.
    """
    heart = heart.astype(np.float32, copy=True)
    lung = lung.astype(np.float32, copy=True)

    if rms_norm:
        heart /= _rms(heart)
        lung /= _rms(lung)

    heart *= 10.0 ** (snr_db / 20.0)

    mix = heart + lung
    peak_val = float(np.max(np.abs(mix)))
    scale = (peak / peak_val) if peak_val > 1e-8 else 1.0

    heart *= scale
    lung *= scale
    mix = heart + lung

    return (
        mix.astype(np.float32, copy=False),
        heart.astype(np.float32, copy=False),
        lung.astype(np.float32, copy=False),
    )


def build_dynamic_h5(
    heart_src: Path,
    lung_src: Path,
    out: Path,
    sample_rate: int,
    target_length: int,
    n_mix: int,
    snr_min: float,
    snr_max: float,
    peak: float,
    rms_norm: bool,
    seed: int,
    compress: bool,
    source_split: bool,
    val_frac: float,
    test_frac: float,
) -> None:
    heart_src = heart_src.expanduser().resolve()
    lung_src = lung_src.expanduser().resolve()
    out = out.expanduser().resolve()

    if not heart_src.is_dir():
        raise FileNotFoundError(f"Heart source directory not found: {heart_src}")
    if not lung_src.is_dir():
        raise FileNotFoundError(f"Lung source directory not found: {lung_src}")

    heart_files = _list_wavs(heart_src)
    lung_files = _list_wavs(lung_src)
    if not heart_files:
        raise RuntimeError(f"No WAV files found under {heart_src}")
    if not lung_files:
        raise RuntimeError(f"No WAV files found under {lung_src}")

    # Preload every source waveform once (100 x 15s x float32 is tiny).
    def _load_all(files: List[Path], label: str) -> List[Tuple[str, np.ndarray]]:
        loaded = []
        for path in files:
            wav, sr = _read_wav(path)
            if sr != sample_rate:
                raise ValueError(f"{path.name}: expected {sample_rate} Hz, got {sr}")
            loaded.append((path.stem, _fit_length(wav, target_length, f"{label}/{path.name}")))
        return loaded

    hearts = _load_all(heart_files, "heart")
    lungs = _load_all(lung_files, "lung")
    print(f"[info] loaded {len(hearts)} heart + {len(lungs)} lung sources")

    rng = np.random.default_rng(seed)
    h_all = list(range(len(hearts)))
    l_all = list(range(len(lungs)))

    # Build (heart_idx, lung_idx, split) pairs. In source-disjoint mode the
    # heart AND lung recordings behind any val/test mixture never appear in a
    # train mixture, so evaluation cannot leak source identity.
    plan: List[Tuple[int, int, str]] = []
    if source_split:
        h_train, h_val, h_test = _partition_sources(len(hearts), val_frac, test_frac, rng)
        l_train, l_val, l_test = _partition_sources(len(lungs), val_frac, test_frac, rng)
        print(
            f"[info] source pools -> heart(train/val/test)="
            f"{len(h_train)}/{len(h_val)}/{len(h_test)}, "
            f"lung={len(l_train)}/{len(l_val)}/{len(l_test)}"
        )
        # val/test capped at their unique pairs (no duplicate pairs); train fills the rest.
        test_pairs = _sample_pairs(h_test, l_test, round(n_mix * test_frac), rng, allow_repeat=False)
        val_pairs = _sample_pairs(h_val, l_val, round(n_mix * val_frac), rng, allow_repeat=False)
        n_train = max(1, n_mix - len(val_pairs) - len(test_pairs))
        train_pairs = _sample_pairs(h_train, l_train, n_train, rng, allow_repeat=True)
        plan += [(h, l, "train") for h, l in train_pairs]
        plan += [(h, l, "val") for h, l in val_pairs]
        plan += [(h, l, "test") for h, l in test_pairs]
    else:
        pairs = _sample_pairs(h_all, l_all, n_mix, rng, allow_repeat=True)
        plan += [(h, l, "") for h, l in pairs]

    total = len(plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    ds_kwargs = {"compression": "gzip", "compression_opts": 4} if compress else {}
    string_dtype = h5py.string_dtype(encoding="utf-8")
    width = max(5, len(str(total)))
    report_interval = max(total // 10, 1)

    with h5py.File(out, "w") as h5:
        h5.attrs["sample_rate"] = sample_rate
        h5.attrs["n_src"] = 2
        h5.attrs["n_samples"] = total
        h5.attrs["target_length"] = target_length
        h5.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["mixing"] = "dynamic_hs_ls"
        h5.attrs["snr_min_db"] = snr_min
        h5.attrs["snr_max_db"] = snr_max
        h5.attrs["peak"] = peak
        h5.attrs["rms_norm"] = bool(rms_norm)
        h5.attrs["seed"] = seed
        h5.attrs["n_heart_sources"] = len(hearts)
        h5.attrs["n_lung_sources"] = len(lungs)
        h5.attrs["split_mode"] = "source_disjoint" if source_split else "none"

        mixture_ids = [f"M{idx:0{width}d}" for idx in range(1, total + 1)]
        h5.create_dataset(
            "mixture_ids",
            data=np.asarray(mixture_ids, dtype=object),
            dtype=string_dtype,
        )
        if source_split:
            h5.create_dataset(
                "split",
                data=np.asarray([s for *_, s in plan], dtype=object),
                dtype=string_dtype,
            )
        mixtures = h5.create_group("mixtures")

        for idx, (hi, li, split) in enumerate(plan):
            heart_id, heart_wav = hearts[hi]
            lung_id, lung_wav = lungs[li]
            snr_db = float(rng.uniform(snr_min, snr_max))

            mix, s1, s2 = _make_mixture(heart_wav, lung_wav, snr_db, peak, rms_norm)

            mid = mixture_ids[idx]
            group = mixtures.create_group(mid)
            group.attrs["heart"] = heart_id
            group.attrs["lung"] = lung_id
            group.attrs["snr_db"] = snr_db
            if split:
                group.attrs["split"] = split
            group.create_dataset("mix", data=mix, **ds_kwargs)
            group.create_dataset("s1", data=s1, **ds_kwargs)
            group.create_dataset("s2", data=s2, **ds_kwargs)

            if (idx + 1) % report_interval == 0 or idx + 1 == total:
                print(f"  ... {idx + 1}/{total} mixtures written")

    mib = out.stat().st_size / (1024 * 1024)
    print(f"[done] wrote {out}")
    print(f"[done] mode=dynamic, n_samples={total}, sample_rate={sample_rate}, n_src=2")
    if source_split:
        n_tr = sum(1 for *_, s in plan if s == "train")
        n_va = sum(1 for *_, s in plan if s == "val")
        n_te = sum(1 for *_, s in plan if s == "test")
        print(f"[done] split_mode=source_disjoint (in-file): train={n_tr}, val={n_va}, test={n_te}")
    else:
        print(f"[done] split_mode=none (H5DataModule will random-split by index)")
    print(f"[done] snr=[{snr_min}, {snr_max}] dB, peak={peak}, rms_norm={rms_norm}")
    print(f"[done] target_length={target_length}, size={mib:.1f} MiB")


# --------------------------------------------------------------------------- #
# Legacy triplet builder (mix != s1 + s2 -- inspection only)
# --------------------------------------------------------------------------- #
def _mixture_ids(src: Path) -> Iterable[str]:
    pattern = re.compile(r"^M(\d+)\.wav$", re.IGNORECASE)
    ids = []
    for path in src.glob("M*.wav"):
        match = pattern.match(path.name)
        if match:
            ids.append(match.group(1))
    return sorted(ids)


def build_triplet_h5(
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

    print(
        "[warn] triplet mode: recorded mix/heart/lung are independent takes "
        "(mix != s1 + s2). Not valid for supervised separation training."
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as h5:
        h5.attrs["sample_rate"] = sample_rate
        h5.attrs["n_src"] = 2
        h5.attrs["n_samples"] = len(triplets)
        h5.attrs["target_length"] = target_length
        h5.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["mixing"] = "recorded_triplet"

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
    print(f"[done] mode=triplet, n_samples={len(triplets)}, sample_rate={sample_rate}, n_src=2")
    print(f"[done] target_length={target_length}, size={mib:.1f} MiB")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an H5DataModule-compatible file from the HLS-CMDS dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=("dynamic", "triplet"),
        default="dynamic",
        help="dynamic: synthesise mix=s1+s2 from HS/LS (default). "
        "triplet: copy recorded M/H/L (mix != s1+s2, inspection only).",
    )
    parser.add_argument("--out", default="data_hls.h5", help="Output HDF5 file")
    parser.add_argument("--sample-rate", type=int, default=4000, help="Expected sample rate")
    parser.add_argument("--target-length", type=int, default=60000, help="Samples per waveform")

    # dynamic mode
    parser.add_argument("--heart-src", default=DEFAULT_HEART_SRC, help="Dir of pure heart WAVs (HS/HS)")
    parser.add_argument("--lung-src", default=DEFAULT_LUNG_SRC, help="Dir of pure lung WAVs (LS/LS)")
    parser.add_argument("--n-mix", type=int, default=1000, help="Number of mixtures to synthesise")
    parser.add_argument("--snr-min", type=float, default=-5.0, help="Min heart-to-lung ratio (dB)")
    parser.add_argument("--snr-max", type=float, default=5.0, help="Max heart-to-lung ratio (dB)")
    parser.add_argument("--peak", type=float, default=0.9, help="Target mixture peak amplitude")
    parser.add_argument("--no-rms-norm", action="store_true", help="Skip per-source RMS normalisation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for pairing/SNR")
    parser.add_argument("--compress", action="store_true", help="gzip datasets (smaller file, slower)")
    parser.add_argument(
        "--no-source-split",
        action="store_true",
        help="Disable in-file source-disjoint split; let H5DataModule random-split by index.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="Val fraction of sources & mixtures")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Test fraction of sources & mixtures")

    # triplet mode
    parser.add_argument("--src", default=DEFAULT_TRIPLET_SRC, help="[triplet] dir of M/H/L WAVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "dynamic":
        build_dynamic_h5(
            heart_src=Path(args.heart_src),
            lung_src=Path(args.lung_src),
            out=Path(args.out),
            sample_rate=args.sample_rate,
            target_length=args.target_length,
            n_mix=args.n_mix,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            peak=args.peak,
            rms_norm=not args.no_rms_norm,
            seed=args.seed,
            compress=args.compress,
            source_split=not args.no_source_split,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    else:
        build_triplet_h5(
            src=Path(args.src),
            out=Path(args.out),
            sample_rate=args.sample_rate,
            target_length=args.target_length,
        )


if __name__ == "__main__":
    main()
