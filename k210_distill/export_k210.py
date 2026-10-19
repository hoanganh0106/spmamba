#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xuất mô hình Student K210SepUNet sang định dạng ONNX tĩnh + dữ liệu hiệu chuẩn cho K210.

Sau khi huấn luyện xong trên cụm DGX A100, chạy lệnh:
    python export_k210.py \\
        --ckpt Experiments/checkpoint/K210-SepUNet-Distill-A100/best_model.pth \\
        --h5_path data/data.h5 \\
        --out_onnx k210_sep3.onnx \\
        --calib_dir k210_calib

Hoặc kiểm tra xuất đồ thị ONNX trước khi train (dùng trọng số ngẫu nhiên):
    python export_k210.py --dummy --out_onnx k210_sep3.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from look2hear.models.k210_unet import K210SepUNet, Resampler16kTo8k


ALLOWED_KPU_LEAF_TYPES = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.Sigmoid,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.Upsample,
)


def audit_k210_kpu_compatibility(model: K210SepUNet) -> None:
    """Kiểm tra nghiêm ngặt mọi lớp trong lõi mạng trước khi xuất sang nncase."""
    core_modules = [
        ("enc1", model.enc1),
        ("pool1", model.pool1),
        ("enc2", model.enc2),
        ("pool2", model.pool2),
        ("enc3", model.enc3),
        ("pool3", model.pool3),
        ("bottleneck", model.bottleneck),
        ("up3", model.up3),
        ("dec3", model.dec3),
        ("up2", model.up2),
        ("dec2", model.dec2),
        ("up1", model.up1),
        ("dec1", model.dec1),
        ("head", model.head),
    ]
    for prefix, root_mod in core_modules:
        for sub_name, mod in root_mod.named_modules():
            if len(list(mod.children())) > 0:
                continue
            full_name = f"{prefix}.{sub_name}" if sub_name else prefix
            if not isinstance(mod, ALLOWED_KPU_LEAF_TYPES):
                raise TypeError(
                    f"[K210 Audit FAIL] Lớp '{full_name}' ({type(mod).__name__}) "
                    "không nằm trong danh sách hỗ trợ phần cứng của KPU K210!"
                )
            if isinstance(mod, nn.Conv2d):
                kh, kw = mod.kernel_size
                if (kh, kw) not in ((1, 1), (3, 3)):
                    raise ValueError(
                        f"[K210 Audit FAIL] Conv2d '{full_name}' có kernel_size={(kh, kw)}. "
                        "KPU K210 chỉ hỗ trợ phần cứng cho kernel 1x1 và 3x3!"
                    )


def load_student_for_export(ckpt_path: str | None, dummy: bool = False) -> K210SepUNet:
    """Nạp K210SepUNet từ best_model.pth, Lightning .ckpt, hoặc khởi tạo mới nếu dummy=True."""
    if dummy or ckpt_path is None:
        model = K210SepUNet(sample_rate=8000)
        model.eval()
        return model

    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")

    # Trường hợp 1: file best_model.pth do BaseModel.serialize() lưu
    if "model_args" in raw and "state_dict" in raw:
        args = dict(raw["model_args"])
        sample_rate = args.pop("sample_rate", 8000)
        model = K210SepUNet(sample_rate=sample_rate, **args)
        model.load_state_dict(raw["state_dict"], strict=True)
        model.eval()
        return model

    # Trường hợp 2: file .ckpt của PyTorch Lightning
    if "state_dict" in raw:
        t_conf = raw.get("training_config", {})
        net_cfg = dict(t_conf.get("audionet", {}).get("audionet_config", {}))
        sample_rate = net_cfg.pop("sample_rate", 8000)
        model = K210SepUNet(sample_rate=sample_rate, **net_cfg)
        clean_sd = {}
        for k, v in raw["state_dict"].items():
            if k.startswith("audio_model."):
                clean_sd[k[len("audio_model.") :]] = v
            elif not k.startswith("resampler.") and not k.startswith("pit_aligner."):
                clean_sd[k] = v
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        return model

    raise ValueError(f"Không nhận diện được định dạng checkpoint tại: {ckpt_path}")


def load_student_checkpoint(ckpt_path: str | None, conf_dir: str | None = None) -> K210SepUNet:
    """Hàm tiện ích nạp checkpoint K210SepUNet cho eval_k210.py và realtime_stream_pc.py."""
    return load_student_for_export(ckpt_path=ckpt_path, dummy=(ckpt_path is None))


def export_calib_samples(
    model: K210SepUNet,
    h5_path: str | None,
    calib_dir: str,
    n_samples: int = 100,
    time_frames: int = 128,
) -> Tuple[int, str]:
    """Sinh tập dữ liệu hiệu chuẩn (Calibration dataset) cho nncase PTQ INT8."""
    out_dir = Path(calib_dir)
    npy_dir = out_dir / "npy"
    bin_dir = out_dir / "bin"
    npy_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    resampler = Resampler16kTo8k()
    saved = 0

    # Số mẫu waveform 8 kHz cần để tạo ra >= time_frames khung STFT (hop=128)
    wav_8k_len = time_frames * model.stride
    wav_16k_len = wav_8k_len * 2

    if h5_path and os.path.isfile(h5_path):
        import h5py

        print(f"[Calibration] Đang trích xuất {n_samples} mẫu phổ từ {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            if "mixture_ids" in f:
                mids = [
                    m.decode("utf-8") if isinstance(m, bytes) else str(m)
                    for m in f["mixture_ids"][:n_samples]
                ]
            else:
                mids = list(f["mixtures"].keys())[:n_samples]

            for idx, mid in enumerate(mids):
                mix_np = f["mixtures"][mid]["mix"][:wav_16k_len]
                wav_16k = torch.from_numpy(np.asarray(mix_np, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    wav_8k = resampler(wav_16k)
                    std = wav_8k.std(dim=-1, keepdim=True).clamp(min=model.eps)
                    _, _, feat = model.compute_stft_feature(wav_8k / std)
                    feat_crop = feat[:, :, : model.n_freqs_net, :time_frames]
                    if feat_crop.shape[-1] < time_frames:
                        pad = time_frames - feat_crop.shape[-1]
                        feat_crop = torch.nn.functional.pad(feat_crop, (0, pad))
                    arr = feat_crop.squeeze(0).cpu().numpy().astype(np.float32)  # [1, 128, 128]

                np.save(npy_dir / f"calib_{idx:04d}.npy", arr)
                arr.tofile(bin_dir / f"calib_{idx:04d}.bin")
                saved += 1
    else:
        print(
            f"[Calibration] Không có --h5_path hợp lệ; sinh {n_samples} mẫu phổ ngẫu nhiên để kiểm thử..."
        )
        rng = np.random.default_rng(42)
        for idx in range(n_samples):
            arr = rng.standard_normal((1, model.n_freqs_net, time_frames)).astype(np.float32)
            np.save(npy_dir / f"calib_{idx:04d}.npy", arr)
            arr.tofile(bin_dir / f"calib_{idx:04d}.bin")
            saved += 1

    return saved, str(bin_dir)


def main():
    ap = argparse.ArgumentParser(description="Xuất K210SepUNet sang ONNX + Calibration cho nncase")
    ap.add_argument("--ckpt", default=None, help="Đường dẫn tới best_model.pth hoặc .ckpt")
    ap.add_argument("--dummy", action="store_true", help="Xuất thử với trọng số ngẫu nhiên")
    ap.add_argument("--out_onnx", default="k210_sep3.onnx", help="Tên file ONNX đầu ra")
    ap.add_argument("--time_frames", type=int, default=128, help="Số khung thời gian tĩnh (mặc định 128 ~ 2.0s @ 8kHz)")
    ap.add_argument("--h5_path", default=None, help="Đường dẫn data.h5 để trích mẫu Calibration INT8")
    ap.add_argument("--calib_dir", default="k210_calib", help="Thư mục lưu mẫu Calibration")
    ap.add_argument("--calib_samples", type=int, default=100, help="Số mẫu Calibration")
    args = ap.parse_args()

    if args.time_frames % 8 != 0:
        raise ValueError(f"--time_frames phải chia hết cho 8 (nhận {args.time_frames})")

    model = load_student_for_export(args.ckpt, dummy=args.dummy)
    audit_k210_kpu_compatibility(model)

    total_params = sum(p.numel() for p in model.parameters())
    freq_bins = model.n_freqs_net
    t_frames = args.time_frames

    # Bật chế độ xuất phần lõi KPU (tách bỏ STFT/iSTFT)
    model.export_k210 = True
    model.eval()

    dummy_input = torch.randn(1, 1, freq_bins, t_frames, dtype=torch.float32)
    with torch.no_grad():
        dummy_out = model(dummy_input)

    print("=" * 68)
    print("KIỂM TRA ĐỒ THỊ K210 KPU TRƯỚC KHI XUẤT ONNX")
    print("=" * 68)
    print(f"  Kiến trúc             : K210SepUNet (export_k210=True)")
    print(f"  Tổng số tham số       : {total_params:,} params (~{total_params / 1024:.1f} KB INT8)")
    print(f"  Input Tensor (NCHW)   : {list(dummy_input.shape)} (float32)")
    print(f"  Output Tensor (Masks) : {list(dummy_out.shape)} (float32, 3 kênh)")
    print(f"  Kiểm toán toán tử KPU : PASS (100% Conv2d 1x1/3x3, BN, ReLU, MaxPool2d, Upsample, Sigmoid)")

    torch.onnx.export(
        model,
        dummy_input,
        args.out_onnx,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input_mag"],
        output_names=["output_masks"],
    )
    print(f"\n[OK] Đã xuất file ONNX tại: {args.out_onnx}")

    # Thử chạy onnx-simplifier nếu môi trường có cài
    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(args.out_onnx)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, args.out_onnx)
            print("[OK] Đã tối ưu hoá đồ thị bằng onnx-simplifier (BatchNorm đã gộp vào Conv2d).")
    except ImportError:
        print("[Gợi ý] Cài thêm `pip install onnx onnxsim` để gộp phẳng BatchNorm trước khi chạy nncase.")

    saved_cnt, bin_dir = export_calib_samples(
        model=model,
        h5_path=args.h5_path,
        calib_dir=args.calib_dir,
        n_samples=args.calib_samples,
        time_frames=t_frames,
    )
    print(f"[OK] Đã lưu {saved_cnt} mẫu hiệu chuẩn tại: {bin_dir}")

    print("\n" + "=" * 68)
    print("LỆNH BIÊN DỊCH SANG .KMODEL BẰNG NNCASE (v0.2.0-beta4)")
    print("=" * 68)
    print(
        f"  ncc compile {args.out_onnx} k210_sep3.kmodel \\\n"
        f"      -i onnx -t k210 \\\n"
        f"      --dataset {bin_dir} \\\n"
        f"      --dataset-format raw \\\n"
        f"      --input-type float32 \\\n"
        f"      --input-shape \"1,1,{freq_bins},{t_frames}\""
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
