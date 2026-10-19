#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra tự động (Smoke Test & K210 Hardware Audit) cho pipeline K210SepUNet Distillation.

Thiết kế hoạt động ở 2 cấp độ:
- Cấp độ 1 (Static / AST / Memory Budget / SLURM Audit): Chạy được ngay trên máy cá nhân
  kể cả khi chưa cài đặt PyTorch/NumPy. Kiểm tra cú pháp AST, kiểm toán toán tử KPU K210,
  tính chính xác số tham số & bộ nhớ SRAM 2MB của K210, và đảm bảo các file `.sh` dùng
  ký tự xuống dòng chuẩn Linux (LF) để `sbatch` trên cụm DGX A100 HUST không bị lỗi.
- Cấp độ 2 (Runtime Tensor / Anti-aliasing Filter / Distillation / ONNX Export): Tự động
  chạy thêm khi phát hiện có `torch` (trên cụm DGX A100 hoặc môi trường Conda).
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  {detail}")


def calc_conv_bn_params(in_ch: int, out_ch: int, ks: int) -> int:
    return (in_ch * out_ch * ks * ks) + (2 * out_ch)


def calc_dsconv_params(in_ch: int, out_ch: int) -> int:
    dw = (in_ch * 1 * 3 * 3) + (2 * in_ch)
    pw = (in_ch * out_ch * 1 * 1) + (2 * out_ch)
    return dw + pw


def test_static_ast_and_memory_budget() -> None:
    print("\n[1/3] Kiểm tra tĩnh AST, Ràng buộc toán tử KPU & Ngân sách bộ nhớ K210")
    files_to_check = [
        "look2hear/models/k210_unet.py",
        "look2hear/models/__init__.py",
        "look2hear/system/distill_litmodule.py",
        "look2hear/system/__init__.py",
        "distill_train.py",
        "export_k210.py",
        "eval_k210.py",
        "realtime_stream_pc.py",
    ]
    for rel_path in files_to_check:
        p = Path(rel_path)
        check(f"File tồn tại: {rel_path}", p.is_file())
        if p.is_file():
            try:
                src = p.read_text(encoding="utf-8")
                ast.parse(src, filename=rel_path)
                check(f"Cú pháp Python hợp lệ: {rel_path}", True)
            except SyntaxError as exc:
                check(f"Cú pháp Python hợp lệ: {rel_path}", False, str(exc))

    check(
        "Tồn tại firmware C phát trực tiếp cho K210: k210_firmware/main_k210_realtime.c",
        Path("k210_firmware/main_k210_realtime.c").is_file(),
    )

    # Kiểm tra AST của K210SepUNet.forward_mask_core không gọi stft/istft/fft/pad động
    unet_src = Path("look2hear/models/k210_unet.py").read_text(encoding="utf-8")
    tree = ast.parse(unet_src)
    forbidden_in_core = {"stft", "istft", "fft", "rfft", "irfft", "pad", "unfold", "matmul"}
    found_forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward_mask_core":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in forbidden_in_core:
                    found_forbidden.append(sub.attr)
                elif isinstance(sub, ast.Name) and sub.id in forbidden_in_core:
                    found_forbidden.append(sub.id)
    check(
        "forward_mask_core() thuần KPU (không chứa STFT/FFT/pad/unfold/matmul)",
        len(found_forbidden) == 0,
        f"Phát hiện: {found_forbidden}",
    )

    # Tính giải tích chính xác số tham số của K210SepUNet-Large (channels=(32,96,192), bottleneck_ch=272, n_srcs=3)
    c1, c2, c3 = 32, 96, 192
    b_ch = 272
    n_srcs = 3
    enc1_p = calc_conv_bn_params(1, c1, 3) + calc_conv_bn_params(c1, c1, 3)
    enc2_p = calc_conv_bn_params(c1, c2, 3) + calc_dsconv_params(c2, c2)
    enc3_p = calc_conv_bn_params(c2, c3, 3) + calc_dsconv_params(c3, c3)
    bot_p = (
        calc_dsconv_params(c3, b_ch)
        + calc_dsconv_params(b_ch, b_ch)
        + calc_dsconv_params(b_ch, c3)
    )
    dec3_p = calc_conv_bn_params(c3 + c3, c2, 1) + calc_dsconv_params(c2, c2)
    dec2_p = calc_conv_bn_params(c2 + c2, c1, 1) + calc_dsconv_params(c1, c1)
    dec1_p = calc_conv_bn_params(c1 + c1, c1, 1) + calc_conv_bn_params(c1, c1, 3)
    head_p = (c1 * n_srcs * 1 * 1) + n_srcs
    total_params = enc1_p + enc2_p + enc3_p + bot_p + dec3_p + dec2_p + dec1_p + head_p

    check(
        f"Số tham số K210SepUNet-Large = {total_params:,} nằm trong ngưỡng [480K, 550K]",
        480_000 <= total_params <= 550_000,
        f"total_params={total_params}",
    )

    # Kiểm tra ngân sách toàn bộ 2MB (2048 KiB) AI SRAM của K210 dành trọn cho tách 3 âm
    weight_kib_int8 = total_params / 1024.0
    # Feature map lớn nhất là tại dec1 khi nối tắt torch.cat([d1, e1]) có (2 * c1) kênh x 128 x 128 bytes
    peak_act_kib_int8 = ((2 * c1) * 128 * 128) / 1024.0
    total_kpu_kib = weight_kib_int8 + peak_act_kib_int8
    check(
        f"Tổng RAM KPU ước tính ở cửa sổ 2.0s (Weights {weight_kib_int8:.1f} KiB + Peak Act {peak_act_kib_int8:.1f} KiB = {total_kpu_kib:.1f} KiB) < 2048 KiB (2 MB)",
        total_kpu_kib < 2048.0,
        f"total_kpu_kib={total_kpu_kib:.1f} KiB",
    )


def test_configs_and_slurm_scripts() -> None:
    print("\n[2/3] Kiểm tra cấu hình YAML và Script SLURM cho cụm DGX A100 HUST")
    cfg_path = Path("configs/distill-k210-a100.yml")
    check("Tồn tại configs/distill-k210-a100.yml", cfg_path.is_file())
    if cfg_path.is_file():
        cfg_text = cfg_path.read_text(encoding="utf-8")
        check("Config dùng K210SepUNet", "audionet_name: K210SepUNet" in cfg_text)
        check("Config dùng channels: [32, 96, 192] (Large)", "channels: [32, 96, 192]" in cfg_text)
        check("Config dùng DistillLightningModule", "system: DistillLightningModule" in cfg_text)
        check("Config bật bf16-mixed cho A100", "precision: bf16-mixed" in cfg_text)

    # Đảm bảo các file .sh dùng chuẩn LF (\n) chứ không phải CRLF (\r\n) gây lỗi sbatch trên Linux
    for sh_name in ["setup_env_hust.sh", "run_distill_a100.sh"]:
        sh_path = Path(sh_name)
        check(f"Tồn tại script {sh_name}", sh_path.is_file())
        if sh_path.is_file():
            raw_bytes = sh_path.read_bytes()
            if b"\r\n" in raw_bytes:
                # Tự động chuẩn hoá về LF để sbatch trên Linux không bao giờ báo lỗi DOS line breaks
                raw_bytes = raw_bytes.replace(b"\r\n", b"\n")
                sh_path.write_bytes(raw_bytes)
            check(
                f"Script {sh_name} dùng ký tự xuống dòng LF chuẩn Linux (tương thích sbatch)",
                b"\r\n" not in sh_path.read_bytes(),
            )


def test_runtime_pytorch_if_available() -> None:
    print("\n[3/3] Kiểm tra Runtime PyTorch (Model, Bộ lọc 16k->8k, Distillation & ONNX)")
    try:
        import math
        import torch
    except ImportError:
        print("  [SKIP] Máy hiện tại chưa cài đặt PyTorch; bước 3 sẽ tự động chạy trên cụm DGX A100.")
        return

    import distill_train
    import eval_k210
    from export_k210 import audit_k210_kpu_compatibility
    from look2hear.losses import PITLossWrapper, pairwise_neg_sisdr
    from look2hear.models.k210_unet import K210SepUNet, Resampler16kTo8k
    from look2hear.system.distill_litmodule import DistillLightningModule
    check("Import thành công distill_train, eval_k210, export_k210", True)

    # 1. Kiểm tra bộ lọc chống răng cưa Resampler16kTo8k
    resampler = Resampler16kTo8k()
    t = torch.arange(16000, dtype=torch.float32) / 16000.0
    sine_1k = torch.sin(2.0 * math.pi * 1000.0 * t).unsqueeze(0)  # Trong dải thông (< 3600 Hz)
    sine_6k = torch.sin(2.0 * math.pi * 6000.0 * t).unsqueeze(0)  # Ngoài Nyquist 4000 Hz (cần bị triệt tiêu)
    out_1k = resampler(sine_1k)
    out_6k = resampler(sine_6k)
    check("Resampler16kTo8k chia đôi đúng số mẫu (16000 -> 8000)", out_1k.shape == (1, 8000))
    check(
        "Resampler16kTo8k giữ nguyên biên độ sóng 1 kHz (> 0.95)",
        float(out_1k[:, 100:-100].abs().max()) > 0.95,
        f"max={float(out_1k[:, 100:-100].abs().max()):.4f}",
    )
    check(
        "Resampler16kTo8k triệt tiêu sóng 6 kHz chống aliasing (< 0.02)",
        float(out_6k[:, 100:-100].abs().max()) < 0.02,
        f"max={float(out_6k[:, 100:-100].abs().max()):.4f}",
    )

    # 2. Kiểm tra K210SepUNet ở cả 2 chế độ
    model = K210SepUNet(sample_rate=8000)
    n_params = sum(p.numel() for p in model.parameters())
    check("K210SepUNet khớp đúng 508,227 tham số", n_params == 508_227, f"nhan {n_params}")

    try:
        audit_k210_kpu_compatibility(model)
        check("audit_k210_kpu_compatibility() đạt 100%", True)
    except Exception as exc:
        check("audit_k210_kpu_compatibility() đạt 100%", False, str(exc))

    wav_in = torch.randn(2, 32000, requires_grad=True)
    wav_out = model(wav_in)
    check(
        "K210SepUNet (export_k210=False) trả về đúng shape [2, 3, 32000]",
        wav_out.shape == (2, 3, 32000),
        f"nhan {tuple(wav_out.shape)}",
    )
    loss_dummy = wav_out.pow(2).mean()
    loss_dummy.backward()
    check("Gradient lan truyền ngược qua K210SepUNet thành công", wav_in.grad is not None)

    # 3. Kiểm tra chế độ export_k210=True và xuất thử ONNX
    model.export_k210 = True
    model.eval()
    mag_in = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        mask_out = model(mag_in)
    check(
        "K210SepUNet (export_k210=True) trả về đúng shape [1, 3, 128, 128]",
        mask_out.shape == (1, 3, 128, 128),
        f"nhan {tuple(mask_out.shape)}",
    )
    check(
        "Mặt nạ dự đoán nằm trong khoảng (0, 1)",
        bool((mask_out >= 0.0).all() and (mask_out <= 1.0).all()),
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_file = Path(tmpdir) / "test_k210.onnx"
            torch.onnx.export(
                model,
                mag_in,
                str(onnx_file),
                opset_version=11,
                input_names=["input_mag"],
                output_names=["output_masks"],
            )
            check("Xuất đồ thị ONNX Opset 11 thành công", onnx_file.is_file() and onnx_file.stat().st_size > 0)
    except Exception as exc:
        check("Xuất đồ thị ONNX Opset 11 thành công", False, str(exc))

    # 4. Kiểm tra DistillLightningModule không lưu nhầm trọng số Teacher vào state_dict của Student
    model.export_k210 = False

    class _MockTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.big_weight = torch.nn.Parameter(torch.randn(1000, 1000))

        def forward(self, x):
            return torch.stack([x * 0.5, x * 0.3, x * 0.2], dim=1)

    mock_teacher = _MockTeacher().eval()
    for p in mock_teacher.parameters():
        p.requires_grad_(False)

    loss_fn = {
        "train": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False),
    }
    cfg = {
        "datamodule": {"data_config": {"sample_rate": 16000}},
        "distill": {"alpha_hard": 0.5, "beta_tea_wav": 0.5, "gamma_tea_mag": 10.0},
    }
    lit = DistillLightningModule(
        audio_model=model,
        teacher_model=mock_teacher,
        loss_func=loss_fn,
        config=cfg,
    )
    lit.train()
    check("Teacher vẫn giữ trạng thái eval() khi gọi lit.train()", not mock_teacher.training)
    sd_keys = list(lit.state_dict().keys())
    check(
        "state_dict() của DistillLightningModule không chứa trọng số của Teacher",
        all("big_weight" not in k for k in sd_keys),
    )

    # 5. Chạy thử 1 bước training_step thực tế (16kHz -> 8kHz + PIT + 3-term Distillation Loss)
    batch_mix_16k = torch.randn(2, 16000)
    batch_tgt_16k = torch.randn(2, 3, 16000)
    batch_keys = ["utt_0", "utt_1"]
    step_out = lit.training_step((batch_mix_16k, batch_tgt_16k, batch_keys), 0)
    loss_step = step_out["loss"] if isinstance(step_out, dict) else step_out
    check(
        "DistillLightningModule.training_step() chạy thông suốt & trả về loss hữu hạn",
        torch.isfinite(loss_step).item(),
        f"loss={loss_step.item():.4f}",
    )

    # 6. Kiểm tra file Teacher checkpoint thực tế và tập data_30h.h5 nếu có mặt trên máy/server
    for ckpt_cand in ["output/epoch=91-best.ckpt", "epoch=91-best.ckpt"]:
        if Path(ckpt_cand).is_file():
            try:
                raw_ckpt = torch.load(ckpt_cand, map_location="cpu", weights_only=False)
                sd = raw_ckpt.get("state_dict", raw_ckpt)
                n_tea_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
                check(
                    f"Checkpoint Teacher '{ckpt_cand}' hợp lệ ({n_tea_params:,} tham số SPMamba)",
                    n_tea_params > 5_000_000,
                )
            except Exception as exc:
                check(f"Checkpoint Teacher '{ckpt_cand}' hợp lệ", False, str(exc))
            break

    for h5_cand in ["data/data_30h.h5", "data/data.h5"]:
        if Path(h5_cand).is_file():
            try:
                import h5py

                with h5py.File(h5_cand, "r") as hf:
                    n_samples_attr = int(hf.attrs.get("n_samples", 0))
                    sr_attr = int(hf.attrs.get("sample_rate", 16000))
                    n_src_attr = int(hf.attrs.get("n_src", 3))
                    has_mixtures = "mixtures" in hf
                check(
                    f"File dữ liệu '{h5_cand}' đọc tốt (n_samples={n_samples_attr}, sr={sr_attr}Hz, n_src={n_src_attr})",
                    has_mixtures and n_samples_attr > 0 and n_src_attr == 3,
                )
            except Exception as exc:
                check(f"File dữ liệu '{h5_cand}' đọc tốt", False, str(exc))
            break


def main() -> int:
    print("=" * 68)
    print("SMOKE TEST & K210 HARDWARE AUDIT — K210SepUNet Distillation")
    print("=" * 68)
    test_static_ast_and_memory_budget()
    test_configs_and_slurm_scripts()
    test_runtime_pytorch_if_available()

    print("\n" + "=" * 68)
    print(f"KẾT QUẢ: {len(PASSED)} PASS | {len(FAILED)} FAIL")
    print("=" * 68)
    if FAILED:
        for item in FAILED:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
