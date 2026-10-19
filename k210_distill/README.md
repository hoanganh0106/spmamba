# Chắt Lọc Tri Thức & Triển Khai Thời Gian Thực Trên Kendryte K210 (8 kHz)

Thư mục này chứa pipeline nén mô hình tách nguồn âm thanh từ kiến trúc **SPMamba** (Teacher, 6.04M tham số, 16 kHz) sang kiến trúc lai nhẹ **`K210SepUNet`** (Student, ~508K tham số KPU, 8 kHz) để triển khai suy luận thời gian thực trên vi điều khiển biên **Kendryte K210** (KPU + Dual-core RISC-V 64-bit 400 MHz).

---

## Kiến trúc hệ thống

Do bộ tăng tốc phần cứng KPU trên K210 chỉ hỗ trợ các toán tử tích chập 2D cơ bản với bộ nhớ đệm 2 MB SRAM (không hỗ trợ STFT phức, phép nhân ma trận động hay SSM của Mamba), mô hình **`K210SepUNet`** phân chia tính toán đồng thiết kế Phần cứng/Phần mềm (*Hardware/Software Co-design*):

1. **Trên bộ tăng tốc KPU (INT8, ~1.48 MB / 2 MB SRAM):**
   - Mạng 2D Depthwise-Separable UNet (`channels = [32, 96, 192]`, `bottleneck_ch = 272`) trích xuất đặc trưng phổ biên độ Log-Magnitude STFT `[1, 128, 64]`.
   - 100% toán tử trong `forward_mask_core()` tương thích trực tiếp với trình biên dịch `nncase` của K210.
2. **Trên nhân CPU RISC-V 64-bit FPU:**
   - Thực hiện biến đổi STFT/iSTFT 256 điểm (cửa sổ 32 ms, bước nhảy 16 ms ở 8 kHz).
   - Khối **Dual-Path BiGRU + Multi-Head Self-Attention** kết hợp ước lượng hiệu chỉnh pha phức (*Complex Phase Residual*) tái tạo 3 kênh tiếng nói đầu ra.

---

## Cấu trúc thư mục

| File / Thư mục | Vai trò |
|---|---|
| `look2hear/models/k210_unet.py` | Kiến trúc `K210SepUNet` và bộ lọc chống chồng phổ `Resampler16kTo8k` |
| `look2hear/system/distill_litmodule.py` | Module huấn luyện PyTorch Lightning kết hợp PIT SI-SNR + Complex STFT Loss |
| `configs/distill-k210-a100.yml` | Cấu hình huấn luyện tối ưu trên GPU NVIDIA A100 80GB |
| `distill_train.py` | Script huấn luyện mô hình Student `K210SepUNet` |
| `smoke_k210_distill.py` | Kiểm tra tự động cú pháp AST, ràng buộc toán tử KPU và ngân sách bộ nhớ SRAM 2MB |
| `export_k210.py` | Xuất mô hình sang đồ thị tĩnh ONNX và trích xuất dữ liệu hiệu chuẩn lượng tử hóa INT8 |
| `eval_k210.py` | Đánh giá chỉ số SI-SNRi, SDRi và hệ số thời gian thực (RTF) của mô hình 8 kHz |
| `realtime_stream_pc.py` | Mô phỏng xử lý âm thanh thời gian thực theo từng khung trượt (Sliding Window) |
| `k210_firmware/main_k210_realtime.c` | Mã nguồn firmware C chạy suy luận thời gian thực trên vi điều khiển Kendryte K210 |
| `run_distill_a100.sh` | Script chạy job SLURM tự động trên cụm máy chủ DGX A100 |
| `setup_env_hust.sh` | Script khởi tạo môi trường Conda trên cụm máy chủ |
| `look2hear.yml` | File môi trường Conda |

---

## Hướng dẫn sử dụng

### 1. Kiểm tra tự động ràng buộc phần cứng K210 (Smoke Test)

```bash
cd k210_distill
python smoke_k210_distill.py
```

### 2. Huấn luyện mô hình `K210SepUNet`

```bash
# Chạy kiểm tra nhanh 1 batch (fast_dev_run)
python distill_train.py --conf_dir configs/distill-k210-a100.yml --fast_dev_run

# Huấn luyện đầy đủ với dữ liệu H5
python distill_train.py \
    --conf_dir configs/distill-k210-a100.yml \
    --h5_path /path/to/data_30h.h5 \
    --wandb
```

### 3. Đánh giá và Xuất mô hình sang K210 (ONNX + Calibration Data)

```bash
# Đánh giá SI-SNRi, SDRi và tốc độ suy luận (RTF)
python eval_k210.py \
    --ckpt Experiments/checkpoint/K210-SepUNet-Attention-A100/best_k210.ckpt \
    --conf_dir configs/distill-k210-a100.yml \
    --h5_path /path/to/data_30h.h5

# Xuất file ONNX tĩnh + tập mẫu hiệu chuẩn INT8 cho trình biên dịch nncase
python export_k210.py \
    --ckpt Experiments/checkpoint/K210-SepUNet-Attention-A100/best_k210.ckpt \
    --conf_dir configs/distill-k210-a100.yml \
    --out_onnx output_k210/k210_sep3.onnx \
    --calib_dir output_k210/k210_calib
```
