# Hệ Thống Tách Nguồn Âm Thanh & Phân Tích Âm Tim/Phổi (SPMamba)

Kho lưu trữ này tổng hợp mã nguồn nghiên cứu và thực nghiệm dựa trên kiến trúc **SPMamba** (*State-Space Model for Audio Source Separation*), được tổ chức thành **3 thư mục độc lập** tương ứng với 3 hạng mục của đề tài:

```text
.
├── spmamba/        # 1. Mô hình SPMamba tách nguồn tiếng nói đa người nói (3 speakers, 16 kHz)
├── hls/            # 2. Tách âm Tim/Phổi & Nhận dạng âm bệnh lý bất thường (HLS-CMDS, 4 kHz)
├── k210_distill/   # 3. Chắt lọc tri thức & Triển khai thời gian thực trên vi điều khiển K210 (8 kHz)
└── LICENSE
```

Mỗi thư mục con đã được đóng gói **đầy đủ và tự chứa** (`look2hear/`, `configs/`, `look2hear.yml`, các script huấn luyện/đánh giá, notebook Colab và `README.md` hướng dẫn chi tiết), có thể chạy độc lập từng phần.

---

## 1. [`spmamba/`](spmamba/README.md) — Tách Nguồn Tiếng Nói Đa Người Nói (16 kHz)

- **Bài toán:** Tách 3 nguồn người nói từ hỗn hợp có nhiễu ở tần số lấy mẫu 16 kHz.
- **Đóng góp & Chỉnh sửa chính:**
  - Xây dựng pipeline nạp dữ liệu HDF5 tốc độ cao (`H5DataModule`) hỗ trợ chia tập tất định (`train`/`val`/`test`), cắt đoạn ngẫu nhiên khi huấn luyện và đánh giá toàn phần (full-length) khi kiểm thử.
  - Bổ sung cơ chế tự động khôi phục huấn luyện (`--resume`), lưu checkpoint định kỳ theo step/epoch và tích hợp theo dõi trực tuyến qua Weights & Biases (`--wandb`).
  - Cung cấp bộ cấu hình tối ưu cho nhiều dòng GPU: Colab T4 16GB (`configs/spmamba-3speakers.yml`), RTX 5080 16GB (`configs/spmamba-h5-5080.yml`), NVIDIA A100 80GB (`configs/spmamba-h5-a100.yml`) và NVIDIA H100 80GB (`configs/spmamba-h5-h100.yml`).
  - Công cụ kiểm toán độ phức tạp tính toán (`measure_macs.py`) có bù phần phép toán `selective_scan` trong kernel CUDA của Mamba.
- Xem hướng dẫn chi tiết tại: **[`spmamba/README.md`](spmamba/README.md)**.

---

## 2. [`hls/`](hls/README.md) — Tách Âm Tim/Phổi & Nhận Dạng Âm Bất Thường (HLS-CMDS, 4 kHz)

- **Bài toán:** Gồm 2 giai đoạn nối tiếp (Cascade) trên bộ dữ liệu âm thanh ống nghe số **HLS-CMDS** (4 kHz):
  1. **Giai đoạn 1 — Tách âm Tim và Phổi (`audio_train.py`, `test_hls.py`):** Huấn luyện SPMamba với hàm mất mát thứ tự cố định **`FixedOrderLossWrapper`** (thay cho PIT) để cố định kênh đầu ra `ch0 = Tim`, `ch1 = Phổi`, loại bỏ hoàn toàn sự nhập nhằng hoán vị kênh khi suy luận thực tế. Phân hoạch dữ liệu đảm bảo **tách biệt nguồn tuyệt đối** (*source-disjoint split*) giữa tập huấn luyện và tập kiểm thử.
  2. **Giai đoạn 2 — Nhận dạng âm bất thường (`train_hls_cls.py`, `test_hls_cls.py`):** Mô hình **`HLSClassifier`** (Log-Mel Frontend + CNN + Dual-Head) dự đoán đồng thời xác suất bất thường (nhị phân) và phân loại bệnh lý đa lớp (10 lớp âm tim, 6 lớp âm phổi) với kiểm định chéo lồng nhau 5-fold (*nested 5-fold cross-validation*).
- Xem hướng dẫn chi tiết tại: **[`hls/README.md`](hls/README.md)** và tài liệu bộ dữ liệu mở rộng **[`hls/DATASETS.md`](hls/DATASETS.md)**.

---

## 3. [`k210_distill/`](k210_distill/README.md) — Nén Mô Hình & Triển Khai Biên Trên Kendryte K210 (8 kHz)

- **Bài toán:** Đưa mô hình tách 3 người nói xuống hoạt động thời gian thực trên vi điều khiển biên **Kendryte K210** (6 MB SRAM + 2 MB KPU SRAM).
- **Phương pháp:**
  - Thiết kế kiến trúc lai **`K210SepUNet`** (~508K tham số KPU, ~1.48 MB SRAM) kết hợp mạng 2D Depthwise-Separable UNet chạy trên bộ tăng tốc phần cứng KPU và khối Dual-Path BiGRU + Multi-Head Self-Attention + ước lượng pha phức chạy trên nhân RISC-V 64-bit FPU.
  - Huấn luyện trên cụm máy chủ NVIDIA DGX A100 (`distill_train.py`, `configs/distill-k210-a100.yml`), xuất mô hình ONNX + tập dữ liệu hiệu chuẩn lượng tử hóa INT8 (`export_k210.py`) và mã nguồn firmware C chạy thời gian thực (`k210_firmware/main_k210_realtime.c`).
- Xem hướng dẫn chi tiết tại: **[`k210_distill/README.md`](k210_distill/README.md)**.

---

## Cài đặt môi trường

Mỗi thư mục đều đi kèm tệp cấu hình môi trường Conda `look2hear.yml` (Python 3.10, PyTorch 2.5.0, `mamba-ssm`, `pytorch-lightning`):

```bash
conda env create -f spmamba/look2hear.yml
conda activate look2hear
```
