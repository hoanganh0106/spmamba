# SPMamba — Tách Nguồn Tiếng Nói Đa Người Nói (16 kHz)

Thư mục này chứa mã nguồn **SPMamba** đã được tinh chỉnh để huấn luyện và đánh giá bài toán tách nguồn âm thanh đa người nói (3 người nói + nhiễu ở tần số 16 kHz) trên định dạng dữ liệu HDF5 (`H5DataModule`).

---

## Cấu trúc thư mục

| File / Thư mục | Mô tả |
|---|---|
| `look2hear/` | Thư viện lõi chứa mô hình `SPMamba`, `H5DataModule`, hàm mất mát `PITLossWrapper`, `AudioLightningModule` |
| `configs/spmamba-3speakers.yml` | Cấu hình huấn luyện tối ưu cho Google Colab T4 (16 GB VRAM, FP16) |
| `configs/spmamba-h5-5080.yml` | Cấu hình huấn luyện cho GPU RTX 5080 (16 GB VRAM, BF16) |
| `configs/spmamba-h5-a100.yml` | Cấu hình huấn luyện cho GPU NVIDIA A100 (80 GB VRAM, BF16) |
| `configs/spmamba-h5-h100.yml` | Cấu hình huấn luyện cho GPU NVIDIA H100 (80 GB VRAM, BF16) |
| `audio_train.py` | Script huấn luyện chính (hỗ trợ `--resume`, `--checkpoint_dir`, `--wandb`) |
| `audio_test.py` | Script đánh giá hàng loạt trên tập test theo cấu hình YAML |
| `test_evaluate.py` | Script tách nguồn từ file WAV hỗn hợp và tính chỉ số SI-SNRi, SDRi |
| `inference.py` | Script suy luận nhanh tách 1 file âm thanh hỗn hợp ra các kênh riêng biệt |
| `measure_macs.py` | Công cụ đo số tham số và MACs/FLOPs chính xác (có bù phần `selective_scan` của Mamba) |
| `cal_flops.py` | Script tính FLOPs nguyên bản của tác giả SPMamba |
| `spmamba3.ipynb` | Notebook huấn luyện trên Google Colab kèm theo dõi trực tuyến qua Weights & Biases |
| `spmamba_test.ipynb` | Notebook tách thử file âm thanh và chấm điểm trực quan trên Colab |
| `measure_macs_colab.ipynb` | Notebook đo MACs/FLOPs tự động trên Colab |
| `look2hear.yml` | File môi trường Conda |

---

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
cd spmamba
conda env create -f look2hear.yml
conda activate look2hear
```

### 2. Huấn luyện (Training)

Chọn file cấu hình phù hợp với phần cứng GPU đang sử dụng:

```bash
# Huấn luyện trên NVIDIA A100 (80GB VRAM)
python audio_train.py --conf_dir configs/spmamba-h5-a100.yml

# Huấn luyện trên Google Colab T4 (16GB VRAM) + lưu checkpoint sang thư mục chỉ định + bật W&B
python audio_train.py \
    --conf_dir configs/spmamba-3speakers.yml \
    --checkpoint_dir /content/drive/MyDrive/SPMamba_checkpoints/SPMamba-3Speakers \
    --wandb

# Tiếp tục huấn luyện (Resume) từ checkpoint gần nhất
python audio_train.py \
    --conf_dir configs/spmamba-3speakers.yml \
    --checkpoint_dir /content/drive/MyDrive/SPMamba_checkpoints/SPMamba-3Speakers \
    --resume
```

### 3. Đánh giá và Tách thử âm thanh (Evaluation & Inference)

```bash
# Tách nguồn và chấm điểm SI-SNRi / SDRi đối chiếu với nguồn tham chiếu
python test_evaluate.py \
    --ckpt Experiments/checkpoint/SPMamba-3Speakers-ColabT4/last.ckpt \
    --mix mix.wav \
    --sources s1.wav s2.wav s3.wav \
    --outdir test_output

# Đo độ phức tạp tính toán (Params & MACs cho 1 giây âm thanh 16 kHz)
python measure_macs.py --conf_dir configs/spmamba-3speakers.yml --seconds 1.0
```
