# Tách Âm Tim/Phổi & Nhận Dạng Âm Bất Thường (HLS-CMDS)

Thư mục này chứa toàn bộ mã nguồn của hệ thống **2 giai đoạn nối tiếp (Cascade)** trên bộ dữ liệu âm thanh ống nghe số **HLS-CMDS** (tần số lấy mẫu 4 kHz):

1. **Giai đoạn 1 — Tách âm Tim và Phổi (`SPMamba` + `FixedOrderLossWrapper`):** Tách tín hiệu hỗn hợp thu từ lồng ngực thành hai kênh cố định: `ch0 = Âm tim (Heart Sound)` và `ch1 = Âm phổi (Lung Sound)`.
2. **Giai đoạn 2 — Nhận dạng âm bất thường (`HLSClassifier`):** Phân tích từng kênh đã tách để phát hiện bất thường (nhị phân Normal/Abnormal) và phân loại bệnh lý cụ thể (10 lớp bệnh tim, 6 lớp bệnh phổi).

```text
                       ┌──────────── Giai đoạn 1: Tách âm ────────────┐
 Hỗn hợp Tim + Phổi ──►│  SPMamba (FixedOrderLossWrapper, 4 kHz)      │
     [T] @ 4 kHz       └───────────┬──────────────────┬───────────────┘
                                   │ ch0 = Tim        │ ch1 = Phổi
                       ┌───────────▼──┐          ┌────▼─────────┐
                       │ HLSClassifier│          │ HLSClassifier│   ◄── Giai đoạn 2: Phân loại
                       │ branch=heart │          │ branch=lung  │
                       └───────┬──────┘          └──────┬───────┘
                   ┌───────────┴────────┐      ┌────────┴───────────┐
                   │ p(bất thường)      │      │ p(bất thường)      │
                   │ p(10 lớp bệnh tim) │      │ p(6 lớp bệnh phổi) │
                   └────────────────────┘      └────────────────────┘
```

---

## Cấu trúc thư mục

| File / Thư mục | Vai trò |
|---|---|
| `build_hls_h5.py` | Dựng file dữ liệu tách âm `data_hls.h5` (chia tập *source-disjoint* không rò rỉ nguồn) |
| `configs/spmamba-hls-t4.yml` | Cấu hình huấn luyện SPMamba tách âm tim/phổi trên GPU T4 |
| `audio_train.py` | Script huấn luyện mô hình tách âm SPMamba (Giai đoạn 1) |
| `test_hls.py` | Đánh giá mô hình tách âm (đo cả Fixed-order SI-SNRi/SDRi, PIT SI-SNRi và độ chính xác gán kênh) |
| `test_evaluate.py` | Module tiện ích nạp checkpoint SPMamba và tính SI-SNR |
| `build_hls_cls_h5.py` | Dựng file dữ liệu phân loại `data_hls_cls.h5` từ WAV gốc, kế thừa phân hoạch từ `data_hls.h5` |
| `configs/hls-cls-t4.yml` | Cấu hình siêu tham số cho bộ phân loại `HLSClassifier` |
| `smoke_hls_cls.py` | Kiểm tra tự động tính toàn vẹn dữ liệu, nhãn và mô hình phân loại trước khi train |
| `train_hls_cls.py` | Huấn luyện bộ phân loại âm tim/phổi (hỗ trợ cả split cố định và 5-fold CV) |
| `test_hls_cls.py` | Đánh giá bộ phân loại ở 3 chế độ: `clean`, `cascade-synth`, `cascade-real` |
| `hls_cls_metrics.py` | Các độ đo đánh giá phân loại (Macro-F1, Balanced Accuracy, Sensitivity, Specificity, AUROC) |
| `train_hls_colab.ipynb` | Notebook Colab huấn luyện tách âm Tim/Phổi (Giai đoạn 1) |
| `test_hls_colab.ipynb` | Notebook Colab kiểm thử và nghe mẫu âm thanh tách được (Giai đoạn 1) |
| `train_cls_colab.ipynb` | Notebook Colab huấn luyện phân loại và chạy 5-fold Cross-Validation (Giai đoạn 2) |
| `DATASETS.md` | Tổng hợp các bộ dữ liệu công khai (CirCor, CinC 2016, HF_Lung_V1, SPRSound) để pretrain/fine-tune |
| `look2hear.yml` | Cấu hình môi trường Conda |

---

## Phần I: Giai đoạn 1 — Tách Âm Tim và Âm Phổi

### 1. Tạo dữ liệu HDF5 (`data_hls.h5`)

Dữ liệu hỗn hợp được trộn động từ các bản ghi nguồn Tim (`HS`) và Phổi (`LS`) với tỷ lệ SNR ngẫu nhiên trong khoảng `[-5, +5] dB`. Quan trọng nhất, phép chia `train`/`val`/`test` được thực hiện **theo bản ghi nguồn gốc** (*source-disjoint*), đảm bảo không có bất kỳ bản ghi nguồn nào của tập test xuất hiện trong tập train:

```bash
cd hls
python build_hls_h5.py \
    --dataset-root "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope/HLS-CMDS" \
    --out /content/data_hls.h5
```

### 2. Huấn luyện mô hình tách âm SPMamba

Sử dụng `FixedOrderLossWrapper` để cố định thứ tự kênh đầu ra (`s1 = Tim`, `s2 = Phổi`):

```bash
python audio_train.py \
    --conf_dir configs/spmamba-hls-t4.yml \
    --checkpoint_dir checkpoints/SPMamba-HLS-HeartLung-T4 \
    --wandb
```

### 3. Đánh giá chất lượng tách âm

```bash
python test_hls.py \
    --ckpt checkpoints/SPMamba-HLS-HeartLung-T4 \
    --h5 /content/data_hls.h5 \
    --conf_dir configs/spmamba-hls-t4.yml \
    --outdir test_hls_out \
    --save_k 3
```

**Kết quả thực nghiệm trên tập kiểm thử source-disjoint (`n_test = 25`):**
- `Fixed SI-SNRi`: **+4.04 dB** (±2.14 dB)
- `PIT SI-SNRi`: **+4.04 dB** (±2.14 dB)
- `SDRi`: **+5.79 dB** (±2.53 dB)
- `Độ chính xác gán kênh (Channel Accuracy, perm == (0, 1))`: **100%** (chênh lệch PIT − Fixed = `0.00 dB`).

---

## Phần II: Giai đoạn 2 — Nhận Dạng Âm Tim/Phổi Bất Thường

Mỗi kênh sau khi tách được đưa vào nhánh `HLSClassifier` tương ứng để trả lời hai câu hỏi:
1. **Có bất thường không?** — Head nhị phân (đồng thời xuất điểm bất thường *anomaly score*).
2. **Bất thường loại nào?** — Head đa lớp (Tim: 10 lớp, Phổi: 6 lớp).

### Lưu ý quan trọng về đặc thù dữ liệu HLS-CMDS

1. **Chia tập theo bản ghi độc lập (Record-level Split):** Bộ dữ liệu gốc chỉ có **50 bản ghi độc lập mỗi nhánh** (lớp hiếm nhất S4 chỉ có 2 bản ghi). Mỗi bản ghi dài 15 giây được cắt thành nhiều cửa sổ 5 giây khi huấn luyện, nhưng mọi phép chia tập trong kho mã nguồn này đều thực hiện nghiêm ngặt **theo bản ghi gốc**, tuyệt đối không chia theo cửa sổ để tránh rò rỉ dữ liệu.
2. **Báo cáo bằng Nested 5-Fold Cross-Validation:** Một lần chia cố định chỉ có ~5 bản ghi test, nên số liệu báo cáo chính thức phải lấy từ **kiểm định chéo 5-fold lồng nhau** (`--fold 0..4`).
3. **Dữ liệu thu từ manikin lâm sàng:** Dữ liệu HLS-CMDS được thu từ manikin lâm sàng bằng ống nghe số; khi trình bày trong báo cáo/luận văn cần nêu rõ phạm vi đánh giá trên manikin và tham khảo thêm hướng mở rộng sang dữ liệu bệnh nhân thực tế tại [`DATASETS.md`](DATASETS.md).

### Quy trình chạy bằng dòng lệnh (CLI)

```bash
# 1) Dựng dữ liệu phân loại, kế thừa phân hoạch nguồn của bộ tách âm
python build_hls_cls_h5.py \
    --dataset-root "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope/HLS-CMDS" \
    --align-split-with /content/data_hls.h5 \
    --out /content/data_hls_cls.h5

# 2) Kiểm tra tự động (Smoke test) trước khi huấn luyện
python smoke_hls_cls.py \
    --h5 /content/data_hls_cls.h5 \
    --dataset_root "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope/HLS-CMDS"

# 3) Huấn luyện từng nhánh (Tim & Phổi)
python train_hls_cls.py --branch heart --h5 /content/data_hls_cls.h5 \
    --conf_dir configs/hls-cls-t4.yml --out_dir out/heart --wandb
python train_hls_cls.py --branch lung  --h5 /content/data_hls_cls.h5 \
    --conf_dir configs/hls-cls-t4.yml --out_dir out/lung  --wandb

# 4) Đánh giá toàn diện trên cả 3 chế độ (clean, cascade-synth, cascade-real)
python test_hls_cls.py --mode all \
    --heart_ckpt out/heart/best.pt \
    --lung_ckpt out/lung/best.pt \
    --cls_h5 /content/data_hls_cls.h5 \
    --sep_ckpt checkpoints/SPMamba-HLS-HeartLung-T4 \
    --h5 /content/data_hls.h5 \
    --out_json out/ket_qua.json

# 5) Chạy Cross-Validation 5-Fold để lấy số liệu báo cáo
for k in 0 1 2 3 4; do
  python train_hls_cls.py --branch heart --fold $k \
      --h5 /content/data_hls_cls.h5 --conf_dir configs/hls-cls-t4.yml \
      --out_dir out/cv/heart
  python train_hls_cls.py --branch lung --fold $k \
      --h5 /content/data_hls_cls.h5 --conf_dir configs/hls-cls-t4.yml \
      --out_dir out/cv/lung
done
```

### Ý nghĩa 3 chế độ đánh giá trong `test_hls_cls.py`

| Chế độ | Đầu vào | Vai trò |
|---|---|---|
| `clean` | Bản ghi âm tim/phổi thuần | **Trần trên (Upper bound)** — chất lượng bộ phân loại khi đầu vào hoàn hảo |
| `cascade-synth` | Hỗn hợp kiểm thử → SPMamba → Phân loại | **Kết quả End-to-End chính**, đảm bảo không rò rỉ dữ liệu |
| `cascade-real` | 145 file `M####.wav` thu hỗn hợp → SPMamba → Phân loại | **Tham khảo**, do các bản ghi nguồn tạo nên 145 file này trùng lặp với 50 bản ghi huấn luyện |

---

## Các cơ chế bảo vệ kỹ thuật đã tích hợp

- **Trích xuất nhãn chuẩn xác kèm đối chiếu dấu vân tay CSV:** `LS.csv` của dataset gốc ghi mã viết tắt khác với tên file (`C` = Fine Crackles, `G` = Coarse Crackles trong khi tên file dùng `FC`, `CC`). Module `look2hear/datas/hls_labels.py` trích xuất nhãn chuẩn và đối chiếu đa tập `(giới tính, vị trí nghe)` với CSV để ngăn chặn mọi sai lệch nhãn.
- **Cross-validation lồng nhau (Nested CV):** Fold `k` dùng để chấm điểm báo cáo, fold `(k+1) % 5` dùng để chọn epoch tốt nhất (early stopping), các fold còn lại dùng để huấn luyện. Các bản ghi thuộc tập test của bộ tách âm được gán `fold = -1` và tách biệt hoàn toàn.
- **Chốt chặn chống rò rỉ tự động (`assert_unseen`):** Mỗi checkpoint lưu danh sách `train_record_ids`; khi chấm điểm, nếu phát hiện bất kỳ bản ghi đánh giá nào nằm trong tập train của checkpoint đó, chương trình sẽ báo lỗi và dừng ngay lập tức.
- **Frontend Log-Mel chạy ở độ chính xác `float32` ngay cả khi bật AMP:** Tránh hiện tượng tràn dưới (*underflow*) với hằng số `1e-6` trong `log(mel + eps)` ở `float16` trên các dải tần năng lượng thấp chứa tiếng thổi tâm thu hoặc tiếng ran phổi.
