# Nhận dạng âm tim/phổi bất thường (HLS-CMDS)

Giai đoạn nối tiếp phần tách âm SPMamba: sau khi tách hỗn hợp thành kênh tim và
kênh phổi, nhận dạng **âm thanh bất thường** trên từng kênh.

```
                       ┌──────────── giai đoạn đã có ────────────┐
 hỗn hợp tim+phổi ───► │  SPMamba  (epoch=36-best / ckpt HLS)    │
    [T] @ 4 kHz        └───────────┬──────────────┬──────────────┘
                                   │ ch0 = tim    │ ch1 = phổi
                       ┌───────────▼──┐      ┌────▼─────────┐
                       │ HLSClassifier│      │ HLSClassifier│    ◄── giai đoạn mới
                       │  branch=heart│      │  branch=lung │
                       └───────┬──────┘      └──────┬───────┘
                   ┌───────────┴────────┐  ┌────────┴───────────┐
                   │ p(bất thường)      │  │ p(bất thường)      │
                   │ p(10 lớp bệnh tim) │  │ p(6 lớp bệnh phổi) │
                   └────────────────────┘  └────────────────────┘
```

Mỗi kênh trả lời hai câu hỏi:

1. **Có bất thường không?** — head nhị phân, đồng thời dùng làm điểm anomaly score.
2. **Bất thường loại nào?** — head đa lớp (tim 10 lớp, phổi 6 lớp).

---

## Đọc phần này trước khi chạy

Ba điều quyết định cách diễn giải mọi con số bạn sẽ thấy.

**Dữ liệu ít hơn vẻ ngoài rất nhiều.** Không phải 1000 mẫu — chỉ có **50 bản ghi
độc lập mỗi nhánh**, lớp hiếm nhất (S4) có đúng 2 bản ghi. Mỗi bản ghi dài 15 s
được cắt thành nhiều cửa sổ, nhưng các cửa sổ của cùng một bản ghi **không** độc
lập với nhau. Vì vậy mọi phép chia trong repo này đều theo *bản ghi*, không theo
cửa sổ. Chia theo cửa sổ sẽ cho accuracy rất cao và hoàn toàn vô nghĩa.

**Số liệu để báo cáo lấy từ cross-validation**, không phải từ một lần chia. Một
lần chia chỉ cho ~5 bản ghi test; riêng với split kế thừa từ `data_hls.h5`, tập
test nhánh tim **không có bản ghi Normal nào**, nên F1 và AUROC nhị phân trên đó
không dùng được. Script sẽ tự in cảnh báo. Dùng Cell 9 của notebook.

**Dữ liệu thu từ manikin lâm sàng, không phải bệnh nhân thật.** Mô hình huấn
luyện và kiểm thử hoàn toàn trên manikin **không chứng minh được gì về bệnh nhân
thật**, dù con số có đẹp đến đâu. Nếu bạn viết luận văn, điều này phải được nêu rõ.

---

## Cách chạy (Colab)

Mở `train_cls_colab.ipynb` và chạy Cell 1 → 9 theo thứ tự.

Bạn cần chuẩn bị sẵn trên Google Drive:

| Thứ | Dùng để | Biến trong Cell 1 |
|---|---|---|
| `data_hls.zip` (chứa `data_hls.h5`) | kế thừa phân hoạch nguồn của bộ tách âm | `DRIVE_H5_ZIP` |
| `HLS-CMDS.zip` (dataset gốc, có file `.wav`) | dựng dữ liệu phân loại | `DRIVE_DATASET_ZIP` |
| Checkpoint SPMamba đã train | đánh giá cascade (Cell 8) | `SEP_CKPT_DIR` |

> **Lưu ý:** file `*.wav` nằm trong `.gitignore` nên clone repo **không** có audio —
> chỉ có 3 file CSV. Bắt buộc phải upload zip dataset gốc lên Drive.

| Cell | Việc | Thời gian |
|---|---|---|
| 1–3 | đường dẫn, môi trường, clone repo | vài phút |
| 4 | dựng `data_hls_cls.h5` | < 1 phút |
| 5 | **kiểm tra nhanh** — chặn lại nếu có gì sai | vài chục giây |
| 6–7 | train bộ phân loại tim, rồi phổi | ~10–20 phút mỗi nhánh trên T4 |
| 8 | đánh giá 3 chế độ | vài phút |
| 9 | cross-validation 5 fold — **số liệu để báo cáo** | ~2 giờ |

Đừng bỏ qua Cell 5. Nó chạy vài chục giây và sẽ bắt lỗi trước khi bạn tốn một giờ
huấn luyện.

## Cách chạy (dòng lệnh)

```bash
# 1) Dựng dữ liệu phân loại, kế thừa phân hoạch nguồn của bộ tách âm
python build_hls_cls_h5.py \
    --dataset-root "hls-cmds_.../HLS-CMDS" \
    --align-split-with /content/data_hls.h5 \
    --out /content/data_hls_cls.h5

# 2) Kiểm tra nhanh
python smoke_hls_cls.py --h5 /content/data_hls_cls.h5 \
    --dataset_root "hls-cmds_.../HLS-CMDS"

# 3) Train từng nhánh
python train_hls_cls.py --branch heart --h5 /content/data_hls_cls.h5 \
    --conf_dir configs/hls-cls-t4.yml --out_dir out/heart --wandb
python train_hls_cls.py --branch lung  --h5 /content/data_hls_cls.h5 \
    --conf_dir configs/hls-cls-t4.yml --out_dir out/lung  --wandb

# 4) Đánh giá cả ba chế độ
python test_hls_cls.py --mode all \
    --heart_ckpt out/heart/best.pt --lung_ckpt out/lung/best.pt \
    --cls_h5 /content/data_hls_cls.h5 \
    --sep_ckpt /content/drive/.../SPMamba-HLS-HeartLung-T4 \
    --h5 /content/data_hls.h5 --out_json out/ket_qua.json

# 5) Cross-validation - SỐ LIỆU ĐỂ BÁO CÁO
for k in 0 1 2 3 4; do
  python train_hls_cls.py --branch heart --fold $k \
      --h5 /content/data_hls_cls.h5 --conf_dir configs/hls-cls-t4.yml \
      --out_dir out/cv/heart
done
```

---

## Đọc kết quả

Script đánh giá in ba chế độ, xếp theo độ sạch giảm dần:

| Chế độ | Đầu vào | Vai trò |
|---|---|---|
| `clean` | bản ghi tim/phổi thuần | **trần trên** — bộ phân loại làm được gì khi đầu vào hoàn hảo |
| `cascade-synth` | hỗn hợp tổng hợp → SPMamba → phân loại | **kết quả chính**, không rò rỉ |
| `cascade-real` | 145 file thu thật → SPMamba → phân loại | **tham khảo**, có rò rỉ |

`cascade-real` thực tế nhất nhưng **không được trích dẫn làm kết quả chính**: các
take `H####`/`L####` dựng nên những hỗn hợp đó trùng lặp nặng với 50 bản ghi dùng
để huấn luyện (53/145 file tim và 57/145 file phổi trùng byte-for-byte). Script in
cảnh báo ngay trong bảng.

Bảng cuối cùng so sánh `clean` với `cascade-synth`. **Chênh lệch đó chính là phần
chất lượng mất đi do bước tách âm** — con số đáng giá nhất của cả thí nghiệm.

Chỉ số nên nhìn: **macro-F1** và **balanced accuracy**, không phải accuracy. Đoán
"bất thường" cho mọi mẫu đã cho 82% accuracy ở nhánh tim mà không học được gì.
Trong bối cảnh y tế, **sensitivity** (tỉ lệ ca bất thường bắt được) quan trọng hơn
specificity vì bỏ sót nguy hiểm hơn báo động nhầm.

---

## Các file

| File | Vai trò |
|---|---|
| `look2hear/datas/hls_labels.py` | bảng nhãn + đối chiếu chéo CSV |
| `build_hls_cls_h5.py` | dựng `data_hls_cls.h5` từ WAV thuần |
| `look2hear/datas/hls_cls_datamodule.py` | dataset cắt cửa sổ, tăng cường, chia fold |
| `look2hear/models/hls_classifier.py` | log-mel + CNN + 2 head (~290K tham số) |
| `train_hls_cls.py` | vòng huấn luyện + W&B |
| `test_hls_cls.py` | đánh giá 3 chế độ |
| `hls_cls_metrics.py` | chỉ số (numpy thuần, không cần scikit-learn) |
| `smoke_hls_cls.py` | kiểm tra nhanh trước khi train |
| `configs/hls-cls-t4.yml` | siêu tham số |
| `train_cls_colab.ipynb` | notebook điều khiển toàn bộ |

---

## Những cạm bẫy đã được xử lý

Ghi lại để bạn không vô tình gỡ bỏ chúng khi sửa code.

**Nhãn lấy từ tên file, không lấy từ cột ID của CSV.** `LS.csv` dùng mã viết tắt
khác tên file: CSV ghi `C` = Fine Crackles và `G` = Coarse Crackles, còn tên file
dùng `FC` và `CC`. Tra nhãn theo mã của CSV sẽ **đảo ngược Coarse ↔ Fine**.

**Đối chiếu chéo dựa vào dấu vân tay, không dựa vào số lượng.** Nhiều lớp có cùng
số bản ghi — phổi có Pleural Rub và Coarse Crackles đều 9; tim có ba cặp như vậy —
nên hoán đổi hai lớp sẽ lọt qua phép đếm. Phép kiểm so đa tập `(giới tính, vị trí)`
của từng lớp, và bắt được mọi hoán đổi.

**Cross-validation lồng.** Fold `k` để báo cáo, fold `k+1` để chọn epoch, phần còn
lại để train. Nếu dùng chính fold báo cáo để dừng sớm thì con số bị thiên lệch lạc
quan — mỗi fold chỉ ~9 bản ghi nên nhiễu rất lớn. `summary.json` ghi kèm cờ
`selection_set_equals_eval_set`.

**Bản ghi thuộc pool test của bộ tách âm nhận `fold = -1`** và bị loại khỏi
cross-validation, để thí nghiệm cascade giữ được bảo đảm không rò rỉ.

**Checkpoint lưu `train_record_ids`.** Lúc đánh giá, chương trình **dừng** nếu
checkpoint từng huấn luyện trên bản ghi đang chấm — một checkpoint chấm nhầm tập
sẽ cho điểm rất cao và trông hoàn toàn hợp lệ.

**Frontend log-mel chạy float32 kể cả khi bật AMP.** Ở float16, `1e-6` trong
`log(mel + eps)` là số dưới chuẩn, nên các dải mel năng lượng thấp — đúng chỗ chứa
tiếng thổi nhẹ và tiếng ran — bị bóp về cùng một giá trị.

**Bản ghi không có cửa sổ nào làm chương trình báo lỗi**, thay vì trả hàng toàn 0.
Hàng toàn 0 qua `argmax` cho lớp 0 = "Normal": một ca mất dữ liệu sẽ âm thầm được
báo là bình thường, kiểu hỏng nguy hiểm nhất với công cụ sàng lọc.

**Không đảo dấu khi tăng cường dữ liệu.** Frontend dùng phổ công suất mà
`|STFT(−x)|² = |STFT(x)|²`, nên phép này không đổi một bit nào ở đầu vào mô hình.

**Không dịch vòng cửa sổ.** Dịch vòng một đoạn đã cắt tạo bước nhảy biên độ ở chỗ
nối — một tiếng "click" nhân tạo, mà tiếng click chính là đặc trưng của Fine
Crackles. Tính bất biến theo thời gian đã có nhờ cắt ngẫu nhiên vị trí cửa sổ.

---

## Xử lý sự cố

**`Khong thay dataset goc HLS-CMDS co du 50+50 file .wav`**
Chưa upload zip dataset lên Drive, hoặc `DRIVE_DATASET_ZIP` trỏ sai. Nhớ rằng repo
có thư mục `HLS-CMDS/HS/HS` nhưng rỗng audio — Cell 4 cố tình bỏ qua nó.

**`So ban ghi moi lop theo ten file khong khop CSV`**
Bảng viết tắt trong `hls_labels.py` sai, hoặc dataset đã bị đổi tên file. Thông báo
lỗi liệt kê chính xác lớp nào lệch và lệch ở đâu.

**`RO RI DU LIEU o che do ...`**
Bạn đang chấm một checkpoint cross-validation bằng sai fold. Dùng đúng `--fold` của
checkpoint đó, hoặc dùng checkpoint train trên split cố định.

**`ban ghi khong xuat hien trong H5 tach am`**
`data_hls.h5` không phải file đã dùng để train bộ tách âm. Dùng đúng file, hoặc
thêm `--allow-unaligned-records` để xếp chúng vào `unused` (loại khỏi cả train lẫn
eval — an toàn hơn là đưa vào train).

**`split 'test' chi co lop Abnormal`**
Đúng như dự kiến với split kế thừa ở nhánh tim. Dùng cross-validation (Cell 9) để
lấy số liệu báo cáo.

---

## Mở rộng dữ liệu

50 bản ghi mỗi nhánh là quá ít để kết luận về 10 lớp. Xem
[DATASETS.md](DATASETS.md) để biết các nguồn công khai dùng được cho pretrain và
fine-tune.
