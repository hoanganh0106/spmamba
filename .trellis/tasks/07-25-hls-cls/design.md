# Design — HLS heart/lung abnormal sound recognition

## Sơ đồ tổng thể

```
                      ┌──────────────── giai đoạn đã có ────────────────┐
hỗn hợp tim+phổi  ──► │  SPMamba (epoch=36-best / ckpt HLS fixed-order) │
   [T] @4kHz          └──────────────┬──────────────┬──────────────────┘
                                     │ ch0 = tim    │ ch1 = phổi
                        ┌────────────▼───┐     ┌────▼────────────┐
                        │ HLSClassifier  │     │ HLSClassifier   │   ◄── phần mới
                        │   branch=heart │     │  branch=lung    │
                        └────────┬───────┘     └────────┬────────┘
                     ┌───────────┴──────┐    ┌──────────┴───────┐
                     │ p(bất thường)    │    │ p(bất thường)    │
                     │ p(10 lớp bệnh)   │    │ p(6 lớp bệnh)    │
                     └──────────────────┘    └──────────────────┘
```

Hai bộ phân loại dùng **chung một lớp mô hình**, khác nhau ở `n_classes` và
checkpoint. Chúng được huấn luyện trên âm **thuần** (`HS/`, `LS/`) rồi áp dụng lên
đầu ra tách âm — chấp nhận có domain gap, và chính khoảng cách Clean → Cascade là
con số cần báo cáo.

## Ranh giới module

| File | Vai trò | Ghi chú |
|---|---|---|
| `look2hear/datas/hls_labels.py` | Bảng nhãn, phân tích tên file, kiểm tra chéo CSV | Không phụ thuộc torch |
| `build_hls_cls_h5.py` | Sinh `data_hls_cls.h5` từ WAV thuần + gán split | CLI, chạy một lần |
| `look2hear/datas/hls_cls_datamodule.py` | Dataset cắt cửa sổ + tăng cường + DataLoader | Nhóm theo bản ghi |
| `look2hear/models/hls_classifier.py` | Frontend log-mel + CNN + 2 head | Đăng ký vào `look2hear.models` |
| `train_hls_cls.py` | Vòng huấn luyện + W&B + lưu checkpoint | PyTorch thuần |
| `test_hls_cls.py` | Đánh giá 3 chế độ (clean / cascade tổng hợp / cascade thật) | Dùng lại `test_evaluate.load_model` |
| `configs/hls-cls-t4.yml` | Siêu tham số cho Colab T4 | Cùng phong cách config hiện có |

## Bảng nhãn (`hls_labels.py`)

Nguồn chân lý là **mã viết tắt trong tên file**, không phải cột ID của CSV:

```python
HEART_ABBR = {"N": "Normal", "AF": "Atrial Fibrillation", "AVB": "AV Block",
              "T": "Tachycardia", "S3": "S3", "S4": "S4",
              "ESM": "Early Systolic Murmur", "MSM": "Mid Systolic Murmur",
              "LSM": "Late Systolic Murmur", "LDM": "Late Diastolic Murmur"}
LUNG_ABBR  = {"N": "Normal", "CC": "Coarse Crackles", "FC": "Fine Crackles",
              "PR": "Pleural Rub", "R": "Rhonchi", "W": "Wheezing"}
```

CSV được dùng **chỉ để kiểm tra chéo**, và phép kiểm không dựa vào số lượng mà dựa
vào **dấu vân tay của từng lớp: đa tập các cặp `(giới tính, vị trí)`**.

Lý do phải mạnh hơn phép đếm: nhiều lớp có cùng số bản ghi, nên hoán đổi hai lớp
trong bảng viết tắt sẽ lọt qua phép đếm.

| Nhánh | Các cặp lớp trùng số lượng |
|---|---|
| Phổi | Pleural Rub / Coarse Crackles (đều 9) |
| Tim | LDM / ESM (6), LSM / S3 (5), Tachycardia / AV Block (3) |

Cột ID của CSV tuy dùng mã viết tắt khác ở nhánh phổi, nhưng **phần giới tính và
phần vị trí thì trùng với tên file**, nên so được. Đã kiểm chứng: không có hai lớp
nào trùng dấu vân tay, và cố tình hoán đổi Pleural Rub ↔ Coarse Crackles hay
Tachycardia ↔ AV Block đều bị bắt (phép đếm thì không bắt được).

Hàm kiểm còn cảnh báo nếu về sau xuất hiện hai lớp trùng dấu vân tay, để không
tưởng nhầm là đã kiểm hết.

Tên file dạng `{Gender}_{Abbr}_{Location}` → `group_id` chính là cả tên file
(ví dụ `F_MSM_LUSB`), tức mỗi bản ghi là một nhóm riêng.

`is_abnormal = (label != "Normal")`.

## Schema `data_hls_cls.h5`

Chỉ 100 bản ghi × 60000 mẫu float32 ≈ 24 MB, nên nạp thẳng dạng mảng phẳng, không
cần nhóm lồng nhau như file tách âm.

```
attrs: sample_rate=4000, target_length=60000, created_at,
       split_source = "<đường dẫn data_hls.h5>" | "stratified"
/heart/audio       float32 [50, 60000]
/heart/record_ids  str     [50]        # "F_MSM_LUSB"
/heart/label_idx   int64   [50]        # chỉ số trong /heart.attrs["classes"]
/heart/is_abnormal int64   [50]        # 0/1
/heart/split       str     [50]        # "train" | "val" | "test"
/heart/fold        int64   [50]        # 0..K-1 cho grouped stratified K-fold
/heart.attrs: classes (JSON list), normal_class="Normal", n_folds
/lung/  … cấu trúc y hệt, 6 lớp
```

### Gán split — điểm mấu chốt chống rò rỉ

`build_hls_h5.py` đã chia **nguồn** thành các pool rời nhau và ghi lại dấu vết:
mỗi nhóm hỗn hợp trong `data_hls.h5` có thuộc tính `heart`, `lung` (tên bản ghi
gốc) và `split`. Vì các pool rời nhau nên mỗi bản ghi chỉ thuộc đúng một split.

Do đó `build_hls_cls_h5.py --align-split-with data_hls.h5` sẽ:

1. Duyệt `data_hls.h5`, dựng `{tên bản ghi tim → split}` và `{tên bản ghi phổi → split}`.
2. **Khẳng định mỗi bản ghi chỉ ánh xạ tới một split**; nếu không, dừng (dấu hiệu
   file H5 được build bằng `--no-source-split`).
3. Bản ghi nào không xuất hiện trong `data_hls.h5` thì xếp vào `train`.

Kết quả: tập test của bộ phân loại nằm trọn trong tập nguồn mà SPMamba chưa từng
thấy, nên thí nghiệm cascade tổng hợp là sạch. Không có cờ này thì rơi về chia
phân tầng theo nhóm với hạt giống cố định.

Cột `fold` luôn được sinh (grouped stratified K-fold, mặc định K=5) để báo cáo
cross-validation mà không phải build lại file. **Nó không độc lập với cột `split`**:
bản ghi có `split ∈ {test, unused}` nhận `fold = -1` và bị loại khỏi mọi fold. Nếu
để chúng lọt vào tập train của một fold thì bộ phân loại đã nhìn thấy nguồn của các
hỗn hợp dùng cho thí nghiệm cascade — đúng thứ mà `--align-split-with` sinh ra để
ngăn.

Bản ghi không truy được về `data_hls.h5` sẽ làm script **dừng với lỗi**, chứ không
mặc định xếp vào `train`. Mặc định `train` là hướng duy nhất có thể âm thầm có lợi
cho tập huấn luyện. Có cờ `--allow-unaligned-records` để xếp chúng vào `unused`
(loại khỏi cả train lẫn eval).

Phép chia phân tầng dùng **phân bổ theo phần dư lớn nhất (Hamilton)**, không làm
tròn từng lớp: `round(5 × 0.10) = 0` theo quy tắc làm tròn về số chẵn, nên làm tròn
từng lớp sẽ loại hẳn mọi lớp có ≤5 bản ghi khỏi tập test — đúng những lớp hiếm nhất.

## Dataset và tăng cường

Bản ghi 15 s được cắt thành cửa sổ `segment` giây (mặc định 5 s = 20000 mẫu):

- **Train**: mỗi lần lấy mẫu cắt ngẫu nhiên một cửa sổ trong bản ghi. `__len__` =
  `n_records × repeat` (mặc định `repeat=8`) để mỗi epoch nhìn đủ nhiều biến thể.
- **Val/test**: cửa sổ trượt tất định (hop = `segment/2`), giữ lại `record_idx` để
  gộp kết quả lên mức bản ghi.

Tăng cường (chỉ khi train): đổi biên độ (±6 dB) → nhiễu Gauss (SNR 20–40 dB) → đảo
dấu ngẫu nhiên. SpecAugment (che tối đa 2 dải tần rộng 8 bin, 2 khoảng thời gian
rộng 16 khung) áp dụng **sau** frontend, đặt trong mô hình vì nó thao tác trên phổ,
và che **riêng cho từng mẫu** trong lô — dùng chung một mặt nạ cho cả lô 32 mẫu sẽ
chỉ cho 1 biến thể thay vì 32, mất gần hết tác dụng chống overfit.

**Cố ý không dịch vòng theo thời gian.** Dịch vòng một cửa sổ đã cắt sẽ dán cuối
đoạn vào đầu đoạn, tạo bước nhảy biên độ tại chỗ nối — tức một tiếng "click" băng
thông rộng nhân tạo. Với bài toán này đó là tăng cường phản tác dụng: tiếng click
chính là đặc trưng của Fine/Coarse Crackles, dạy mô hình coi click là bình thường
sẽ làm hỏng đúng khả năng cần có. Tính bất biến theo thời gian đã có sẵn nhờ cắt
ngẫu nhiên vị trí cửa sổ trong bản ghi 15 s.

## Mô hình `HLSClassifier`

**Frontend log-mel tự cài** bằng `torch.stft` + ma trận mel dựng bằng numpy — cố
tình không dùng `torchaudio`/`librosa` để không thêm phụ thuộc runtime, và để
frontend khả vi nếu sau này muốn fine-tune chung với SPMamba.

Tham số ở 4 kHz: `n_fft=256` (64 ms), `hop=64` (16 ms), `n_mels=64`, `fmin=20`,
`fmax=2000`. Cửa sổ 5 s → phổ `[64 mel × 313 khung]`. Dải này phủ đủ nội dung
tim (20–600 Hz) và phổi (100–2000 Hz) ở tần số lấy mẫu 4 kHz.

Chuẩn hoá: log(mel + 1e-6) rồi chuẩn hoá theo từng mẫu (trừ trung bình, chia độ
lệch chuẩn) — làm cho mô hình bất biến với mức âm lượng, quan trọng vì SPMamba xuất
ra biên độ khác với bản ghi thuần.

**Frontend chạy float32 kể cả khi bật AMP** (`autocast(enabled=False)`). Nếu để
autocast tác động, phép nhân ma trận mel chạy float16 và `log(mel + 1e-6)` mất độ
chính xác nghiêm trọng: `1e-6` là số dưới chuẩn trong float16 (số chuẩn nhỏ nhất
≈ 6e-5), nên các dải mel năng lượng thấp — đúng chỗ chứa tiếng thổi nhẹ và tiếng
ran — bị bóp về cùng một giá trị.

Thân CNN — 3 khối, mỗi khối 2 lớp tích chập 3×3 + BatchNorm + ReLU + MaxPool 2×2 +
Dropout2d, số kênh 32 → 64 → 128. Sau đó gộp trung bình theo trục tần số, rồi
**gộp có trọng số chú ý theo trục thời gian** (một lớp tuyến tính sinh trọng số,
softmax theo thời gian) — phù hợp vì bất thường tim/phổi là sự kiện cục bộ
(tiếng thổi, tiếng ran) chứ không trải đều toàn đoạn.

Hai head trên vector 128 chiều dùng chung: `Linear(128, n_classes)` và
`Linear(128, 1)`. Tổng khoảng 250K tham số.

**Hàm mất mát**: `CrossEntropy(có trọng số lớp) + λ · BCEWithLogits(pos_weight)`,
mặc định `λ = 0.5`. Trọng số lớp = `n_total / (n_classes · n_lớp_đó)`.

Head nhị phân về mặt toán học có thể suy ra từ head đa lớp, nhưng vẫn giữ riêng vì
(a) nó cho điểm số hiệu chỉnh tốt hơn cho ngưỡng bất thường, và (b) khi mở rộng
sang phát hiện bất thường không giám sát thì nó là đầu ra sẵn sàng dùng lại. Lúc
đánh giá in **cả hai**: nhị phân trực tiếp và nhị phân suy ra từ head đa lớp.

## Cross-validation lồng

Với `--fold k`, ba vai trò tách hẳn nhau:

| Vai trò | Bản ghi | Dùng để |
|---|---|---|
| test | fold `k` | **báo cáo** — không bao giờ tham gia chọn epoch |
| val | fold `(k+1) mod K` | chọn epoch tốt nhất, dừng sớm |
| train | các fold còn lại | huấn luyện |

Nếu dùng chính fold `k` vừa để dừng sớm vừa để báo cáo, con số bị thiên lệch lạc
quan: ta đã lấy epoch trông đẹp nhất trên đúng tập đem báo cáo, mà mỗi fold chỉ có
~9 bản ghi nên nhiễu rất lớn. `summary.json` ghi kèm cờ
`selection_set_equals_eval_set` để người đọc kết quả biết ngay, thay vì phải đọc mã
nguồn.

Checkpoint lưu cả `train_record_ids`. Lúc đánh giá, `assert_unseen()` dừng chương
trình nếu checkpoint từng huấn luyện trên bản ghi đang chấm — chốt chặn cuối cùng,
vì một checkpoint cross-validation chấm nhầm tập sẽ cho điểm rất cao và trông hoàn
toàn hợp lệ.

## Giao thức đánh giá

Suy luận luôn ở mức cửa sổ rồi gộp lên bản ghi bằng **trung bình xác suất** (ổn
định hơn bỏ phiếu đa số khi số cửa sổ ít). Bản ghi không nhận được cửa sổ nào làm
hàm gộp **báo lỗi**, vì hàng toàn 0 đi qua `argmax` sẽ cho lớp 0 = "Normal", tức
âm thầm báo một ca mất dữ liệu là bình thường.

| Chế độ | Đầu vào | Nhãn | Vai trò |
|---|---|---|---|
| `clean` | WAV thuần, split=test | tên file | trần trên |
| `cascade-synth` | test split `data_hls.h5` → SPMamba | attrs `heart`/`lung` | **kết quả chính** |
| `cascade-real` | 145 `M####.wav` → SPMamba | `Mix.csv` | tham khảo, có rò rỉ |

Chỉ số: macro-F1, balanced accuracy, ma trận nhầm lẫn, recall từng lớp; head nhị
phân thêm AUROC + độ nhạy/độ đặc hiệu. Chế độ `cascade-real` in kèm cảnh báo rò rỉ
ngay trong bảng kết quả để không bị trích dẫn nhầm.

## Đánh đổi đã cân nhắc

- **Huấn luyện trên âm thuần thay vì trên đầu ra SPMamba.** Train thẳng trên đầu ra
  tách sẽ khớp miền hơn, nhưng buộc phải cố định một checkpoint tách âm và làm hỏng
  tính độc lập của hai giai đoạn. Bắt đầu bằng âm thuần, và nếu khoảng cách Clean →
  Cascade quá lớn thì bổ sung tăng cường bằng chính đầu ra tách âm.
- **Không đưa `Mix/H####`, `L####` vào train ở bước đầu.** Chúng trùng lặp nặng và
  không truy ngược được về bản ghi gốc bằng metadata, nên rất dễ gây rò rỉ âm thầm.
  Có cờ `--include-mix-takes` để thử sau, kèm khử trùng lặp bằng mã băm.
- **CNN thay vì Mamba.** Với 50 bản ghi, mô hình có quy nạp mạnh và ít tham số
  thắng thế; giữ CNN làm chuẩn đối chiếu trước khi thử backbone Mamba.

## Tương thích ngược

Chỉ thêm file mới, trừ hai chỗ nối tối thiểu: thêm dòng export trong
`look2hear/models/__init__.py` và `look2hear/datas/__init__.py`. Đường dẫn tách âm
hiện có (`audio_train.py`, `test_hls.py`, các config) không thay đổi.
