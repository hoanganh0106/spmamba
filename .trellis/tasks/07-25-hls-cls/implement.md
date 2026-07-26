# Implement — HLS heart/lung abnormal sound recognition

Thứ tự thực hiện đi từ dữ liệu ra ngoài: nhãn → file H5 → dataset → mô hình →
huấn luyện → đánh giá → notebook. Mỗi bước có lệnh kiểm chứng chạy được ngay.

Máy local hiện **thiếu numpy/torch/h5py** (chỉ có Python 3.11 rỗng), nên mọi lệnh
kiểm chứng có tính toán đều chạy trên Colab. Các bước không cần numpy có ghi rõ
"chạy được ở local".

## Bước 1 — Bảng nhãn `look2hear/datas/hls_labels.py` ✅

- [x] `HEART_ABBR`, `LUNG_ABBR`, `HEART_CLASSES`, `LUNG_CLASSES` (thứ tự cố định).
- [x] `parse_stem(stem)` → `(gender, abbr, location)`.
- [x] `label_for(stem, branch)` → tên lớp đầy đủ; báo lỗi nếu mã viết tắt lạ.
- [x] `verify_against_csv(dataset_root, branch)` — đếm theo tên file, đối chiếu số
      lượng mỗi lớp với CSV, ném `ValueError` kèm bảng chênh lệch khi lệch.

**Đã kiểm chứng thật**: cả hai nhánh khớp CSV 100%; `F_CC_LUA` → Coarse Crackles,
`F_FC_LUA` → Fine Crackles (bẫy C/G của LS.csv đã bị chặn).

Kiểm chứng (chạy được ở local, chỉ dùng thư viện chuẩn):

```bash
python -c "from look2hear.datas.hls_labels import verify_against_csv as v; \
  v('hls-cmds_.../HLS-CMDS','heart'); v('hls-cmds_.../HLS-CMDS','lung'); print('nhan OK')"
```

Kỳ vọng: tim `Normal 9, MSM 7, LDM 6, ESM 6, LSM 5, S3 5, AF 4, T 3, AVB 3, S4 2`;
phổi `Normal 12, PR 9, CC 9, R 8, W 7, FC 5`.

## Bước 2 — `build_hls_cls_h5.py` ✅

- [x] Đọc 50 WAV tim + 50 WAV phổi, kiểm tra 4 kHz / 60000 mẫu.
- [x] Gán nhãn qua `hls_labels`, gọi `verify_against_csv` trước khi ghi.
- [x] `--align-split-with data_hls.h5`: dựng bảng bản ghi → split từ attrs
      `heart`/`lung` + `/split`; **khẳng định mỗi bản ghi chỉ có một split**.
- [x] Không có cờ trên thì chia phân tầng theo nhóm với hạt giống cố định.
- [x] Luôn sinh cột `fold` (grouped stratified K-fold, K=5).
- [x] In bảng tóm tắt: số bản ghi mỗi split, phân bố lớp mỗi split, số rò rỉ = 0.

**Đã chạy thật, cả hai chế độ**: rò rỉ = 0, fold cân bằng 10/10/10/10/10, file 22.9 MiB.
Chế độ kế thừa cho train/val/test = 37/8/5 mỗi nhánh.

**Phát hiện quan trọng khi chạy**: với split kế thừa, **tập test nhánh tim có 0 bản
ghi Normal** → F1/AUROC nhị phân trên split đó vô nghĩa. Script đã in cảnh báo tự
động. Hệ quả: **cross-validation (cột `fold`) là giao thức báo cáo chính**, split cố
định chỉ dùng cho thí nghiệm cascade.

Kiểm chứng (Colab):

```bash
python build_hls_cls_h5.py --dataset-root "hls-cmds_.../HLS-CMDS" \
    --align-split-with /content/data_hls.h5 --out /content/data_hls_cls.h5
```

Cổng kiểm tra: dòng `[check] ban ghi o nhieu split: 0` và tổng 50 + 50 bản ghi.

## Bước 3 — `look2hear/datas/hls_cls_datamodule.py`

- [ ] `HLSClsDataset(h5, branch, split|fold, segment, mode)` — train cắt ngẫu
      nhiên, eval cửa sổ trượt tất định, trả `(wav, label_idx, is_abnormal, record_idx)`.
- [ ] Tăng cường: dịch vòng, đổi biên độ ±6 dB, nhiễu Gauss 20–40 dB, đảo dấu.
- [ ] `HLSClsDataModule` — dựng 3 dataloader + phơi ra `class_weights`, `pos_weight`.
- [ ] Thêm export vào `look2hear/datas/__init__.py`.

Kiểm chứng: in kích thước lô, dải giá trị, và **khẳng định tập bản ghi của train và
test rời nhau**.

## Bước 4 — `look2hear/models/hls_classifier.py`

- [ ] `LogMelFrontend` — ma trận mel bằng numpy + `torch.stft`, đăng ký `buffer`.
- [ ] `SpecAugment` — chỉ hoạt động khi `self.training`.
- [ ] `HLSClassifier` — 3 khối CNN, gộp trung bình theo tần số, gộp chú ý theo thời
      gian, 2 head. Có `from_branch("heart"|"lung")` để dựng nhanh.
- [ ] Đăng ký vào `look2hear/models/__init__.py`.

Kiểm chứng: chạy tiến một lô ngẫu nhiên `[4, 20000]`, in shape hai đầu ra và tổng
số tham số (kỳ vọng < 500K).

## Bước 5 — `train_hls_cls.py`

- [ ] CLI: `--branch heart|lung`, `--h5`, `--conf_dir`, `--out_dir`, `--fold`,
      `--wandb*` (đặt tên cờ giống `audio_train.py`).
- [ ] AdamW + cosine, mất mát có trọng số lớp, early stopping theo macro-F1 nhị
      phân **ở mức bản ghi** trên tập val.
- [ ] Mỗi epoch log: loss, macro-F1 (nhị phân + đa lớp) ở cả mức cửa sổ và mức bản ghi.
- [ ] Lưu `best.pt` + `last.pt` kèm `classes`, `branch`, siêu tham số frontend, để
      `test_hls_cls.py` dựng lại mô hình mà không cần config.

Kiểm chứng: chạy `--epochs 2` xem loss có giảm và checkpoint có sinh ra không.

## Bước 6 — `test_hls_cls.py`

- [ ] `clean` — WAV thuần split=test.
- [ ] `cascade-synth` — test split `data_hls.h5` → SPMamba → 2 bộ phân loại; nhãn
      từ attrs `heart`/`lung`.
- [ ] `cascade-real` — 145 `M####.wav` → SPMamba → phân loại; nhãn từ `Mix.csv`;
      **in cảnh báo rò rỉ ngay trong bảng**.
- [ ] Chỉ số dùng chung: macro-F1, balanced accuracy, recall từng lớp, ma trận nhầm
      lẫn; nhị phân thêm AUROC + độ nhạy/độ đặc hiệu.
- [ ] Bảng cuối so sánh Clean vs Cascade-synth để thấy phần mất do tách âm.

## Bước 7 — `configs/hls-cls-t4.yml` và `train_cls_colab.ipynb`

- [ ] Config: tham số frontend, kiến trúc, tăng cường, tối ưu hoá, phần `heart`/
      `lung` ghi đè riêng (số lớp, số epoch).
- [ ] Notebook theo đúng khuôn `train_hls_colab.ipynb`: cấu hình đường dẫn + GPU →
      môi trường micromamba → clone repo → dựng H5 phân loại → train tim → train
      phổi → chạy đánh giá. Dùng lại `ENV_DIR`, `MM`, mẫu W&B secret sẵn có.

## Bước 8 — `smoke_hls_cls.py` (bổ sung ngoài kế hoạch ban đầu)

Máy local không có numpy/torch và người dùng không muốn cài, nên **không kiểm chứng
được gì ngoài bảng nhãn và script dựng H5 ở local**. Để bù lại, thêm một script kiểm
tra nhanh chạy trên Colab trong vài chục giây trước khi train thật:

- [x] Chỉ số đối chiếu với giá trị tính tay (AUROC kể cả trường hợp hoà,
      sensitivity/specificity không bị hoán đổi, macro-F1 bỏ qua lớp vắng mặt).
- [x] Bảng nhãn: Normal ở chỉ số 0, `CC`/`FC` đúng, khớp CSV.
- [x] Mô hình: shape hai head, số tham số < 500K, bất biến với mức âm lượng,
      SpecAugment tắt khi eval, `config()` dựng lại được mô hình.
- [x] Dataset: cửa sổ phủ hết bản ghi, fold train/val rời nhau và phủ đủ 50 bản ghi.
- [x] Gộp theo bản ghi và trường hợp bản ghi thiếu cửa sổ.
- [x] Một vòng train ngắn + lưu/nạp checkpoint cho kết quả trùng khớp.

Lệnh: `python smoke_hls_cls.py --h5 /content/data_hls_cls.h5 --dataset_root <HLS-CMDS>`

## Cổng kiểm tra chất lượng

1. Nhãn khớp CSV cho cả hai nhánh. — **đã đạt (chạy thật ở local)**
2. Số bản ghi nằm ở nhiều split = 0; tập bản ghi train ∩ test = ∅. — **đã đạt (chạy thật)**
3. `smoke_hls_cls.py` báo FAIL = 0. — cần chạy trên Colab
4. Số tham số < 500K; chạy tiến ra đúng shape. — smoke test bao phủ
5. Train 2 epoch chạy trọn không lỗi trên T4. — cần chạy trên Colab
6. Cả ba chế độ đánh giá chạy được và in ma trận nhầm lẫn. — cần chạy trên Colab
7. Đạt tiêu chí trong `prd.md`: macro-F1 nhị phân > 0.70. **Đo bằng cross-validation**,
   không đo bằng split cố định (tập test nhánh tim không có bản ghi Normal).

## Điểm quay lui

Toàn bộ là file mới, trừ hai dòng export thêm vào `look2hear/datas/__init__.py` và
`look2hear/models/__init__.py`. Quay lui = xoá các file mới và hoàn tác hai dòng
đó; đường dẫn tách âm không bị ảnh hưởng.
