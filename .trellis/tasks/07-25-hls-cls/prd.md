# HLS heart/lung abnormal sound recognition

Nối tiếp phần tách âm SPMamba: sau khi tách hỗn hợp thành kênh tim (ch0) và kênh
phổi (ch1), nhận dạng **âm thanh bất thường** trên từng kênh.

## Goal

Xây dựng pipeline phân loại âm tim/phổi chạy trên đầu ra của SPMamba, trả lời hai
câu hỏi cho mỗi kênh:

1. **Bất thường hay không?** (nhị phân Normal / Abnormal) — chỉ số chính, đồng
   thời dùng làm điểm anomaly score.
2. **Bất thường loại nào?** (đa lớp: tim 10 lớp, phổi 6 lớp) — thông tin lâm sàng
   tham khảo.

Kết quả cuối cùng phải đo được ở dạng **cascade end-to-end**: hỗn hợp thô →
SPMamba → 2 bộ phân loại → nhãn tim + nhãn phổi.

## Bối cảnh dữ liệu (đã kiểm chứng trên đĩa)

Dataset HLS-CMDS, toàn bộ 4000 Hz, mọi bản ghi đúng 15.0 s.

| Nguồn | Số file | Nội dung |
|---|---|---|
| `HS/HS/*.wav` | 50 | âm tim thuần, 10 lớp |
| `LS/LS/*.wav` | 50 | âm phổi thuần, 6 lớp |
| `Mix/Mix/{M,H,L}####.wav` | 145×3 | hỗn hợp thu thật + take tim/phổi độc lập |

Phân bố lớp (số **bản ghi**, không phải số đoạn):

- Tim: Normal 9, Mid Systolic Murmur 7, Late Diastolic Murmur 6, Early Systolic
  Murmur 6, Late Systolic Murmur 5, S3 5, Atrial Fibrillation 4, Tachycardia 3,
  AV Block 3, S4 2.
- Phổi: Normal 12, Pleural Rub 9, Coarse Crackles 9, Rhonchi 8, Wheezing 7,
  Fine Crackles 5.

Ràng buộc quan trọng phát sinh từ dữ liệu:

- **Chỉ có 50 bản ghi độc lập mỗi nhánh.** Lớp nhỏ nhất (S4) có 2 bản ghi. Mọi
  con số đa lớp 10 lớp sẽ có phương sai rất lớn; phải báo cáo kèm cross-validation
  chứ không phải một lần chia train/test.
- **Đơn vị độc lập là bản ghi, không phải đoạn audio.** Cắt 15 s thành nhiều cửa
  sổ rồi chia ngẫu nhiên theo cửa sổ sẽ rò rỉ dữ liệu và cho accuracy ảo cao.
- `Mix/Mix/H####.wav` và `L####.wav` **trùng lặp nặng** với 50 bản ghi gốc: 53/145
  file tim và 57/145 file phổi trùng byte-for-byte, và chỉ có 89/145 (tim) và
  74/145 (phổi) mã băm khác nhau. Không được coi chúng là dữ liệu độc lập.
- **LS.csv dùng mã viết tắt khác tên file**: CSV ghi `C` = Fine Crackles và
  `G` = Coarse Crackles, trong khi tên file dùng `FC` = Fine Crackles và
  `CC` = Coarse Crackles. Đối chiếu theo cặp (giới tính, vị trí) xác nhận
  `FC`↔`C` (5 file) và `CC`↔`G` (9 file). Ghép nhãn theo mã viết tắt của CSV sẽ
  **đảo ngược Coarse ↔ Fine**. Phần tim không có vấn đề này.
- `Mix.csv` chỉ có một cột `Location` cho cả bản ghi hỗn hợp, không cho biết vị
  trí của từng take tim/phổi thành phần, nên không thể truy ngược `H####` về bản
  ghi gốc chỉ bằng metadata.

## Requirements

### R1 — Nhãn và nhóm

- Sinh nhãn từ tên file theo bảng viết tắt cố định (`CC`→Coarse Crackles,
  `FC`→Fine Crackles, …), rồi **đối chiếu chéo với CSV để xác nhận số lượng mỗi
  lớp**; sai lệch phải làm chương trình dừng với thông báo rõ ràng.
- Mỗi đoạn audio mang một `group_id` = tên bản ghi gốc (ví dụ `F_MSM_LUSB`). Mọi
  phép chia dữ liệu phải nhóm theo `group_id`.

### R2 — Chống rò rỉ dữ liệu

- Chia train/val/test theo bản ghi, phân tầng theo lớp.
- Split của bộ phân loại phải **kế thừa đúng phân hoạch nguồn của `data_hls.h5`**
  (đọc từ thuộc tính `heart`/`lung` của từng hỗn hợp và mảng `/split`), để bản ghi
  dùng test bộ phân loại cũng là bản ghi mà SPMamba chưa từng thấy. Đây là điều
  kiện để thí nghiệm cascade có ý nghĩa.
- Có chế độ **grouped stratified K-fold** để báo cáo, vì một lần chia chỉ cho ~5
  bản ghi test.

### R3 — Mô hình

- Frontend log-mel + CNN nhẹ (mục tiêu < 500K tham số), hai đầu ra dùng chung
  thân: head nhị phân và head đa lớp.
- Một kiến trúc dùng chung cho cả tim và phổi, khác nhau ở số lớp và tham số
  frontend; huấn luyện thành hai checkpoint riêng.
- Suy luận ở mức cửa sổ, tổng hợp lên mức file bằng trung bình xác suất.

### R4 — Huấn luyện

- Xử lý mất cân bằng lớp bằng trọng số lớp trong hàm mất mát.
- Tăng cường dữ liệu: dịch vòng theo thời gian, đổi biên độ ngẫu nhiên, thêm
  nhiễu Gauss, SpecAugment (che dải tần / che khung thời gian).
- Chạy được trên Colab T4, ghi log qua W&B giống quy trình tách âm hiện có.

### R5 — Đánh giá

Báo cáo trên ba chế độ, theo thứ tự độ sạch giảm dần:

1. **Clean** — phân loại trực tiếp bản ghi tim/phổi thuần (trần trên của bộ phân loại).
2. **Cascade tổng hợp** — test split của `data_hls.h5` → SPMamba → phân loại.
   Nhãn lấy từ thuộc tính `heart`/`lung` của mỗi hỗn hợp. Đây là số liệu
   end-to-end sạch, không rò rỉ.
3. **Cascade thu thật** — 145 file `M####.wav` → SPMamba → phân loại, nhãn từ
   `Mix.csv`. Thực tế nhất nhưng **có rò rỉ** (xem ràng buộc ở trên); phải ghi rõ
   là số liệu tham khảo, không phải kết quả chính.

Chỉ số: macro-F1, balanced accuracy, recall từng lớp, ma trận nhầm lẫn; riêng head
nhị phân thêm AUROC, độ nhạy và độ đặc hiệu.

## Acceptance Criteria

- [ ] `build_hls_cls_h5.py` sinh được file H5 phân loại; tự kiểm tra nhãn khớp CSV
      và **dừng với lỗi rõ ràng** nếu Coarse/Fine bị hoán đổi.
- [ ] Split kế thừa từ `data_hls.h5` được xác nhận: không bản ghi nào xuất hiện ở
      hai split khác nhau (in ra số bản ghi rò rỉ = 0).
- [ ] `train_hls_cls.py` train xong bộ phân loại tim và phổi trên Colab T4, log W&B.
- [ ] `test_hls_cls.py` in đủ cả ba chế độ đánh giá kèm ma trận nhầm lẫn.
- [ ] Head nhị phân trên chế độ Clean đạt macro-F1 > 0.70 ở mức file (mốc tối
      thiểu để coi là có tín hiệu; dưới mức này phải xem lại frontend/tăng cường).
- [ ] Báo cáo nêu rõ khoảng cách giữa Clean và Cascade tổng hợp — đây chính là
      phần chất lượng mất đi do bước tách âm.

## Ngoài phạm vi (ghi nhận để làm sau)

- Phát hiện bất thường không giám sát (chỉ train trên Normal). Head nhị phân đã
  sinh sẵn điểm số dùng được làm anomaly score khi cần chuyển hướng.
- Fine-tune đồng thời SPMamba và bộ phân loại (huấn luyện chung end-to-end).
- Dùng thêm bộ dữ liệu ngoài (ICBHI, PhysioNet) để tiền huấn luyện.
