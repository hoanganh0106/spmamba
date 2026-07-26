# Nguồn dữ liệu công khai để pretrain / fine-tune

HLS-CMDS chỉ có **50 bản ghi độc lập mỗi nhánh**, lớp hiếm nhất có 2 bản ghi. Tài
liệu này liệt kê các bộ dữ liệu công khai dùng được để pretrain rồi fine-tune về
HLS-CMDS.

Mọi số liệu dưới đây đã được **kiểm chứng từ trang chính thức** (PhysioNet, GitHub/
GitLab, Mendeley Data, bài báo gốc) vào **26/07/2026**, và được một lượt fact-check
độc lập soát lại. Chỗ nào không kiểm chứng được thì ghi rõ.

---

## Kết luận trước, chi tiết sau

**Mục tiêu nhị phân Normal/Abnormal được dữ liệu công khai hỗ trợ rất tốt.** Có
hàng nghìn bản ghi cho cả hai nhánh.

**Phân loại 10 lớp bệnh tim thì không.** Dữ liệu công khai chỉ phủ được 4/10 lớp
tim. Không có nguồn công khai nào cho AF, AV Block, Tachycardia, S3, S4 và Late
Diastolic Murmur dưới dạng nhãn âm thanh tim.

**Phổi khá hơn: 5/6 lớp có nguồn.** Chỉ Pleural Rub là không có ở bất kỳ bộ nào —
lớp đó phải hoàn toàn dựa vào HLS-CMDS.

Đây chính là lý do nên gộp lớp thành các nhóm lớn hơn, hoặc đặt trọng tâm vào head
nhị phân, thay vì cố kết luận trên 10 lớp với 50 bản ghi.

### Độ phủ theo lớp

| Lớp tim | Nguồn công khai | Ghi chú |
|---|---|---|
| Normal | CirCor, CinC 2016 | dồi dào |
| Early Systolic Murmur | CirCor (`Early-systolic`) | trực tiếp |
| Mid Systolic Murmur | CirCor (`Mid-systolic`) | trực tiếp |
| Late Systolic Murmur | CirCor (`Late-systolic`) | trực tiếp |
| Late Diastolic Murmur | **không có** | CirCor chỉ có Early/Mid/Holo-diastolic |
| Atrial Fibrillation | **không có** | CirCor chỉ chú thích tiếng thổi, không chú thích nhịp |
| AV Block | **không có** | |
| Tachycardia | **không có** | |
| S3 | **không có** | |
| S4 | **không có** | |

| Lớp phổi | Nguồn công khai | Ghi chú |
|---|---|---|
| Normal | SPRSound, HF_Lung_V1, ICBHI, KAUH | dồi dào |
| Fine Crackles | SPRSound (912 sự kiện train) | **chỉ SPRSound tách được thô/mịn** |
| Coarse Crackles | SPRSound (49 sự kiện train) | hiếm ngay cả ở đây |
| Rhonchi | HF_Lung_V1 (4.740), SPRSound (39) | |
| Wheezing | HF_Lung_V1 (8.457), ICBHI (1.392 chu kỳ), SPRSound (452) | dồi dào |
| Pleural Rub | **không có** | chỉ HLS-CMDS mới có |

---

## Bảng tổng hợp

| Bộ dữ liệu | Nhánh | Quy mô tải được | Tần số | License | Rào cản |
|---|---|---|---|---|---|
| [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/) | tim | 3.163 bản ghi / 942 bệnh nhân | **4000 Hz** | ODC-By v1.0 | không |
| [PhysioNet/CinC 2016](https://physionet.org/content/challenge-2016/1.0.0/) | tim | 3.240 bản ghi | 2000 Hz | ODC-By v1.0 | không |
| [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) | phổi | 2.683 bản ghi / 292 bệnh nhi | 8000 Hz | CC BY 4.0 | không |
| [HF_Lung_V1](https://gitlab.com/techsupportHF/HF_Lung_V1) | phổi | 9.765 file × 15 s / 279 người | **4000 Hz** | CC BY 4.0 | không |
| [ICBHI 2017](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) | phổi | 920 bản ghi / 126 người | **trộn 4k/10k/44.1k** | **không có** | SSL trang gốc lỗi |
| [KAUH](https://data.mendeley.com/datasets/jwyy9np4gv/3) | phổi | **112 bản ghi thật** | **4000 Hz** | CC BY 4.0 | không |
| Yaseen 2018 | tim | 1.000 bản ghi | 8000 Hz | **không có** | **tránh dùng** |

---

## Nhánh tim

### CirCor DigiScope — nguồn chính

Bản ghi tại [PhysioNet v1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/),
DOI `10.13026/tshs-mw03`. License **Open Data Commons Attribution v1.0**, tải tự do,
không cần tài khoản hay thoả thuận sử dụng.

**Tải về được 3.163 bản ghi từ 942 bệnh nhân.** Con số 5.272 / 1.568 hay gặp trong
các bài báo là *toàn bộ* dataset, gồm cả 40% bị giữ kín làm tập chấm điểm của
Challenge 2022 — phần đó không tải được và sẽ không bao giờ được phát hành.

Tần số 4000 Hz, **đúng bằng HLS-CMDS**, nên không phải resample. Lưu ý kỹ thuật:
định dạng cho phép mỗi bệnh nhân có tần số riêng (dòng đầu file `.txt` của từng
bệnh nhân ghi tần số), và không trang chính thức nào cam kết toàn bộ đồng nhất — dù
mọi file kiểm tra đều là 4000 Hz. **Script tải nên đọc tần số từ header của từng
bản ghi thay vì gán cứng.**

Nhãn ở mức bệnh nhân trong `training_data.csv`, trong đó cột quan trọng nhất là
`Systolic murmur timing` với các giá trị `Early-systolic`, `Mid-systolic`,
`Late-systolic`, `Holosystolic`. Ba giá trị đầu ánh xạ thẳng sang ESM / MSM / LSM.

Cạm bẫy: nhãn tiếng thổi là **mức bệnh nhân**, còn tiếng thổi chỉ nghe được ở một
số vị trí nghe. Phải lọc theo cột `Murmur locations`, nếu không bạn sẽ gán nhãn
"có tiếng thổi" cho những bản ghi mà tiếng thổi không hề xuất hiện.

### PhysioNet/CinC Challenge 2016 — pretrain nhị phân

[Bản ghi v1.0.0](https://physionet.org/content/challenge-2016/1.0.0/), license
**ODC-By v1.0**, tải tự do. 3.240 bản ghi, **2000 Hz đồng nhất**, nhãn **chỉ nhị
phân** normal/abnormal (`REFERENCE.csv`, `-1` = normal, `1` = abnormal), trong đó
2.575 bản ghi normal.

Dùng để pretrain head nhị phân. Phải upsample 2 kHz → 4 kHz, hoặc pretrain ở 2 kHz
rồi đổi frontend — nêu rõ lựa chọn nào trong luận văn.

Có kèm `REFERENCE-SQI.csv` đánh giá chất lượng tín hiệu; lọc `SQI=1` cho khoảng
2.304 bản ghi normal sạch.

### Yaseen 2018 — không nên dùng

1.000 bản ghi (200 × 5 lớp), 8000 Hz. **Kho dữ liệu không có file LICENSE và
GitHub API trả về `license: null`** — tức không có giấy phép sử dụng nào. Với một
luận văn thì đây là rủi ro không đáng chấp nhận.

Ngoài ra 4 lớp bệnh của nó (hẹp van động mạch chủ, hẹp van hai lá, hở van hai lá,
sa van hai lá) là **chẩn đoán bệnh van tim**, không phải nhãn thì của tiếng thổi.
Ánh xạ chúng sang ESM/MSM/LSM là suy diễn theo sách giáo khoa, bản thân dataset
không khẳng định điều đó.

---

## Nhánh phổi

### SPRSound — nguồn chính, và là nguồn duy nhất tách được thô/mịn

[Kho GitHub SJTU](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound), license
**CC BY 4.0**, bài báo gốc IEEE TBioCAS 16(5):867-881 (2022), DOI
`10.1109/TBCAS.2022.3204910`.

Bản phát hành BioCAS2022 có 2.683 bản ghi từ 292 bệnh nhi (1 tháng – 18 tuổi),
**8000 Hz**, 16-bit mono. Mỗi file `.wav` đi kèm một file `.json` cùng tên. Cộng
cả các bộ test của những năm sau (2023/2024/2025) thì kho có 6.567 bản ghi và
nhiều file tóm tắt bệnh nhân riêng cho từng năm.

> ### ⚠ Bẫy nghiêm trọng nhất của kho này
>
> Kho **không chỉ** có các thư mục theo năm. Nó còn có `Classification/`,
> `Detection/` và `Compression/`, và đó là **các bản đóng gói lại của cùng một
> audio**. Cụ thể: `Classification/train_classification_*` chính là tập train
> BioCAS2022, còn `Classification/valid_classification_*` chính là test BioCAS2022
> cộng test BioCAS2023.
>
> Hệ quả: nếu bạn `glob` toàn kho lấy `*.wav`, bạn nhận khoảng **19.000 file trùng
> lặp chồng chéo**, và audio dùng để đánh giá sẽ **âm thầm lọt vào tập huấn
> luyện**. Chỉ dùng các thư mục theo năm (`BioCAS2022/`, …) hoặc chỉ dùng các thư
> mục theo tác vụ — **không trộn lẫn hai cách**.

Điểm giá trị nhất: **nhãn ở mức sự kiện kèm mốc thời gian mili-giây**, với 7 lớp —
Normal, Rhonchi, Wheeze, Stridor, **Coarse Crackle**, **Fine Crackle**,
Wheeze+Crackle. Đây là bộ công khai duy nhất phân biệt tiếng ran thô với tiếng ran
mịn, đúng như phân loại của HLS-CMDS.

Phân bố sự kiện trong tập train BioCAS2022: Normal 5.159, Fine Crackle 912,
Wheeze 452, Coarse Crackle 49, Rhonchi 39, Wheeze+Crackle 30, Stridor 15.

**Cạm bẫy quan trọng về chia tập.** SPRSound có sẵn hai tập test:

- `inter_test` (355 bản ghi, 41 bệnh nhân) — **không trùng bệnh nhân nào với train**. Dùng tập này.
- `intra_test` (379 bản ghi, 162 bệnh nhân) — **toàn bộ 162 bệnh nhân đều có trong train**. Số liệu trên tập này bị rò rỉ và lạc quan; nhiều bài báo vẫn báo cáo trên nó.

Cạm bẫy khác: 9% bản ghi train có nhãn `Poor Quality` — quyết định rõ ràng là bỏ
hay giữ làm lớp từ chối, đừng để chúng âm thầm thành `Normal`. Một số file JSON ở
các bộ test năm sau chỉ có `event_annotation` mà không có `record_annotation`, nên
loader phải chịu được khoá bị thiếu. Khi tự chia fold, gom nhóm theo mã bệnh nhân
(trường đầu tiên trong tên file) — mỗi trẻ đóng góp tới 4 vị trí nghe gần như đồng
thời.

Phải resample 8 kHz → 4 kHz để khớp HLS-CMDS. Việc này bỏ đi dải 2–4 kHz, nơi có
năng lượng thật của tiếng ran mịn — nêu rõ trong luận văn.

### HF_Lung_V1 — lớn nhất, định dạng trùng khít

[Kho GitLab](https://gitlab.com/techsupportHF/HF_Lung_V1), license **CC BY 4.0**,
bài báo Hsu et al., PLOS ONE 16(7):e0254134.

9.765 file, **mỗi file đúng 15 giây**, **4000 Hz** — tức trùng khít định dạng
HLS-CMDS, không phải resample, không phải cắt lại. 279 người (261 + 18). Tổng
khoảng 40,7 giờ.

Nhãn ở mức sự kiện có mốc thời gian: thở vào/thở ra, wheeze (8.457 sự kiện),
rhonchus (4.740), và tiếng ran không liên tục (15.606). **Không tách thô/mịn** —
đây là lý do vẫn cần SPRSound.

Cách tải: `git clone` kéo về lịch sử nhiều GB và dễ đầy đĩa Colab; nên tải thẳng
các phần `.7z` rồi ghép (`7z x train.7z.001`), và kiểm tra bằng `7z l` sau khi ghép
vì một phần bị cụt sẽ hỏng khá âm thầm.

Lưu ý: **HF_Lung_V2 không lấy được trọn vẹn** — phần bổ sung `HF_Lung_V1_IP` phát
hành audio *không kèm nhãn*, nhãn phải xin phép công ty Heroic Faith Medical Science.

### ICBHI 2017 — dùng được nhưng nhiều vướng

920 bản ghi, 126 người, 6.898 chu kỳ hô hấp có chú thích, 5,5 giờ. Bài báo gốc:
Rocha et al., *Physiological Measurement* 40 035001 (2019).

Bốn vấn đề thực tế:

1. **Không có giấy phép chính thức nào** được công bố trên trang gốc — không có
   trang license, không có tuyên bố bản quyền. Cần cân nhắc với luận văn.
2. **Chứng chỉ SSL của trang gốc đã hết hạn từ 26/11/2022**, nên mọi client HTTP
   mặc định đều lỗi. Phải dùng `curl -k`, `wget --no-check-certificate`, hoặc
   `requests(..., verify=False)`. Có bản mirror trên Harvard Dataverse **giống hệt
   từng byte** — tải dữ liệu từ đó được, nhưng **trích dẫn theo điều khoản của
   AUTH, đừng dùng nhãn CC0 mà Dataverse gắn**.
3. **Tần số lấy mẫu trộn lẫn: 4 kHz, 10 kHz và 44,1 kHz**, độ sâu bit trộn 16/24.
   File 24-bit cần `soundfile`; `scipy.io.wavfile` xử lý không tốt.
4. **File chia tập chính thức có một lỗi chưa được sửa.** File
   `ICBHI_challenge_train_test.txt` (chia 539 train / 381 test) liệt kê
   `226_1b1_Pl_sc_Meditron`, nhưng trong dataset file đó thực tế tên là
   `226_1b1_Pl_sc_LittC2SE`. Nối bảng chia tập với tên file một cách ngây thơ sẽ
   làm mất hoặc làm hỏng 1 trong 920 bản ghi. Lỗi này **không** nằm trong
   `filename_differences.txt` của dataset.

Nhãn chỉ là hai cờ nhị phân crackle/wheeze theo từng chu kỳ, nên phủ được nhiều
nhất 3/6 lớp phổi và **không tách được thô/mịn**.

### KAUH — nhỏ nhưng đúng tần số

[Mendeley Data v3](https://data.mendeley.com/datasets/jwyy9np4gv/3), license
**CC BY 4.0**, bài mô tả: Fraiwan et al., *Data in Brief* 35 (2021).

Cẩn thận với con số: bộ này có 336 file WAV, nhưng đó là **112 bệnh nhân × 3 kiểu
lọc ống nghe** (Bell / Diaphragm / Extended) của **cùng một bản ghi**. Chỉ có
**112 bản ghi thật sự khác nhau**. Gom nhóm theo bệnh nhân là bắt buộc, nếu không
ba biến thể của cùng một bản ghi sẽ rơi vào cả train lẫn test.

**4000 Hz**, đúng bằng HLS-CMDS. Bệnh nhân người lớn (12–90 tuổi). Nhãn nằm ngay
trong tên file. Không có Rhonchi, không có Pleural Rub, không tách thô/mịn. Bảng
chú giải mã nhãn trong file Excel không định nghĩa hết mọi ký hiệu — ý nghĩa của
mã `B` hiện chưa xác định được từ tài liệu chính thức, đừng đoán.

Giá trị chính của KAUH không phải để phân loại mà để **tổng hợp hỗn hợp cho bước
tách âm**: 4 kHz sẵn nên trộn với bản ghi tim mà không cần resample. Nhưng phải
trung thực trong luận văn — âm phổi thu trên thành ngực người thật **đã lẫn sẵn
tiếng tim chưa được gán nhãn**, nên hỗn hợp tổng hợp từ nó có năng lượng tim nằm
trong chính "nguồn phổi sạch", làm sai lệch hàm mất mát của bộ tách âm.

---

## Phân loại rút gọn `--taxonomy public`

Vì dữ liệu công khai chỉ phủ được một phần phân loại đầy đủ, `build_hls_cls_h5.py`
có sẵn tuỳ chọn rút gọn khớp đúng với phần phủ được:

```bash
python build_hls_cls_h5.py --taxonomy public --align-split-with data_hls.h5 \
    --out data_hls_cls.h5
```

| | `--taxonomy full` (mặc định) | `--taxonomy public` |
|---|---|---|
| Tim | 10 lớp | **5 lớp**: Normal, Early/Mid/Late Systolic Murmur, Other Abnormal |
| Phổi | 6 lớp | **6 lớp**: Normal, Coarse/Fine Crackles, Rhonchi, Wheezing, Other Abnormal |

Bốn lớp tim đầu tiên là đúng bốn lớp mà CirCor cung cấp được, nên checkpoint
pretrain trên CirCor ghép thẳng vào được. Với phổi, `Other Abnormal` chính là
Pleural Rub — lớp duy nhất không nguồn công khai nào có.

Hai điểm thiết kế đáng lưu ý:

- **Không bản ghi nào bị vứt bỏ.** 23 bản ghi tim thuộc các lớp không phủ được
  (LDM 6, S3 5, AF 4, Tachycardia 3, AV Block 3, S4 2) gộp vào `Other Abnormal`
  thay vì bị loại. Loại bỏ sẽ mất 46% dữ liệu nhánh tim.
- **Nhãn nhị phân không đổi theo phân loại.** Nó luôn suy từ nhãn đầy đủ, nên
  head "có bất thường hay không" vẫn được nuôi bằng cả 41/50 bản ghi bất thường
  của nhánh tim dù bạn chọn phân loại nào.

## Công thức đề xuất

### Nhánh phổi

1. Cắt các sự kiện SPRSound theo mốc thời gian trong JSON, resample 8 kHz → 4 kHz.
2. Pretrain 5 lớp: Normal / Fine Crackle / Coarse Crackle / Wheeze / Rhonchi. Bỏ
   hoặc gán ignore-index cho Stridor và Wheeze+Crackle.
3. Bổ sung HF_Lung_V1 cho Normal / Wheeze / Rhonchi (đã là 4 kHz, 15 s — không cần
   xử lý gì).
4. Báo cáo số liệu pretrain trên `inter_test` của SPRSound để bước pretrain cũng
   kiểm chứng được.
5. Fine-tune trên HLS-CMDS, **thêm logit Pleural Rub chỉ ở bước này** vì không
   nguồn công khai nào có lớp đó.

### Nhánh tim

1. Pretrain head nhị phân trên CinC 2016 (3.240 bản ghi, upsample 2 → 4 kHz).
2. Pretrain tiếp head đa lớp trên CirCor cho Normal / ESM / MSM / LSM, lọc theo
   `Murmur locations`.
3. Fine-tune trên HLS-CMDS. Sáu lớp còn lại (LDM, AF, AV Block, Tachycardia, S3,
   S4) **chỉ học được từ HLS-CMDS**, với 2–6 bản ghi mỗi lớp.

Bước 3 là chỗ nên thành thật với chính mình: sáu lớp đó sẽ không đạt độ tin cậy để
kết luận. Hoặc gộp chúng lại thành nhóm lớn hơn, hoặc báo cáo chúng như quan sát
định tính chứ không phải kết quả.

### Mở rộng bộ tách âm

`build_hls_h5.py` đang tổng hợp hỗn hợp bằng cách cộng một bản ghi tim với một bản
ghi phổi. Không có gì buộc hai nguồn phải đến từ manikin: lấy tim từ CirCor (4 kHz)
và phổi từ HF_Lung_V1 (4 kHz) cho ra hàng nghìn cặp thay vì tối đa 2.500 cặp từ
50 × 50 hiện nay, và không cần resample bên nào.

Đánh đổi đã nêu ở trên: bản ghi thật luôn lẫn nguồn kia, nên "nguồn sạch" chỉ là
gần đúng. Manikin sạch hơn hẳn về mặt này — đó chính là giá trị riêng của HLS-CMDS
và nên được nêu như vậy trong luận văn.

---

## Khoảng cách miền, phải nêu trong luận văn

| | HLS-CMDS | Dữ liệu công khai |
|---|---|---|
| Đối tượng | manikin lâm sàng | bệnh nhân thật |
| Tuổi | không áp dụng | CirCor/SPRSound: trẻ em; KAUH: người lớn 12–90 |
| Nhiễu nền | gần như không | tiếng thở, tiếng ruột, tiếng cọ ống nghe, tiếng phòng khám |
| Lẫn nguồn | không (thu riêng từng nguồn) | có (tim lẫn trong phổi và ngược lại) |

Hai hệ quả:

- Mô hình huấn luyện **và** kiểm thử hoàn toàn trên manikin không chứng minh được
  gì về bệnh nhân thật.
- Pretrain trên dữ liệu thật rồi fine-tune về manikin vẫn có ích (đặc trưng phổ ở
  tầng thấp chuyển giao được), nhưng chiều ngược lại — suy ra hiệu năng lâm sàng từ
  kết quả trên manikin — thì không hợp lệ.

Cách trình bày mạnh nhất: dùng dữ liệu công khai làm tập huấn luyện và đánh giá
chính, còn HLS-CMDS làm tập kiểm thử có nguồn sạch để đo riêng phần chất lượng mất
đi do bước tách âm — vốn là câu hỏi trung tâm của luận văn.
