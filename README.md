# SPMamba (Modified Version)

Đây là kho lưu trữ SPMamba đã được sửa đổi một chút so với mã nguồn gốc của tác giả. 

**Mục đích chính của bản sửa đổi:**
Đảm bảo quá trình huấn luyện (training) diễn ra thành công. Những tinh chỉnh chủ yếu nằm ở phần xử lý dữ liệu (`look2hear/datas`), module nạp dữ liệu (datamodule) và cấu hình nhằm tương thích tốt hơn khi chạy huấn luyện thực tế với `audio_train.py`.

## Cách sử dụng

Để chạy huấn luyện, bạn có thể thực thi lệnh sau với file config tương ứng:

```bash
python audio_train.py --conf_dir configs/spmamba-h5-a100.yml
```

## Hai giai đoạn

Kho này gồm hai giai đoạn nối tiếp nhau:

1. **Tách âm** — SPMamba tách hỗn hợp thành kênh tim và kênh phổi (`audio_train.py`,
   `test_hls.py`, `configs/spmamba-*.yml`).
2. **Nhận dạng âm bất thường** — phân loại từng kênh đã tách. Xem hướng dẫn riêng ở
   [README_HLS_CLS.md](README_HLS_CLS.md), và [DATASETS.md](DATASETS.md) cho các
   nguồn dữ liệu công khai dùng để pretrain/fine-tune.
