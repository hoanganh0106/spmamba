# Plan HLS-CMDS cho SPMamba

Muc tieu: train SPMamba tach 2 nguon tieng tim va tieng phoi tren dataset HLS-CMDS, chay duoc tren Colab GPU T4 va theo doi bang W&B.

## Quyet dinh chinh

- Dataset HLS-CMDS Mix/Mix co cac bo ba `M####.wav`, `H####.wav`, `L####.wav`.
- WAV thuc te la mono 16-bit PCM, sample rate 4000 Hz, do dai 15 giay, tuong duong 60000 mau.
- Khong dua WAV vao git hoac Colab; build `data_hls.h5` tren laptop, nen zip, upload Drive, roi notebook giai nen/copy H5 ve SSD local.
- Tai su dung `H5DataModule` hien co bang cach convert dataset sang HDF5 voi `n_src=2`.
- Khong sua `audio_train.py`, `look2hear/datas/*`, hoac `datas/__init__.py`.

## Artifact

- `build_hls_h5.py`: convert triplet WAV thanh `data_hls.h5`.
- `configs/spmamba-hls-t4.yml`: config HLS 2 nguon cho Colab T4.
- `train_hls_colab.ipynb`: notebook Colab cai env, clone repo, giai nen/copy H5 da build san, train va liet ke checkpoint.

## H5 schema

```text
/mixtures/{M####}/mix
/mixtures/{M####}/s1   # heart
/mixtures/{M####}/s2   # lung
/mixture_ids
attrs: sample_rate=4000, n_src=2, n_samples=<count>, target_length=60000
```

## Luu y

- Loss dang dung `PITLossWrapper`, nen thu tu output tim/phoi co the bi hoan doi.
- Dataset chi co 145 mau, rui ro lon nhat la overfit.
- Giai doan sau co the them dynamic mixing tu HS/LS rieng, random gain/crop, hoac loss khong hoan vi neu can output channel co dinh.
