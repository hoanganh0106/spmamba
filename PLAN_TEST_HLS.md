# Plan: file chạy test tách âm tim/phổi trên Colab

Mục tiêu: viết `test_hls.py` để **đánh giá** checkpoint SPMamba đã train trên **tập test source-disjoint** trong `data_hls.h5`, in SDR/SDRi/SI-SNR/SI-SNRi trung bình, và lưu vài ví dụ audio (mix, est, ref) để nghe. Chạy trên Colab bằng cùng môi trường micromamba của `train_hls_colab.ipynb`.

## Điều kiện đầu vào

- Checkpoint Lightning `.ckpt` trong `CKPT_DIR` (vd `/content/drive/MyDrive/SPMamba_checkpoints/SPMamba-HLS-HeartLung-T4/{epoch}-best.ckpt`). Mỗi ckpt chứa `state_dict` (tiền tố `audio_model.`) + `training_config`.
- `data_hls.h5` (bản dynamic mixing) tại `/content/data_hls.h5`, có `/split` source-disjoint → `test = 25 mẫu` (`mix = s1 + s2`, `s1`=tim, `s2`=phổi).
- Config `configs/spmamba-hls-t4.yml` (lấy `data_config`: `n_src=2, sample_rate=4000, segment=4.0, normalize_audio=false`).

## Nguyên tắc quan trọng

1. **Tái dùng code sẵn có**, đừng viết lại: `test_evaluate.load_model()` (đã lo cả config nested lẫn flattened) và `test_evaluate.si_snr()`.
2. **Lấy test set qua `H5DataModule`** — nó tự đọc `/split` nên đúng 25 mẫu test source-disjoint, **không** rò rỉ nguồn. Dùng `data_test` (đã set `segment=None` ⇒ full-length 15s), `batch_size=1`.
3. **PIT khi tính metric**: loss train là `PITLossWrapper` nên thứ tự kênh ra tim/phổi có thể hoán vị ⇒ phải thử cả 2 hoán vị `(0,1)`/`(1,0)` rồi lấy tốt nhất (giống `test_evaluate.evaluate`).
4. **Baseline** = mix so với từng nguồn; `SDRi = SDR(est) − SDR(mix)`, tương tự SI-SNRi. Đây mới là chỉ số cho biết model "tách" được bao nhiêu.
5. **normalize phải khớp train**: config `normalize_audio=false` ⇒ đưa mix thô vào model, không chuẩn hóa. Nếu sau này bật normalize thì phải nhân lại scale trước khi so với ref.

## Thiết kế `test_hls.py`

```
load_model(ckpt) ──▶ model, sr           # reuse test_evaluate.load_model
build test loader từ config ──▶ data_test # H5DataModule.setup(); dùng /split
for mixture, sources, name in data_test:
    est = model(mixture[None].to(device))[0]        # [2, T]
    metrics = eval_pit(est, sources, mixture)       # PIT 2 hoán vị
    (tùy chọn) lưu wav ví dụ cho K mẫu đầu
tổng hợp mean ± std → in bảng
```

Khung code đề xuất:

```python
"""Đánh giá SPMamba trên test split source-disjoint của data_hls.h5."""
import argparse, glob, os
import numpy as np, torch, soundfile as sf, yaml, fast_bss_eval
from itertools import permutations

from test_evaluate import load_model, si_snr          # tái sử dụng
from look2hear.datas import H5DataModule


def build_test_set(h5_path, conf_dir):
    cfg = yaml.safe_load(open(conf_dir))["datamodule"]["data_config"]
    cfg["h5_path"] = h5_path
    cfg["num_workers"] = 0
    dm = H5DataModule(**cfg)
    dm.setup()                       # đọc /split → 25 mẫu test, segment=None
    return dm.data_test


def eval_pit(est, gt, mix):
    """est, gt: [n, T]; mix: [T]. Trả SDR/SDRi/SI-SNR/SI-SNRi (PIT)."""
    n = gt.shape[0]
    L = min(est.shape[-1], gt.shape[-1], mix.shape[-1])
    est, gt, mix = est[..., :L], gt[..., :L], mix[:L]
    mix_rep = mix.unsqueeze(0).expand(n, -1)

    sdr      = -fast_bss_eval.sdr_pit_loss(est, gt)          # [n], đã PIT
    sdr_base = -fast_bss_eval.sdr_pit_loss(mix_rep, gt)

    best = (-1e9, None, None)
    for perm in permutations(range(n)):
        vals = [si_snr(est[perm[i]], gt[i]).item() for i in range(n)]
        if np.mean(vals) > best[0]:
            best = (np.mean(vals), vals, perm)
    base = [si_snr(mix_rep[i], gt[i]).item() for i in range(n)]

    return dict(
        sdr=sdr.mean().item(),
        sdr_i=(sdr - sdr_base).mean().item(),
        si_snr=best[0],
        si_snr_i=float(np.mean([best[1][i] - base[i] for i in range(n)])),
        perm=best[2],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="đường dẫn .ckpt (hoặc thư mục để tự lấy *-best.ckpt mới nhất)")
    ap.add_argument("--h5", default="/content/data_hls.h5")
    ap.add_argument("--conf_dir", default="configs/spmamba-hls-t4.yml")
    ap.add_argument("--outdir", default="/content/test_hls_out")
    ap.add_argument("--save_k", type=int, default=3, help="số mẫu lưu audio để nghe")
    args = ap.parse_args()

    ckpt = args.ckpt
    if os.path.isdir(ckpt):                     # tự chọn best mới nhất
        cands = glob.glob(os.path.join(ckpt, "*best*.ckpt")) or glob.glob(os.path.join(ckpt, "*.ckpt"))
        ckpt = max(cands, key=os.path.getmtime)
        print("[auto] dùng checkpoint:", ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sr = load_model(ckpt, device)
    test_set = build_test_set(args.h5, args.conf_dir)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for i in range(len(test_set)):
        mixture, sources, name = test_set[i]     # [T], [2, T], str
        with torch.no_grad():
            est = model(mixture.unsqueeze(0).to(device))[0].cpu()   # [2, T]
        m = eval_pit(est, sources, mixture)
        rows.append(m)
        print(f"{name:>8}  SDRi={m['sdr_i']:+.2f}  SI-SNRi={m['si_snr_i']:+.2f}  perm={m['perm']}")

        if i < args.save_k:                      # lưu audio để nghe
            p = est[list(m["perm"])]             # sắp lại theo perm: p[0]~tim, p[1]~phổi
            sf.write(f"{args.outdir}/{name}_mix.wav",       mixture.numpy(), sr)
            sf.write(f"{args.outdir}/{name}_est_heart.wav", p[0].numpy(),    sr)
            sf.write(f"{args.outdir}/{name}_est_lung.wav",  p[1].numpy(),    sr)
            sf.write(f"{args.outdir}/{name}_ref_heart.wav", sources[0].numpy(), sr)
            sf.write(f"{args.outdir}/{name}_ref_lung.wav",  sources[1].numpy(), sr)

    def agg(k): return np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])
    print("\n" + "=" * 46)
    for k in ("sdr", "sdr_i", "si_snr", "si_snr_i"):
        mu, sd = agg(k)
        print(f"  {k:<9} mean={mu:+.2f} dB   std={sd:.2f}")
    print(f"  n_test = {len(rows)}")
    print("=" * 46)


if __name__ == "__main__":
    main()
```

> Lưu ý về `perm`: config `spmamba-hls-t4.yml` giờ dùng `FixedOrderLossWrapper` (loss **không PIT**), nên model học **kênh cố định**: output ch0=tim, ch1=phổi. Khi **suy luận thực tế không có ref** vẫn biết kênh nào là gì. `eval_pit` ở đây vẫn để PIT nhằm đo "best-case", nhưng nếu train hội tụ tốt thì `perm` sẽ luôn là `(0, 1)`; có thể so thêm metric thứ tự cố định (ép `perm=(0,1)`) để kiểm tra model có gán đúng kênh không.

## Chạy trên Colab

Thêm 1 cell vào `train_hls_colab.ipynb` (hoặc notebook test riêng), **sau** Cell 1–4 đã cấu hình đường dẫn + có `/content/data_hls.h5` và đã pull repo:

```python
# === Cell test: đánh giá trên test split ===
import subprocess, os
env = os.environ.copy(); env["PYTHONPATH"] = REPO_DIR
cmd = [MM, "run", "-p", ENV_DIR, "python", "-u", "test_hls.py",
       "--ckpt", CKPT_DIR,                       # tự lấy *-best.ckpt mới nhất
       "--h5", LOCAL_H5,
       "--conf_dir", f"{REPO_DIR}/configs/spmamba-hls-t4.yml",
       "--outdir", "/content/test_hls_out", "--save_k", "3"]
subprocess.run(cmd, cwd=REPO_DIR, env=env, check=False)
```

Cell nghe thử audio đã lưu:

```python
# === Cell nghe kết quả ===
from IPython.display import Audio, display
import glob, os
for name in sorted({os.path.basename(p).rsplit("_",1)[0] for p in glob.glob("/content/test_hls_out/*_mix.wav")})[:3]:
    base = f"/content/test_hls_out/{name}"
    print(f"### {name}")
    for tag in ["mix", "est_heart", "ref_heart", "est_lung", "ref_lung"]:
        print(tag); display(Audio(f"{base}_{tag}.wav"))
```

## Kết quả kỳ vọng & cách đọc

- In từng mẫu `SDRi / SI-SNRi / perm`, rồi bảng `mean ± std` trên 25 mẫu test.
- **SDRi / SI-SNRi > 0** = model tách tốt hơn mix gốc; càng cao càng tốt. ~0 hoặc âm = chưa học được tách.
- Tập test chỉ 25 mẫu (giới hạn của source-disjoint với 50 nguồn) ⇒ `std` sẽ khá lớn, coi là chỉ báo sơ bộ.

## Cạm bẫy cần tránh

- **Sai sample rate/độ dài**: model train ở 4000 Hz, full-length 60000 mẫu; `H5DataModule` test dùng `segment=None` nên tự đúng — đừng ép crop.
- **normalize không khớp train** (xem mục Nguyên tắc #5).
- **Quên PYTHONPATH=REPO_DIR** ⇒ import `look2hear`/`test_evaluate` lỗi.
- **Lẫn checkpoint**: `*-best.ckpt` là theo `val_loss` thấp nhất; `last.ckpt` chỉ để resume, không nhất thiết tốt nhất.
- **Đừng test bằng H5 chế độ `triplet`** (`mix ≠ s1+s2`) — metric sẽ vô nghĩa; chỉ test trên bản dynamic.
```
