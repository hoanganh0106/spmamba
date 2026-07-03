"""Đánh giá SPMamba trên test split source-disjoint của data_hls.h5.

Báo cả hai loại metric:
  - fixed : dùng output ch0 vs tim (s1), ch1 vs phổi (s2) theo đúng thứ tự
            (phù hợp model train bằng FixedOrderLossWrapper).
  - PIT   : lấy hoán vị tốt nhất -> "best-case" trần trên.
Nếu fixed ~ PIT và channel-accuracy cao => model đã gán kênh đúng
(ch0=tim, ch1=phổi). Nếu fixed << PIT => tách được nhưng đảo kênh.

Ví dụ:
    python test_hls.py --ckpt /content/drive/MyDrive/SPMamba_checkpoints/...-T4-fixed \
        --h5 /content/data_hls.h5 --conf_dir configs/spmamba-hls-t4.yml --save_k 3
"""
import argparse
import glob
import os
from itertools import permutations

import numpy as np
import torch
import soundfile as sf
import yaml
import fast_bss_eval

from test_evaluate import load_model, si_snr          # tái sử dụng loader + si_snr
from look2hear.datas import H5DataModule


def build_test_set(h5_path, conf_dir):
    cfg = yaml.safe_load(open(conf_dir))["datamodule"]["data_config"]
    cfg = dict(cfg)
    cfg["h5_path"] = h5_path
    cfg["num_workers"] = 0
    dm = H5DataModule(**cfg)
    dm.setup()                       # đọc /split -> test source-disjoint, segment=None
    return dm.data_test


def metrics_one(est, gt, mix):
    """est, gt: [n, T]; mix: [T]. Trả fixed & PIT SI-SNR(i) + SDR(i) (PIT)."""
    n = gt.shape[0]
    L = min(est.shape[-1], gt.shape[-1], mix.shape[-1])
    est, gt, mix = est[..., :L], gt[..., :L], mix[:L]
    mix_rep = mix.unsqueeze(0).expand(n, -1).contiguous()

    base = [si_snr(mix_rep[i], gt[i]).item() for i in range(n)]      # baseline mix vs ref

    # Fixed order (identity): ch i vs gt i
    fixed_vals = [si_snr(est[i], gt[i]).item() for i in range(n)]
    fixed_sisnr = float(np.mean(fixed_vals))
    fixed_sisnri = float(np.mean([fixed_vals[i] - base[i] for i in range(n)]))

    # PIT: best permutation
    best = (-1e9, None, None)
    for perm in permutations(range(n)):
        vals = [si_snr(est[perm[i]], gt[i]).item() for i in range(n)]
        if np.mean(vals) > best[0]:
            best = (float(np.mean(vals)), vals, perm)
    pit_sisnr, pit_vals, perm = best
    pit_sisnri = float(np.mean([pit_vals[i] - base[i] for i in range(n)]))

    # SDR (PIT) via fast_bss_eval, giống test_evaluate.py
    sdr = -fast_bss_eval.sdr_pit_loss(est, gt)
    sdr_base = -fast_bss_eval.sdr_pit_loss(mix_rep, gt)

    return dict(
        fixed_sisnr=fixed_sisnr,
        fixed_sisnri=fixed_sisnri,
        pit_sisnr=pit_sisnr,
        pit_sisnri=pit_sisnri,
        sdr=sdr.mean().item(),
        sdr_i=(sdr - sdr_base).mean().item(),
        perm=perm,
        channel_correct=(perm == tuple(range(n))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="đường dẫn .ckpt, hoặc thư mục để tự lấy *best*.ckpt mới nhất")
    ap.add_argument("--h5", default="/content/data_hls.h5")
    ap.add_argument("--conf_dir", default="configs/spmamba-hls-t4.yml")
    ap.add_argument("--outdir", default="/content/test_hls_out")
    ap.add_argument("--save_k", type=int, default=3, help="số mẫu lưu audio để nghe")
    args = ap.parse_args()

    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        cands = glob.glob(os.path.join(ckpt, "*best*.ckpt")) or glob.glob(os.path.join(ckpt, "*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"Không thấy .ckpt trong {ckpt}")
        ckpt = max(cands, key=os.path.getmtime)
        print("[auto] dùng checkpoint:", ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sr = load_model(ckpt, device)
    test_set = build_test_set(args.h5, args.conf_dir)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for i in range(len(test_set)):
        mixture, sources, name = test_set[i]          # [T], [2, T], str
        with torch.no_grad():
            est = model(mixture.unsqueeze(0).to(device))[0].cpu()   # [2, T]
        m = metrics_one(est, sources, mixture)
        rows.append(m)
        flag = "ok " if m["channel_correct"] else "SWAP"
        print(f"{name:>8}  fixedSI-SNRi={m['fixed_sisnri']:+.2f}  "
              f"pitSI-SNRi={m['pit_sisnri']:+.2f}  SDRi={m['sdr_i']:+.2f}  "
              f"perm={m['perm']} [{flag}]")

        if i < args.save_k:            # fixed order -> ch0=tim, ch1=phổi
            sf.write(f"{args.outdir}/{name}_mix.wav",          mixture.numpy(),   sr)
            sf.write(f"{args.outdir}/{name}_est_ch0_heart.wav", est[0].numpy(),   sr)
            sf.write(f"{args.outdir}/{name}_est_ch1_lung.wav",  est[1].numpy(),   sr)
            sf.write(f"{args.outdir}/{name}_ref_heart.wav",     sources[0].numpy(), sr)
            sf.write(f"{args.outdir}/{name}_ref_lung.wav",      sources[1].numpy(), sr)

    def agg(k):
        v = [r[k] for r in rows]
        return float(np.mean(v)), float(np.std(v))

    ch_acc = 100.0 * np.mean([r["channel_correct"] for r in rows])
    print("\n" + "=" * 54)
    print(f"  n_test = {len(rows)}")
    for k in ("fixed_sisnr", "fixed_sisnri", "pit_sisnr", "pit_sisnri", "sdr", "sdr_i"):
        mu, sd = agg(k)
        print(f"  {k:<13} mean={mu:+.2f} dB   std={sd:.2f}")
    print(f"  channel-accuracy (perm==(0,1)) = {ch_acc:.0f}%")
    gap = agg("pit_sisnri")[0] - agg("fixed_sisnri")[0]
    print(f"  PIT - fixed SI-SNRi gap = {gap:+.2f} dB "
          f"({'gán kênh TỐT' if gap < 1.0 else 'model hay ĐẢO kênh'})")
    print("=" * 54)


if __name__ == "__main__":
    main()
