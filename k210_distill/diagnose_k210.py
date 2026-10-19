"""Chẩn đoán chất lượng K210SepUNet: Student vs Teacher vs Oracle mask (trần thiết kế).

Chạy:
  python diagnose_k210.py --student_ckpt <ckpt> --teacher_ckpt output/epoch=91-best.ckpt \
      --h5_path data/data_30h.h5 --conf_dir configs/distill-k210-a100.yml --n 200
"""
import argparse

import numpy as np
import torch

from export_k210 import load_student_checkpoint
from look2hear.datas.h5_datamodule import H5DataModule
from look2hear.losses import PITLossWrapper, PairwiseNegSDR
from look2hear.models.k210_unet import Resampler16kTo8k
from look2hear.system.distill_litmodule import load_teacher_model

pit_sisdr = PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=True), pit_from="pw_mtx")
pit_sdr = PITLossWrapper(PairwiseNegSDR("snr", zero_mean=True), pit_from="pw_mtx")


def metrics(est, tgt):
    return -pit_sisdr(est, tgt).item(), -pit_sdr(est, tgt).item()


def oracle(mix, tgt, kind, n_fft=256, hop=128, n_bins=128):
    """Mask lý tưởng tính từ nguồn thật, giữ pha mixture (đúng như K210SepUNet)."""
    win = torch.hann_window(n_fft, device=mix.device)
    st = lambda x: torch.stft(x, n_fft, hop, n_fft, win, center=True, return_complex=True)
    M = st(mix)                                # [1, F, T]
    S = st(tgt.reshape(-1, tgt.shape[-1]))     # [3, F, T]
    if kind == "irm":                          # |S| / sum|S|
        m = S.abs() / (S.abs().sum(0, keepdim=True) + 1e-8)
    elif kind == "ibm":
        m = (S.abs() == S.abs().max(0, keepdim=True).values).float()
    elif kind == "psm":                        # phase-sensitive, clip [0,1]
        m = (S.abs() / (M.abs() + 1e-8) * torch.cos(S.angle() - M.angle())).clamp(0, 1)
    m[:, n_bins:, :] = 0                       # K210 chỉ dùng 128/129 bins
    est = torch.istft(M * m, n_fft, hop, n_fft, win, center=True, length=mix.shape[-1])
    return est.unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--conf_dir", default="configs/distill-k210-a100.yml")
    p.add_argument("--h5_path", default="data/data_30h.h5")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = load_student_checkpoint(a.student_ckpt, conf_dir=a.conf_dir).to(dev).eval()
    student.export_k210 = False
    teacher = load_teacher_model(a.teacher_ckpt).to(dev).eval()
    rs = Resampler16kTo8k().to(dev)

    dm = H5DataModule(h5_path=a.h5_path, n_src=3, sample_rate=16000, segment=None,
                      normalize_audio=False, batch_size=1, num_workers=0,
                      val_ratio=0.1, test_ratio=0.05, seed=42)
    dm.setup()
    test_set = dm.make_sets[2]
    n = min(a.n, len(test_set))

    keys = ["mixture@8k", "student@8k", "teacher@16k", "teacher->8k",
            "oracle_IRM@8k", "oracle_IBM@8k", "oracle_PSM@8k"]
    res = {k: [] for k in keys}
    with torch.no_grad():
        for i in range(n):
            mix16 = test_set[i][0].float().unsqueeze(0).to(dev)
            tgt16 = test_set[i][1].float().unsqueeze(0).to(dev)
            mix8, tgt8 = rs(mix16), rs(tgt16)

            res["mixture@8k"].append(metrics(mix8.unsqueeze(1).repeat(1, 3, 1), tgt8))
            res["student@8k"].append(metrics(student(mix8), tgt8))
            t16 = teacher(mix16)
            res["teacher@16k"].append(metrics(t16, tgt16))
            res["teacher->8k"].append(metrics(rs(t16), tgt8))
            for kind in ["irm", "ibm", "psm"]:
                res[f"oracle_{kind.upper()}@8k"].append(metrics(oracle(mix8[0], tgt8[0], kind), tgt8))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n}", flush=True)

    base = np.array(res["mixture@8k"])
    print("\n" + "=" * 78)
    print(f"KẾT QUẢ CHẨN ĐOÁN trên {n} mẫu test (PIT, dB)")
    print("=" * 78)
    print(f"{'Hệ thống':<16}{'SI-SNR':>10}{'SI-SNRi':>10}{'SDR':>10}{'SDRi':>10}")
    for k in keys:
        v = np.array(res[k])
        b = base if "16k" not in k else None
        si, sd = v[:, 0].mean(), v[:, 1].mean()
        if k == "teacher@16k":
            print(f"{k:<16}{si:>10.2f}{'(16k)':>10}{sd:>10.2f}{'':>10}")
        else:
            print(f"{k:<16}{si:>10.2f}{si - b[:, 0].mean():>10.2f}{sd:>10.2f}{sd - b[:, 1].mean():>10.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
