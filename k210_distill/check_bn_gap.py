"""Kiểm tra nguyên nhân khoảng cách train (+1.35 dB) vs val (-0.94 dB):
BatchNorm train/eval mismatch hay overfit thật?

So sánh cùng 1 checkpoint trên tập train & val (crop 4s, batch 32):
  - eval mode  (BN dùng running stats)
  - train mode (BN dùng batch stats, không cập nhật trọng số)
"""
import argparse

import torch
from torch.utils.data import DataLoader

from export_k210 import load_student_checkpoint
from look2hear.datas.h5_datamodule import H5DataModule
from look2hear.losses import PITLossWrapper, PairwiseNegSDR
from look2hear.models.k210_unet import Resampler16kTo8k

pit = PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=True), pit_from="pw_mtx")


def run(model, loader, rs, dev, mode, n_batches):
    model.train() if mode == "train" else model.eval()
    vals = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n_batches:
                break
            mix, tgt = rs(b[0].float().to(dev)), rs(b[1].float().to(dev))
            vals.append(-pit(model(mix), tgt).item())
    return sum(vals) / len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--conf_dir", default="configs/distill-k210-a100.yml")
    p.add_argument("--h5_path", default="data/data_30h.h5")
    p.add_argument("--n_batches", type=int, default=8)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    rs = Resampler16kTo8k().to(dev)
    dm = H5DataModule(h5_path=a.h5_path, n_src=3, sample_rate=16000, segment=4.0,
                      normalize_audio=False, batch_size=32, num_workers=4,
                      val_ratio=0.1, test_ratio=0.05, seed=42)
    dm.setup()
    tr, va, _ = dm.make_sets
    for name, ds in [("train", tr), ("val", va)]:
        for mode in ["eval", "train"]:
            model = load_student_checkpoint(a.ckpt, conf_dir=a.conf_dir).to(dev)
            model.export_k210 = False
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            s = run(model, loader, rs, dev, mode, a.n_batches)
            print(f"  data={name:<5} BN-mode={mode:<5}  SI-SNR = {s:+.2f} dB", flush=True)


if __name__ == "__main__":
    main()
