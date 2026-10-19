"""Đánh giá chất lượng mô hình K210SepUNet (8 kHz): SI-SNR, SI-SNRi, SDR, SDRi & Real-Time Factor.

Chạy được trên cả cụm DGX A100 (sau khi train) VÀ trên Laptop Windows/macOS/Linux (chỉ cần CPU):
  python eval_k210.py \
      --ckpt Experiments/checkpoint/K210-SepUNet-Distill-A100/best_model.pth \
      --conf_dir configs/distill-k210-a100.yml \
      --h5_path data/data.h5
"""

import argparse
import csv
import os
import time
from typing import Dict, List

from export_k210 import load_student_checkpoint


def evaluate_k210_model(
    ckpt_path: str,
    conf_dir: str,
    h5_path: str,
    split: str = "test",
    max_samples: int = 0,
    save_csv: str = "",
    device_str: str = "auto",
) -> Dict[str, float]:
    import h5py
    import numpy as np
    import torch

    from look2hear.datas.h5_datamodule import H5DataModule
    from look2hear.losses import PITLossWrapper, PairwiseNegSDR
    from look2hear.models.k210_unet import Resampler16kTo8k

    try:
        import fast_bss_eval

        has_fast_bss = True
    except ImportError:
        has_fast_bss = False

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 72)
    print(f"[EVAL] Đang nạp K210SepUNet từ: {ckpt_path}")
    print(f"[EVAL] Thiết bị chạy kiểm thử  : {device}")
    print("=" * 72)

    model = load_student_checkpoint(ckpt_path, conf_dir=conf_dir).to(device)
    model.eval()
    model.export_k210 = False

    n_params = sum(p.numel() for p in model.parameters())
    resampler = Resampler16kTo8k().to(device)

    pit_sisdr = PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=True), pit_from="pw_mtx")
    pit_snr = PITLossWrapper(PairwiseNegSDR("snr", zero_mean=True), pit_from="pw_mtx")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu H5: {h5_path}")

    if not save_csv:
        exp_dir = os.path.dirname(os.path.abspath(ckpt_path))
        res_dir = os.path.join(exp_dir, "results")
        os.makedirs(res_dir, exist_ok=True)
        save_csv = os.path.join(res_dir, "metrics_k210.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)) or ".", exist_ok=True)

    dm = H5DataModule(
        h5_path=h5_path,
        n_src=3,
        sample_rate=16000,
        segment=None,
        normalize_audio=False,
        batch_size=1,
        num_workers=0,
        val_ratio=0.1,
        test_ratio=0.05,
        seed=42,
    )
    dm.setup()
    train_set, val_set, test_set = dm.make_sets
    eval_set = test_set if split == "test" else (val_set if split == "val" else train_set)

    all_sisnr: List[float] = []
    all_sisnri: List[float] = []
    all_sdr: List[float] = []
    all_sdri: List[float] = []
    total_infer_sec = 0.0
    total_audio_sec = 0.0

    num_total = len(eval_set)
    num_eval = min(num_total, max_samples) if max_samples > 0 else num_total
    print(f"[EVAL] Tập đánh giá: '{split}' — Số mẫu: {num_eval}/{num_total}")

    with open(save_csv, "w", newline="", encoding="utf-8") as csv_f:
        fieldnames = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", "infer_ms"]
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for idx in range(num_eval):
                sample_tuple = eval_set[idx]
                mix_16k = sample_tuple[0].float().unsqueeze(0).to(device)     # [1, T_16k]
                tgt_16k = sample_tuple[1].float().unsqueeze(0).to(device)     # [1, 3, T_16k]
                snt_key = sample_tuple[2] if len(sample_tuple) > 2 else idx

                # Hạ mẫu về 8 kHz bằng đúng bộ lọc chống răng cưa FIR 63-tap
                mix_8k = resampler(mix_16k)                                   # [1, T_8k]
                tgt_8k = resampler(tgt_16k)                                   # [1, 3, T_8k]

                t0 = time.perf_counter()
                est_8k = model(mix_8k)                                        # [1, 3, T_8k]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_sec = time.perf_counter() - t0

                audio_sec = mix_8k.shape[-1] / 8000.0
                total_infer_sec += infer_sec
                total_audio_sec += audio_sec

                # 1. Tính SI-SNR và SI-SNRi (dB)
                n_srcs = tgt_8k.shape[1]
                mix_rep = mix_8k.unsqueeze(1).expand(-1, n_srcs, -1)          # [1, 3, T_8k]

                sisnr_est = -float(pit_sisdr(est_8k, tgt_8k).item())
                sisnr_mix = -float(pit_sisdr(mix_rep, tgt_8k).item())
                sisnr_i = sisnr_est - sisnr_mix

                # 2. Tính SDR và SDRi (dB)
                if has_fast_bss:
                    sdr_est = -float(
                        fast_bss_eval.sdr_pit_loss(est_8k.squeeze(0), tgt_8k.squeeze(0)).mean().item()
                    )
                    sdr_mix = -float(
                        fast_bss_eval.sdr_pit_loss(mix_rep.squeeze(0), tgt_8k.squeeze(0)).mean().item()
                    )
                else:
                    sdr_est = -float(pit_snr(est_8k, tgt_8k).item())
                    sdr_mix = -float(pit_snr(mix_rep, tgt_8k).item())
                sdr_i = sdr_est - sdr_mix

                all_sisnr.append(sisnr_est)
                all_sisnri.append(sisnr_i)
                all_sdr.append(sdr_est)
                all_sdri.append(sdr_i)

                writer.writerow(
                    {
                        "snt_id": snt_key,
                        "sdr": round(sdr_est, 4),
                        "sdr_i": round(sdr_i, 4),
                        "si-snr": round(sisnr_est, 4),
                        "si-snr_i": round(sisnr_i, 4),
                        "infer_ms": round(infer_sec * 1000.0, 2),
                    }
                )

                if (idx + 1) % 25 == 0 or (idx + 1) == num_eval:
                    print(
                        f"  [{idx + 1:4d}/{num_eval:4d}] "
                        f"SI-SNRi: {np.mean(all_sisnri):+6.2f} dB | "
                        f"SDRi: {np.mean(all_sdri):+6.2f} dB | "
                        f"SI-SNR: {np.mean(all_sisnr):6.2f} dB | "
                        f"SDR: {np.mean(all_sdr):6.2f} dB"
                    )

        summary = {
            "sdr_avg": float(np.mean(all_sdr)),
            "sdr_std": float(np.std(all_sdr)),
            "sdri_avg": float(np.mean(all_sdri)),
            "sdri_std": float(np.std(all_sdri)),
            "sisnr_avg": float(np.mean(all_sisnr)),
            "sisnr_std": float(np.std(all_sisnr)),
            "sisnri_avg": float(np.mean(all_sisnri)),
            "sisnri_std": float(np.std(all_sisnri)),
            "rtf": total_infer_sec / max(total_audio_sec, 1e-6),
        }

        writer.writerow(
            {
                "snt_id": "avg",
                "sdr": round(summary["sdr_avg"], 4),
                "sdr_i": round(summary["sdri_avg"], 4),
                "si-snr": round(summary["sisnr_avg"], 4),
                "si-snr_i": round(summary["sisnri_avg"], 4),
                "infer_ms": round((total_infer_sec / max(num_eval, 1)) * 1000.0, 2),
            }
        )
        writer.writerow(
            {
                "snt_id": "std",
                "sdr": round(summary["sdr_std"], 4),
                "sdr_i": round(summary["sdri_std"], 4),
                "si-snr": round(summary["sisnr_std"], 4),
                "si-snr_i": round(summary["sisnri_std"], 4),
                "infer_ms": 0.0,
            }
        )

    print("\n" + "=" * 72)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG TÁCH 3 NGƯỜI NÓI (K210SepUNet 8 kHz)")
    print("=" * 72)
    print(f"  * Số tham số mô hình   : {n_params:,} params (~{n_params / 1024:.1f} KiB INT8)")
    print(f"  * SI-SNR trung bình    : {summary['sisnr_avg']:6.2f} ± {summary['sisnr_std']:.2f} dB")
    print(f"  * SI-SNRi (độ cải thiện): {summary['sisnri_avg']:+6.2f} ± {summary['sisnri_std']:.2f} dB")
    print(f"  * SDR trung bình       : {summary['sdr_avg']:6.2f} ± {summary['sdr_std']:.2f} dB")
    print(f"  * SDRi (độ cải thiện)  : {summary['sdri_avg']:+6.2f} ± {summary['sdri_std']:.2f} dB")
    print(f"  * Real-Time Factor (RTF): {summary['rtf']:.4f}x (nhỏ hơn 1.0 là chạy nhanh hơn thời gian thực)")
    print(f"  * Đã lưu bảng CSV tại  : {save_csv}")
    print("=" * 72)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Tính thông số SDR, SDRi, SI-SNR, SI-SNRi cho K210SepUNet"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="Experiments/checkpoint/K210-SepUNet-Distill-A100/best_model.pth",
        help="Đường dẫn tới best_model.pth hoặc epoch=*.ckpt",
    )
    parser.add_argument(
        "--conf_dir",
        type=str,
        default="configs/distill-k210-a100.yml",
        help="Đường dẫn tới file YAML cấu hình",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default="data/data.h5",
        help="Đường dẫn tới file data.h5",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Tập dữ liệu dùng để tính điểm (mặc định: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Giới hạn số mẫu cần chấm (0 = chấm toàn bộ tập test)",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Đường dẫn lưu file CSV kết quả (mặc định: <exp_dir>/results/metrics_k210.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Chạy trên 'cpu' (Laptop) hoặc 'cuda' (A100)",
    )
    args = parser.parse_args()
    evaluate_k210_model(
        ckpt_path=args.ckpt,
        conf_dir=args.conf_dir,
        h5_path=args.h5_path,
        split=args.split,
        max_samples=args.max_samples,
        save_csv=args.save_csv,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
