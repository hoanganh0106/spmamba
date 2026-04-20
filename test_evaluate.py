"""Evaluate SPMamba: separate mix → compute SDRi/SI-SNRi vs ground truth."""
import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
import fast_bss_eval
from itertools import permutations

import look2hear.models


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("training_config") or ckpt["hyper_parameters"]
    audionet_name = config["audionet"]["audionet_name"]
    audionet_config = config["audionet"]["audionet_config"]
    sample_rate = config["datamodule"]["data_config"]["sample_rate"]

    model = getattr(look2hear.models, audionet_name)(
        sample_rate=sample_rate, **audionet_config
    )
    state_dict = {
        k.replace("audio_model.", ""): v
        for k, v in ckpt["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"Loaded {audionet_name} from {ckpt_path}")
    return model, sample_rate


def read_mono(path):
    wav, sr = sf.read(path)
    return (wav[:, 0] if wav.ndim > 1 else wav), sr


def si_snr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    s_target = torch.sum(est * ref) * ref / (torch.sum(ref ** 2) + 1e-8)
    e_noise = est - s_target
    return 10 * torch.log10(torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8))


def evaluate(mix_path, source_paths, est_sources):
    mix_wav, _ = read_mono(mix_path)
    mix_t = torch.from_numpy(mix_wav).float()

    gt_list = [torch.from_numpy(read_mono(p)[0]).float() for p in source_paths]
    gt = torch.stack(gt_list)
    n_srcs = gt.shape[0]

    # Align lengths
    L = min(est_sources.shape[-1], gt.shape[-1], mix_t.shape[-1])
    est, gt, mix_t = est_sources[..., :L], gt[..., :L], mix_t[:L]
    mix_rep = mix_t.unsqueeze(0).expand(n_srcs, -1)

    # SDR (PIT)
    sdr = -fast_bss_eval.sdr_pit_loss(est, gt)
    sdr_base = -fast_bss_eval.sdr_pit_loss(mix_rep, gt)

    # SI-SNR (brute-force PIT)
    best_avg, best_vals, best_perm = -float('inf'), None, None
    for perm in permutations(range(n_srcs)):
        vals = [si_snr(est[perm[i]], gt[i]).item() for i in range(n_srcs)]
        avg = np.mean(vals)
        if avg > best_avg:
            best_avg, best_vals, best_perm = avg, vals, perm

    sisnr_base = [si_snr(mix_rep[i], gt[i]).item() for i in range(n_srcs)]

    return {
        "sdr": sdr.mean().item(),
        "sdr_i": (sdr - sdr_base).mean().item(),
        "si_snr": best_avg,
        "si_snr_i": np.mean([best_vals[i] - sisnr_base[i] for i in range(n_srcs)]),
        "per_src": [(sdr[i].item(), best_vals[i]) for i in range(n_srcs)],
        "perm": best_perm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--outdir", default="/content/test_output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    model, sr = load_model(args.ckpt, device)

    # Separate
    mix_wav, _ = read_mono(args.mix)
    with torch.no_grad():
        est = model(torch.from_numpy(mix_wav).float().unsqueeze(0).to(device))
    est = est.squeeze(0).cpu()  # [n_srcs, T]

    for i in range(est.shape[0]):
        p = os.path.join(args.outdir, f"estimated_s{i+1}.wav")
        sf.write(p, est[i].numpy(), sr)
        print(f"Saved: {p}")

    # Evaluate
    r = evaluate(args.mix, args.sources, est)
    print(f"\n{'='*50}")
    print(f"  SDR:     {r['sdr']:+.2f} dB")
    print(f"  SDRi:    {r['sdr_i']:+.2f} dB")
    print(f"  SI-SNR:  {r['si_snr']:+.2f} dB")
    print(f"  SI-SNRi: {r['si_snr_i']:+.2f} dB")
    for i, (s, si) in enumerate(r['per_src']):
        print(f"  Spk {i+1}: SDR={s:+.2f}, SI-SNR={si:+.2f}")
    print(f"{'='*50}")
