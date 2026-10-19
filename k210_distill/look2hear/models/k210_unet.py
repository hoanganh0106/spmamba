# -*- coding: utf-8 -*-
"""Kiến trúc Student K210SepUNet và bộ lọc hạ mẫu chống răng cưa Resampler16kTo8k.

Thiết kế chuyên biệt cho vi điều khiển RISC-V Kendryte K210 (8 MB SRAM = 2 MB KPU + 6 MB CPU):
1. ``Resampler16kTo8k``: Bộ lọc thông thấp FIR 63-tap cửa sổ Blackman (tần số cắt 3.6 kHz < Nyquist 4 kHz)
   triệt tiêu hoàn toàn hiện tượng gập phổ (aliasing) khi hạ mẫu trực tiếp trên GPU từ 16 kHz xuống 8 kHz.
2. ``K210SepUNet``:
   - Nhánh KPU (``enc1..dec1, head``): Chỉ sử dụng các toán tử được phần cứng KPU K210 hỗ trợ trực tiếp:
     ``Conv2d`` (kernel ``1x1`` hoặc ``3x3``), Depthwise ``Conv2d`` ``3x3``, ``BatchNorm2d``, ``ReLU``,
     ``MaxPool2d(2, 2)``, ``Upsample(scale_factor=2, mode='nearest')``, ``Sigmoid``, ``Concat``.
   - Nhánh RISC-V CPU FPU (khi bật ``use_riscv_attention=True``): Bổ sung khối
     **Dual-Path BiGRU + Multi-Head Self-Attention** dọc theo cả trục Tần số (gom họa âm toàn dải)
     và trục Thời gian (khóa giọng 3 người nói xuyên suốt câu, chống nhầm lẫn hoán vị) kết hợp
     đầu ra **Complex Phase-Sensitive Mask** trên FPU 32-bit của nhân CPU RISC-V 400 MHz.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class Resampler16kTo8k(nn.Module):
    """Bộ lọc hạ mẫu 16 kHz -> 8 kHz có chống răng cưa (Anti-aliasing FIR Low-pass Filter).

    Sử dụng bộ lọc thông thấp sinc nhân cửa sổ Blackman bậc 63 (cutoff = 3600 Hz < 4000 Hz Nyquist)
    và lấy mẫu xuống hệ số 2 (``stride=2``) bằng ``F.conv1d`` trực tiếp trên GPU.
    """

    def __init__(self, num_taps: int = 63, cutoff_hz: float = 3600.0, sr_in: int = 16000):
        super().__init__()
        if num_taps % 2 == 0:
            num_taps += 1
        self.num_taps = num_taps
        self.pad = num_taps // 2

        fc = cutoff_hz / float(sr_in)
        n = torch.arange(num_taps, dtype=torch.float64)
        mid = (num_taps - 1) / 2.0
        x = n - mid

        h = torch.where(
            x == 0,
            torch.tensor(2.0 * fc, dtype=torch.float64),
            torch.sin(2.0 * math.pi * fc * x) / (math.pi * x),
        )
        window = (
            0.42
            - 0.5 * torch.cos(2.0 * math.pi * n / (num_taps - 1))
            + 0.08 * torch.cos(4.0 * math.pi * n / (num_taps - 1))
        )
        kernel = (h * window)
        kernel = (kernel / kernel.sum()).to(dtype=torch.float32)
        self.register_buffer("kernel", kernel.view(1, 1, num_taps), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        orig_shape = wav.shape
        orig_dtype = wav.dtype
        t_in = orig_shape[-1]

        with torch.autocast(device_type=wav.device.type, enabled=False):
            flat = wav.float().reshape(-1, 1, t_in)
            padded = F.pad(flat, (self.pad, self.pad), mode="reflect")
            out = F.conv1d(padded, self.kernel.to(flat.device), stride=2)

        return out.reshape(*orig_shape[:-1], out.shape[-1]).to(dtype=orig_dtype)


class ConvBNReLU(nn.Module):
    """Tích chập 2D chuẩn (1x1 hoặc 3x3) + BatchNorm2d + ReLU (tương thích cứng KPU K210)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError(f"KPU K210 chi ho tro kernel_size 1 hoac 3, nhan {kernel_size}")
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConvBlock(nn.Module):
    """Depthwise Separable Conv2D (DW 3x3 + PW 1x1) được KPU K210 hỗ trợ phần cứng."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class K210DualPathGridBlock(nn.Module):
    """Khối Dual-Path (Frequency BiGRU + Attention & Temporal BiGRU + Attention) chạy trên CPU RISC-V FPU.

    - Quét dọc trục Tần số (Intra-Frame): Liên kết âm cơ bản (f0) và toàn bộ họa âm của cùng 1 người nói.
    - Quét dọc trục Thời gian (Inter-Frame): Theo dõi và khóa giọng từng người nói xuyên suốt câu,
      chống hiện tượng nhảy kênh / nhầm lẫn giữa 3 người nói.
    """

    def __init__(self, channels: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        # 1. Nhánh Tần số (Intra-chunk across F)
        self.f_norm1 = nn.LayerNorm(channels)
        self.f_gru = nn.GRU(channels, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.f_proj = nn.Linear(hidden_dim * 2, channels)
        self.f_norm2 = nn.LayerNorm(channels)
        self.f_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.f_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # 2. Nhánh Thời gian (Inter-chunk across T)
        self.t_norm1 = nn.LayerNorm(channels)
        self.t_gru = nn.GRU(channels, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.t_proj = nn.Linear(hidden_dim * 2, channels)
        self.t_norm2 = nn.LayerNorm(channels)
        self.t_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.t_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        b, c, f, t = x.shape

        # --- Bước 1: Quét toàn dải Tần số (F) ---
        xf = x.permute(0, 3, 2, 1).contiguous().view(b * t, f, c)
        xf_gru, _ = self.f_gru(self.f_norm1(xf))
        xf = xf + self.f_proj(xf_gru)
        xf_n = self.f_norm2(xf)
        xf_att, _ = self.f_attn(xf_n, xf_n, xf_n, need_weights=False)
        xf = (xf + xf_att).view(b, t, f, c).permute(0, 3, 2, 1).contiguous()
        x = x + torch.tanh(self.f_gate) * xf

        # --- Bước 2: Quét toàn chuỗi Thời gian (T) ---
        xt = x.permute(0, 2, 3, 1).contiguous().view(b * f, t, c)
        xt_gru, _ = self.t_gru(self.t_norm1(xt))
        xt = xt + self.t_proj(xt_gru)
        xt_n = self.t_norm2(xt)
        xt_att, _ = self.t_attn(xt_n, xt_n, xt_n, need_weights=False)
        xt = (xt + xt_att).view(b, f, t, c).permute(0, 3, 1, 2).contiguous()
        x = x + torch.tanh(self.t_gate) * xt

        return x


class K210SubbandFullResRefiner(nn.Module):
    """Bộ tinh chỉnh độ phân giải cao (16 ms/khung, 128 bin tần số) trước đầu ra mặt nạ."""

    def __init__(self, channels: int = 32, subbands: int = 16, hidden_dim: int = 96, num_heads: int = 4):
        super().__init__()
        self.subbands = subbands
        self.down_f = nn.Conv2d(channels, channels, kernel_size=(4, 1), stride=(4, 1), padding=0, bias=False)
        self.dp = K210DualPathGridBlock(channels, hidden_dim=hidden_dim, num_heads=num_heads)
        self.up_f = nn.ConvTranspose2d(channels, channels, kernel_size=(4, 1), stride=(4, 1), padding=0, bias=False)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.refine_conv[-1].weight)
        nn.init.zeros_(self.refine_conv[-1].bias)

    def forward(self, d1: torch.Tensor) -> torch.Tensor:
        # d1: [B, C, 128, T]
        sub = self.down_f(d1)           # [B, C, 32, T]
        sub = self.dp(sub)              # [B, C, 32, T] (quét đầy đủ mọi khung 16ms T và 32 dải tần)
        up = self.up_f(sub)             # [B, C, 128, T]
        return d1 + self.refine_conv(torch.cat([d1, up], dim=1))


class K210SepUNet(BaseModel):
    """Mô hình tách N người nói trên K210 (hỗ trợ cả thuần KPU và Hybrid KPU + RISC-V Attention)."""

    def __init__(
        self,
        input_dim: int = 129,
        n_srcs: int = 3,
        n_fft: int = 256,
        stride: int = 128,
        window: str = "hann",
        channels: Tuple[int, int, int] = (32, 96, 192),
        bottleneck_ch: int = 272,
        eps: float = 1.0e-5,
        sample_rate: int = 8000,
        export_k210: bool = False,
        use_riscv_attention: bool = False,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate)
        if len(channels) != 3:
            raise ValueError(f"K210SepUNet yeu cau dung 3 tang encoder channels, nhan {channels}")
        if n_fft % 2 != 0:
            raise ValueError(f"n_fft phai la so chan, nhan {n_fft}")

        self.input_dim = input_dim
        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.stride = stride
        self.window_name = window
        self.channels = tuple(int(c) for c in channels)
        self.bottleneck_ch = int(bottleneck_ch)
        self.eps = eps
        self.export_k210 = export_k210
        self.use_riscv_attention = bool(use_riscv_attention)

        self.n_freqs = n_fft // 2 + 1
        self.n_freqs_net = (self.n_freqs // 8) * 8

        win_fn = getattr(torch, f"{window}_window")
        self.register_buffer("window", win_fn(n_fft), persistent=False)

        c1, c2, c3 = self.channels
        b_ch = self.bottleneck_ch

        # --- Encoder (3 tầng trên KPU) ---
        self.enc1 = nn.Sequential(
            ConvBNReLU(1, c1, kernel_size=3),
            ConvBNReLU(c1, c1, kernel_size=3),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            ConvBNReLU(c1, c2, kernel_size=3),
            DSConvBlock(c2, c2),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            ConvBNReLU(c2, c3, kernel_size=3),
            DSConvBlock(c3, c3),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck (trên KPU) ---
        self.bottleneck = nn.Sequential(
            DSConvBlock(c3, b_ch),
            DSConvBlock(b_ch, b_ch),
            DSConvBlock(b_ch, c3),
        )

        # --- Decoder (3 tầng Upsample nearest + Skip connection trên KPU) ---
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = nn.Sequential(
            ConvBNReLU(c3 + c3, c2, kernel_size=1),
            DSConvBlock(c2, c2),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(
            ConvBNReLU(c2 + c2, c1, kernel_size=1),
            DSConvBlock(c1, c1),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(
            ConvBNReLU(c1 + c1, c1, kernel_size=1),
            ConvBNReLU(c1, c1, kernel_size=3),
        )

        # --- Mask Prediction Head (trên KPU) ---
        self.head = nn.Sequential(
            nn.Conv2d(c1, n_srcs, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # --- Nhánh RISC-V CPU FPU: Dual-Path BiGRU + Self-Attention + Complex Phase Head ---
        if self.use_riscv_attention:
            self.phase_stem = nn.Sequential(
                nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.PReLU(c1),
                nn.Conv2d(c1, c1, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.phase_stem[-1].weight)
            nn.init.zeros_(self.phase_stem[-1].bias)

            self.dp_bottleneck = nn.Sequential(
                K210DualPathGridBlock(c3, hidden_dim=128, num_heads=4),
                K210DualPathGridBlock(c3, hidden_dim=128, num_heads=4),
            )
            self.dp_dec3 = K210DualPathGridBlock(c2, hidden_dim=96, num_heads=4)
            self.dp_dec2 = K210DualPathGridBlock(c1, hidden_dim=64, num_heads=4)
            self.dp_dec1 = K210SubbandFullResRefiner(c1, subbands=16, hidden_dim=96, num_heads=4)

            # Đầu ra tinh chỉnh mặt nạ phức (Delta Real + Delta Imag) trên FPU RISC-V
            self.complex_head = nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.PReLU(c1),
                nn.Conv2d(c1, n_srcs * 2, kernel_size=1, padding=0, bias=True),
            )
            nn.init.zeros_(self.complex_head[-1].weight)
            nn.init.zeros_(self.complex_head[-1].bias)

    @property
    def num_spk(self) -> int:
        return self.n_srcs

    def forward_mask_core(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """Lõi mạng chạy hoàn toàn trên KPU K210 (thuần toán tử Conv2d/BN/ReLU/Pool/Upsample)."""
        e1 = self.enc1(feat_2d)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)

    def forward_hybrid_core(
        self, feat_2d: torch.Tensor, phase_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Chạy kết hợp KPU Conv2D + RISC-V Dual-Path BiGRU/Attention + Complex Mask."""
        e1 = self.enc1(feat_2d)
        if self.use_riscv_attention and phase_feat is not None:
            e1 = e1 + self.phase_stem(phase_feat)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)
        if self.use_riscv_attention:
            b = self.dp_bottleneck(b)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        if self.use_riscv_attention:
            d3 = self.dp_dec3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        if self.use_riscv_attention:
            d2 = self.dp_dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        if self.use_riscv_attention:
            d1 = self.dp_dec1(d1)

        base_mask = self.head(d1)  # [B, n_srcs, 128, T]
        complex_corr = None
        if self.use_riscv_attention:
            raw_corr = 2.0 * torch.tanh(self.complex_head(d1))
            complex_corr = raw_corr.view(
                d1.shape[0], self.n_srcs, 2, d1.shape[2], d1.shape[3]
            )
        return base_mask, complex_corr

    def compute_stft_feature(self, wav_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tính STFT và đặc trưng log-magnitude chuẩn hoá cho lõi UNet."""
        with torch.autocast(device_type=wav_norm.device.type, enabled=False):
            wav_f32 = wav_norm.float()
            spec = torch.stft(
                wav_f32,
                n_fft=self.n_fft,
                hop_length=self.stride,
                win_length=self.n_fft,
                window=self.window.to(wav_f32.device),
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )  # [B, F=129, T]
            mag = torch.abs(spec)
            mag_crop = mag[:, : self.n_freqs_net, :]
            log_mag = torch.log1p(mag_crop)
            mean = log_mag.mean(dim=(1, 2), keepdim=True)
            std = log_mag.std(dim=(1, 2), keepdim=True)
            feat = ((log_mag - mean) / (std + self.eps)).unsqueeze(1)  # [B, 1, 128, T]

        return spec, mag, feat.to(dtype=wav_norm.dtype)

    def forward_with_details(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chạy toàn bộ quá trình tách âm và trả kèm phổ ước lượng để tính loss."""
        if input.ndim == 1:
            input = input.unsqueeze(0)
        elif input.ndim == 3:
            if input.shape[1] == 1:
                input = input.squeeze(1)
            elif input.shape[2] == 1:
                input = input.squeeze(2)
            else:
                input = input[:, 0, :]

        n_samples = input.shape[-1]
        mix_std = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        wav_norm = input / mix_std

        spec, mag, feat = self.compute_stft_feature(wav_norm)
        bsz, _, _, n_frames = feat.shape

        pad_t = (8 - (n_frames % 8)) % 8
        if pad_t > 0:
            feat_padded = F.pad(feat, (0, pad_t))
        else:
            feat_padded = feat

        if self.use_riscv_attention:
            with torch.autocast(device_type=input.device.type, enabled=False):
                spec_crop = spec[:, : self.n_freqs_net, :]
                mag_crop = mag[:, : self.n_freqs_net, :].clamp(min=self.eps)
                cos_p = (spec_crop.real / mag_crop).unsqueeze(1).to(dtype=feat.dtype)
                sin_p = (spec_crop.imag / mag_crop).unsqueeze(1).to(dtype=feat.dtype)
                phase_feat = torch.cat([feat, cos_p, sin_p], dim=1)
                if pad_t > 0:
                    phase_feat = F.pad(phase_feat, (0, pad_t))
            masks_net, complex_corr = self.forward_hybrid_core(feat_padded, phase_feat)
            if pad_t > 0 and complex_corr is not None:
                complex_corr = complex_corr[..., :n_frames]
        else:
            masks_net = self.forward_mask_core(feat_padded)
            complex_corr = None

        if pad_t > 0:
            masks_net = masks_net[..., :n_frames]

        freq_diff = self.n_freqs - self.n_freqs_net
        if freq_diff > 0:
            masks_full = F.pad(masks_net, (0, 0, 0, freq_diff))
        else:
            masks_full = masks_net

        with torch.autocast(device_type=input.device.type, enabled=False):
            masks_f32 = masks_full.float()
            if complex_corr is not None:
                if freq_diff > 0:
                    complex_corr = F.pad(complex_corr.float(), (0, 0, 0, freq_diff))
                else:
                    complex_corr = complex_corr.float()
                m_real = masks_f32 + complex_corr[:, :, 0]
                m_imag = complex_corr[:, :, 1]
                x_r = spec.real.unsqueeze(1)
                x_i = spec.imag.unsqueeze(1)
                est_r = x_r * m_real - x_i * m_imag
                est_i = x_r * m_imag + x_i * m_real
                est_specs = torch.complex(est_r, est_i)
                est_mags = torch.abs(est_specs)
            else:
                est_mags = mag.unsqueeze(1) * masks_f32
                est_specs = spec.unsqueeze(1) * masks_f32

            est_specs_flat = est_specs.reshape(bsz * self.n_srcs, self.n_freqs, n_frames)
            est_wavs_flat = torch.istft(
                est_specs_flat,
                n_fft=self.n_fft,
                hop_length=self.stride,
                win_length=self.n_fft,
                window=self.window.to(est_specs_flat.device),
                center=True,
                normalized=False,
                onesided=True,
                length=n_samples,
            )
            est_wavs = est_wavs_flat.view(bsz, self.n_srcs, n_samples)
            est_wavs = est_wavs * mix_std.unsqueeze(1).float()

        return est_wavs.to(dtype=input.dtype), est_mags, masks_full

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.export_k210:
            return self.forward_mask_core(input)
        est_wavs, _, _ = self.forward_with_details(input)
        return est_wavs

    def get_model_args(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "n_srcs": self.n_srcs,
            "n_fft": self.n_fft,
            "stride": self.stride,
            "window": self.window_name,
            "channels": list(self.channels),
            "bottleneck_ch": self.bottleneck_ch,
            "eps": self.eps,
            "sample_rate": self.sample_rate,
            "use_riscv_attention": self.use_riscv_attention,
        }
