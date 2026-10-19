"""Bộ phân loại âm tim/phổi bất thường cho HLS-CMDS.

Chạy trên đầu ra tách âm của SPMamba (ch0 = tim, ch1 = phổi) hoặc trực tiếp trên
bản ghi thuần. Kiến trúc: log-mel + CNN nhẹ + hai head dùng chung thân mạng.

* head nhị phân  — Normal / Abnormal, đồng thời dùng làm điểm anomaly score
* head đa lớp    — loại bệnh lý (tim 10 lớp, phổi 6 lớp)

Frontend log-mel được cài trực tiếp bằng ``torch.stft`` + ma trận mel dựng bằng
numpy, cố ý **không** dùng ``torchaudio``/``librosa``: bớt một phụ thuộc runtime
trên Colab, và giữ frontend khả vi để sau này có thể huấn luyện chung với SPMamba.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Tham số frontend mặc định cho tín hiệu 4 kHz. n_fft=256 -> cửa sổ 64 ms, hop=64
# -> bước 16 ms. Dải 20-2000 Hz phủ đủ nội dung tim (20-600 Hz) và phổi
# (100-2000 Hz) ở tần số lấy mẫu này (Nyquist = 2000 Hz).
DEFAULT_FRONTEND = dict(
    sample_rate=4000,
    n_fft=256,
    hop_length=64,
    n_mels=64,
    f_min=20.0,
    f_max=2000.0,
)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float
) -> np.ndarray:
    """Ma trận lọc mel tam giác, shape ``[n_mels, n_fft // 2 + 1]``."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        if right <= left:
            continue
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)

    # Chuẩn hoá Slaney: mỗi bộ lọc có diện tích bằng nhau, tránh việc các dải mel
    # rộng ở tần số cao lấn át các dải hẹp ở tần số thấp - nơi chứa tiếng tim.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    fb *= enorm[:, None]
    return fb.astype(np.float32)


class LogMelFrontend(nn.Module):
    """Sóng âm ``[B, T]`` -> log-mel đã chuẩn hoá ``[B, 1, n_mels, n_frames]``."""

    def __init__(
        self,
        sample_rate: int = 4000,
        n_fft: int = 256,
        hop_length: int = 64,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float = 2000.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps

        fb = mel_filterbank(sample_rate, n_fft, n_mels, f_min, min(f_max, sample_rate / 2))
        self.register_buffer("mel_fb", torch.from_numpy(fb), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"Mong doi dau vao [B, T], nhan {tuple(wav.shape)}")

        # Toàn bộ frontend chạy float32, kể cả khi bên ngoài đang bật AMP. Nếu để
        # autocast tác động, phép nhân ma trận mel sẽ chạy ở float16 và
        # ``log(mel + 1e-6)`` mất độ chính xác nghiêm trọng: 1e-6 là số dưới chuẩn
        # trong float16 (số chuẩn nhỏ nhất ~6e-5), nên các dải mel năng lượng thấp
        # - chính là chỗ chứa tiếng thổi và tiếng ran nhẹ - bị bóp về cùng giá trị.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = torch.stft(
                wav.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            power = spec.real.pow(2) + spec.imag.pow(2)       # [B, F, N]
            mel = torch.matmul(self.mel_fb, power)            # [B, n_mels, N]
            log_mel = torch.log(mel + self.eps)

            # Chuẩn hoá theo từng mẫu: mô hình bất biến với mức âm lượng. Cần thiết
            # vì đầu ra SPMamba có biên độ khác hẳn bản ghi thuần lúc huấn luyện.
            mean = log_mel.mean(dim=(1, 2), keepdim=True)
            std = log_mel.std(dim=(1, 2), keepdim=True)
            log_mel = (log_mel - mean) / (std + 1e-5)

        return log_mel.unsqueeze(1)                           # [B, 1, n_mels, N]


class SpecAugment(nn.Module):
    """Che ngẫu nhiên dải tần / khoảng thời gian. Chỉ hoạt động khi training."""

    def __init__(
        self,
        n_freq_masks: int = 2,
        freq_mask_width: int = 8,
        n_time_masks: int = 2,
        time_mask_width: int = 16,
    ):
        super().__init__()
        self.n_freq_masks = n_freq_masks
        self.freq_mask_width = freq_mask_width
        self.n_time_masks = n_time_masks
        self.time_mask_width = time_mask_width

    def _mask_one_axis(
        self, x: torch.Tensor, axis: int, n_masks: int, max_width: int
    ) -> torch.Tensor:
        """Che ``n_masks`` dải trên trục ``axis`` (2 = tần số, 3 = thời gian).

        Mỗi mẫu trong lô nhận vị trí và độ rộng che RIÊNG. Nếu dùng chung một mặt
        nạ cho cả lô (cách viết đơn giản hơn) thì với lô 32 mẫu ta chỉ có 1 biến
        thể tăng cường thay vì 32 - mất gần hết tác dụng chống overfit, vốn là
        điều tối quan trọng khi chỉ có 50 bản ghi.
        """
        length = x.shape[axis]
        if length < 2:
            return x
        batch = x.shape[0]
        width_cap = min(max_width, length - 1)
        if width_cap < 1:
            return x

        idx = torch.arange(length, device=x.device)
        shape = [1] * x.dim()
        shape[0] = batch
        shape[axis] = length

        for _ in range(n_masks):
            width = torch.randint(0, width_cap + 1, (batch,), device=x.device)
            # start ngẫu nhiên trong [0, length - width]
            span = (length - width).clamp(min=1)
            start = (torch.rand(batch, device=x.device) * span).long()
            mask = (idx >= start[:, None]) & (idx < (start + width)[:, None])
            x = x.masked_fill(mask.view(shape), 0.0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = self._mask_one_axis(x, 2, self.n_freq_masks, self.freq_mask_width)
        x = self._mask_one_axis(x, 3, self.n_time_masks, self.time_mask_width)
        return x


class _ConvBlock(nn.Module):
    """2 lớp tích chập 3x3 + BN + ReLU, rồi MaxPool 2x2 và Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.body(x)))


class HLSClassifier(nn.Module):
    """Phân loại một nhánh (tim hoặc phổi) từ sóng âm thô.

    ``forward`` trả ``(logits_multi, logit_binary)``:

    * ``logits_multi``  — ``[B, n_classes]``
    * ``logit_binary``  — ``[B]``, logit của "bất thường"

    Hai head cùng nằm trên một thân mạng. Về mặt toán học head nhị phân có thể suy
    ra từ head đa lớp, nhưng vẫn tách riêng vì nó cho điểm số hiệu chỉnh tốt hơn
    để đặt ngưỡng phát hiện bất thường, và là đầu ra dùng lại được nếu sau này
    chuyển sang hướng phát hiện bất thường không giám sát.
    """

    def __init__(
        self,
        n_classes: int,
        branch: str = "heart",
        classes: Optional[List[str]] = None,
        channels: Tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
        head_dropout: float = 0.4,
        frontend: Optional[Dict] = None,
        specaugment: Optional[Dict] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.branch = branch
        self.classes = list(classes) if classes else [str(i) for i in range(n_classes)]
        if len(self.classes) != n_classes:
            raise ValueError(f"classes co {len(self.classes)} phan tu nhung n_classes={n_classes}")

        self.frontend_config = {**DEFAULT_FRONTEND, **(frontend or {})}
        self.specaug_config = dict(specaugment or {})
        self.arch_config = dict(
            channels=tuple(channels), dropout=dropout, head_dropout=head_dropout
        )

        self.frontend = LogMelFrontend(**self.frontend_config)
        self.specaug = SpecAugment(**self.specaug_config)

        blocks: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(_ConvBlock(in_ch, out_ch, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        feat_dim = in_ch

        # Trọng số chú ý theo thời gian: bất thường tim/phổi là sự kiện cục bộ
        # (tiếng thổi, tiếng ran) chứ không trải đều toàn cửa sổ, nên gộp trung
        # bình đơn thuần sẽ pha loãng chúng.
        self.attention = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 2, 1),
            nn.Tanh(),
            nn.Conv1d(feat_dim // 2, 1, 1),
        )

        pooled_dim = feat_dim * 2  # gộp chú ý + gộp trung bình, nối lại
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_multi = nn.Linear(pooled_dim, n_classes)
        self.head_binary = nn.Linear(pooled_dim, 1)

    @classmethod
    def from_branch(cls, branch: str, **kwargs) -> "HLSClassifier":
        """Dựng mô hình với đúng bảng lớp của nhánh."""
        from look2hear.datas.hls_labels import classes as branch_classes

        names = branch_classes(branch)
        return cls(n_classes=len(names), branch=branch, classes=names, **kwargs)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.frontend(wav)              # [B, 1, n_mels, N]
        x = self.specaug(x)
        x = self.blocks(x)                  # [B, C, n_mels', N']
        x = x.mean(dim=2)                   # gộp trung bình theo trục tần số -> [B, C, N']

        weights = torch.softmax(self.attention(x), dim=-1)   # [B, 1, N']
        attended = (x * weights).sum(dim=-1)                 # [B, C]
        averaged = x.mean(dim=-1)                            # [B, C]
        return torch.cat([attended, averaged], dim=1)        # [B, 2C]

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.head_dropout(self.embed(wav))
        return self.head_multi(feats), self.head_binary(feats).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(prob_multi [B, n_classes], prob_abnormal [B])``."""
        was_training = self.training
        self.eval()
        logits_multi, logit_bin = self.forward(wav)
        if was_training:
            self.train()
        return torch.softmax(logits_multi, dim=-1), torch.sigmoid(logit_bin)

    def config(self) -> Dict:
        """Đủ để dựng lại mô hình mà không cần file config."""
        return dict(
            n_classes=self.n_classes,
            branch=self.branch,
            classes=self.classes,
            frontend=self.frontend_config,
            specaugment=self.specaug_config,
            **self.arch_config,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_checkpoint(
    path: str, map_location: str = "cpu"
) -> Tuple[HLSClassifier, Dict]:
    """Nạp checkpoint do ``train_hls_cls.py`` lưu.

    Trả ``(model, metadata)``. Phần metadata chứa ``data_config`` (đặc biệt là
    ``segment``) để lúc đánh giá cắt cửa sổ đúng bằng lúc huấn luyện — dùng sai độ
    dài cửa sổ sẽ làm số khung của phổ lệch đi và kết quả tụt mà không rõ lý do.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "model_config" not in ckpt:
        raise KeyError(
            f"{path} khong co 'model_config'. Checkpoint nay khong phai do "
            "train_hls_cls.py sinh ra."
        )
    cfg = dict(ckpt["model_config"])
    cfg["channels"] = tuple(cfg.get("channels", (32, 64, 128)))
    model = HLSClassifier(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


def build_from_checkpoint(path: str, map_location: str = "cpu") -> HLSClassifier:
    """Nạp checkpoint và chỉ trả mô hình."""
    return load_checkpoint(path, map_location)[0]
