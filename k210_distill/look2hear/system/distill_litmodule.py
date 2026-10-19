"""LightningModule cho huấn luyện Chưng cất tri thức trực tuyến (Online Knowledge Distillation).

Hỗ trợ chưng cất từ mô hình Teacher ``SPMamba`` (16 kHz, 6.04M tham số) sang mô hình
Student siêu nhẹ ``K210SepUNet`` (8 kHz, ~235K tham số) trên cụm máy chủ DGX A100:

1. Teacher được giữ ngoài danh sách ``nn.Module`` con của LightningModule (lưu trong
   ``self._teacher_holder``) để:
   - Lightning không tự động bật ``.train()`` lên Teacher ở đầu mỗi epoch.
   - File checkpoint ``.ckpt`` của Student không bị phình lên bởi 6.04M tham số của Teacher.
2. Tự động hạ mẫu chống răng cưa 16 kHz -> 8 kHz trên GPU bằng ``Resampler16kTo8k``.
3. Căn chỉnh thứ tự hoán vị 3 người nói (PIT Permutation Alignment) trước khi tính
   tổ hợp 3 hàm mất mát:
   - ``hard_loss``: PIT negative SI-SNR giữa Student và nhãn gốc 8 kHz.
   - ``tea_wav_loss``: Negative SI-SNR giữa sóng âm Student và Teacher đã căn chỉnh.
   - ``tea_mag_loss``: L1 loss trên phổ ``log1p(|STFT|)`` chuẩn hoá giữa Student và Teacher.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..losses import PITLossWrapper, multisrc_neg_sisdr, pairwise_neg_sisdr
from ..models.k210_unet import Resampler16kTo8k
from .audio_litmodule import flatten_dict


def load_teacher_model(ckpt_path: str, map_location: str = "cpu") -> nn.Module:
    """Nạp checkpoint Teacher SPMamba (ví dụ ``epoch=91-best.ckpt``) và đóng băng toàn bộ trọng số."""
    import look2hear.models

    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=map_location)

    if "training_config" in ckpt and "audionet" in ckpt["training_config"]:
        t_conf = ckpt["training_config"]
        audionet_name = t_conf["audionet"]["audionet_name"]
        audionet_cfg = dict(t_conf["audionet"]["audionet_config"])
        sample_rate = t_conf.get("datamodule", {}).get("data_config", {}).get("sample_rate", 16000)
    else:
        audionet_name = "SPMamba"
        sample_rate = 16000
        audionet_cfg = {
            "input_dim": 257,
            "n_srcs": 3,
            "n_fft": 512,
            "stride": 128,
            "window": "hann",
            "n_imics": 1,
            "n_layers": 6,
            "lstm_hidden_units": 128,
            "attn_n_head": 4,
            "attn_approx_qk_dim": 512,
            "emb_dim": 16,
            "emb_ks": 4,
            "emb_hs": 1,
            "activation": "prelu",
            "eps": 1.0e-5,
            "use_builtin_complex": True,
        }

    teacher_cls = getattr(look2hear.models, audionet_name)
    teacher = teacher_cls(
        sample_rate=sample_rate,
        **audionet_cfg,
    )

    state_dict = ckpt.get("state_dict", ckpt)
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("audio_model."):
            clean_sd[k[len("audio_model.") :]] = v
        elif k.startswith("model."):
            clean_sd[k[len("model.") :]] = v
        else:
            clean_sd[k] = v

    missing, unexpected = teacher.load_state_dict(clean_sd, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint Teacher thieu tham so quan trong: {missing[:5]}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


class DistillLightningModule(pl.LightningModule):
    """Module huấn luyện Student ``K210SepUNet`` có chưng cất từ Teacher ``SPMamba``."""

    def __init__(
        self,
        audio_model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        video_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_func: Optional[Dict[str, Any]] = None,
        train_loader: Any = None,
        val_loader: Any = None,
        test_loader: Any = None,
        scheduler: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config

        # Lưu Teacher trong list thường để KHÔNG đăng ký vào state_dict của Student
        self._teacher_holder = [teacher_model] if teacher_model is not None else []

        # Cấu hình tần số lấy mẫu dữ liệu gốc (16 kHz) và Student (8 kHz)
        data_cfg = self.config.get("datamodule", {}).get("data_config", {})
        self.source_sr = int(data_cfg.get("sample_rate", 16000))
        student_sr = getattr(self.audio_model, "_sample_rate", 8000)
        self.student_sr = int(student_sr)

        if self.source_sr == 16000 and self.student_sr == 8000:
            self.resampler: Optional[nn.Module] = Resampler16kTo8k()
        elif self.source_sr == self.student_sr:
            self.resampler = None
        else:
            raise ValueError(
                f"Chi ho tro (source_sr=16000, student_sr=8000) hoac bang nhau, "
                f"nhan source_sr={self.source_sr}, student_sr={self.student_sr}"
            )

        # Trọng số hàm mất mát chưng cất
        distill_cfg = self.config.get("distill", {})
        self.alpha_hard = float(distill_cfg.get("alpha_hard", 0.5))
        self.beta_tea_wav = float(distill_cfg.get("beta_tea_wav", 0.5))
        self.gamma_tea_mag = float(distill_cfg.get("gamma_tea_mag", 10.0))

        # Dynamic mixing (chống overfit): trộn lại nguồn giữa các mẫu trong batch.
        # data_30h.h5: mix = s1 + s2 + s3 + noise (gain ~1.0), nên noise = mix - sum(s).
        dm_cfg = self.config.get("dynamic_mix", {}) or {}
        self.dm_enabled = bool(dm_cfg.get("enabled", False))
        self.dm_prob = float(dm_cfg.get("prob", 0.8))
        self.dm_gain_db = float(dm_cfg.get("gain_db", 3.0))

        # Bộ căn chỉnh thứ tự kênh (PIT) dùng chung cho Teacher và Student
        self.pit_aligner = PITLossWrapper(
            pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False
        )

        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        self.validation_step_outputs = []
        self.test_step_outputs = []

    @property
    def teacher_model(self) -> Optional[nn.Module]:
        return self._teacher_holder[0] if self._teacher_holder else None

    def _ensure_teacher_on_device(self, device: torch.device) -> Optional[nn.Module]:
        teacher = self.teacher_model
        if teacher is None:
            return None
        if next(teacher.parameters()).device != device:
            teacher.to(device)
        if teacher.training:
            teacher.eval()
        return teacher

    def _maybe_downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.resampler is None:
            return x
        return self.resampler(x)

    def _compute_norm_log_mag(self, wavs: torch.Tensor) -> torch.Tensor:
        """Tính phổ ``log1p(|STFT|)`` đã chuẩn hoá biên độ cho tensor ``[B, n_srcs, T_8k]``."""
        b, s, t = wavs.shape
        n_fft = getattr(self.audio_model, "n_fft", 256)
        hop = getattr(self.audio_model, "stride", 128)
        eps = getattr(self.audio_model, "eps", 1e-5)

        with torch.autocast(device_type=wavs.device.type, enabled=False):
            flat = wavs.float().reshape(b * s, t)
            std = flat.std(dim=-1, keepdim=True).clamp(min=eps)
            flat_norm = flat / std
            window = torch.hann_window(n_fft, device=flat.device)
            spec = torch.stft(
                flat_norm,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            log_mag = torch.log1p(torch.abs(spec))
        return log_mag.view(b, s, log_mag.shape[-2], log_mag.shape[-1])

    def _compute_complex_stft_loss(
        self, est_wavs: torch.Tensor, tgt_wavs: torch.Tensor
    ) -> torch.Tensor:
        """Tính hàm mất mát phổ phức (Log-Magnitude + Real/Imag) đã chuẩn hoá theo biên độ nguồn thật."""
        b, s, t = est_wavs.shape
        n_fft = getattr(self.audio_model, "n_fft", 256)
        hop = getattr(self.audio_model, "stride", 128)
        eps = getattr(self.audio_model, "eps", 1e-5)

        with torch.autocast(device_type=est_wavs.device.type, enabled=False):
            est_f = est_wavs.float().reshape(b * s, t)
            tgt_f = tgt_wavs.float().reshape(b * s, t)
            est_zm = est_f - est_f.mean(dim=-1, keepdim=True)
            tgt_zm = tgt_f - tgt_f.mean(dim=-1, keepdim=True)

            # Căn chỉnh biên độ tối ưu (projection) để L1 trên Real/Imag nhất quán với SI-SNR
            dot = torch.sum(est_zm * tgt_zm, dim=-1, keepdim=True)
            est_energy = torch.sum(est_zm ** 2, dim=-1, keepdim=True).clamp(min=eps)
            scale = (dot / est_energy).detach().clamp(min=0.05, max=20.0)
            est_scaled = est_zm * scale

            tgt_std = tgt_zm.std(dim=-1, keepdim=True).clamp(min=eps)
            est_norm = est_scaled / tgt_std
            tgt_norm = tgt_zm / tgt_std

            window = torch.hann_window(n_fft, device=est_f.device)
            spec_est = torch.stft(
                est_norm, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=True, pad_mode="reflect",
                normalized=False, onesided=True, return_complex=True,
            )
            spec_tgt = torch.stft(
                tgt_norm, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, center=True, pad_mode="reflect",
                normalized=False, onesided=True, return_complex=True,
            )
            mag_loss = F.l1_loss(torch.log1p(torch.abs(spec_est)), torch.log1p(torch.abs(spec_tgt)))
            ri_loss = F.l1_loss(spec_est.real, spec_tgt.real) + F.l1_loss(spec_est.imag, spec_tgt.imag)
            return mag_loss + 0.5 * ri_loss

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.audio_model(wav)

    @torch.no_grad()
    def _dynamic_mix(
        self, mixtures: torch.Tensor, sources: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trộn lại nguồn trong batch: mỗi kênh nguồn hoán vị độc lập + gain ngẫu nhiên,
        cộng noise (mix - sum(s)) của một mẫu ngẫu nhiên. Áp dụng với xác suất ``dm_prob``."""
        bsz, n_src, _ = sources.shape
        if bsz < 2 or torch.rand(()).item() > self.dm_prob:
            return mixtures, sources
        dev = sources.device
        noise = mixtures - sources.sum(dim=1)
        new_src = torch.stack(
            [sources[torch.randperm(bsz, device=dev), i] for i in range(n_src)], dim=1
        )
        gain_db = (torch.rand(bsz, n_src, 1, device=dev) * 2 - 1) * self.dm_gain_db
        new_src = new_src * (10.0 ** (gain_db / 20.0))
        new_mix = new_src.sum(dim=1) + noise[torch.randperm(bsz, device=dev)]
        return new_mix, new_src

    def training_step(self, batch, batch_nb):
        mixtures_raw, targets_raw = batch[0], batch[1]
        if self.dm_enabled:
            mixtures_raw, targets_raw = self._dynamic_mix(mixtures_raw, targets_raw)
        teacher = self._ensure_teacher_on_device(mixtures_raw.device)

        # 1. Nếu có Teacher SPMamba (16 kHz), suy luận Teacher trên tín hiệu gốc trước khi hạ mẫu
        teacher_est_8k = None
        if teacher is not None:
            with torch.no_grad():
                teacher_est_16k = teacher(mixtures_raw)
                teacher_est_8k = self._maybe_downsample(teacher_est_16k)

        # 2. Hạ mẫu hỗn hợp và nhãn thật xuống 8 kHz cho Student K210SepUNet
        mixtures_8k = self._maybe_downsample(mixtures_raw)
        targets_8k = self._maybe_downsample(targets_raw)

        # 3. Chạy Student trên tín hiệu 8 kHz
        est_sources_8k = self(mixtures_8k)

        # 4. Tính Hard PIT Loss và lấy est_sources đã căn chỉnh theo thứ tự targets_8k
        hard_loss, stu_aligned = self.loss_func["train"](
            est_sources_8k, targets_8k, return_ests=True
        )

        if teacher_est_8k is not None:
            with torch.no_grad():
                _, tea_aligned = self.pit_aligner(
                    teacher_est_8k, targets_8k, return_ests=True
                )
                tea_log_mag = self._compute_norm_log_mag(tea_aligned)

            tea_wav_loss = multisrc_neg_sisdr(stu_aligned.float(), tea_aligned.float()).mean()
            stu_log_mag = self._compute_norm_log_mag(stu_aligned)
            tea_mag_loss = F.l1_loss(stu_log_mag, tea_log_mag)

            loss = (
                self.alpha_hard * hard_loss
                + self.beta_tea_wav * tea_wav_loss
                + self.gamma_tea_mag * tea_mag_loss
            )

            self.log("train_hard_sisnr", -hard_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
            self.log("train_tea_sisnr", -tea_wav_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
            self.log("train_mag_l1", tea_mag_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        else:
            stft_loss = self._compute_complex_stft_loss(stu_aligned, targets_8k)
            loss = hard_loss + self.gamma_tea_mag * stft_loss
            self.log("train_hard_sisnr", -hard_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
            self.log("train_mag_l1", stft_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        if dataloader_idx == 0:
            mixtures_raw, targets_raw = batch[0], batch[1]
            mixtures_8k = self._maybe_downsample(mixtures_raw)
            targets_8k = self._maybe_downsample(targets_raw)

            est_sources_8k = self(mixtures_8k)
            loss = self.loss_func["val"](est_sources_8k, targets_8k)
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.validation_step_outputs.append(loss)
            return {"val_loss": loss}

        if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
            mixtures_raw, targets_raw = batch[0], batch[1]
            mixtures_8k = self._maybe_downsample(mixtures_raw)
            targets_8k = self._maybe_downsample(targets_raw)

            est_sources_8k = self(mixtures_8k)
            tloss = self.loss_func["val"](est_sources_8k, targets_8k)
            self.log(
                "test_loss",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.test_step_outputs.append(tloss)
            return {"test_loss": tloss}

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            val_loss = torch.mean(self.all_gather(avg_loss))
            if self.optimizer is not None:
                self.log(
                    "lr",
                    self.optimizer.param_groups[0]["lr"],
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            self.log("val_pit_sisnr", -val_loss, on_epoch=True, prog_bar=True)

        if (self.trainer.current_epoch) % 10 == 0 and self.test_step_outputs:
            avg_tloss = torch.stack(self.test_step_outputs).mean()
            test_loss = torch.mean(self.all_gather(avg_tloss))
            self.log("test_pit_sisnr", -test_loss, on_epoch=True, prog_bar=False)

        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
