"""Huấn luyện bộ phân loại âm tim/phổi bất thường (HLS-CMDS).

Chạy riêng cho từng nhánh, sinh ra hai checkpoint độc lập::

    python train_hls_cls.py --branch heart --h5 /content/data_hls_cls.h5 \
        --conf_dir configs/hls-cls-t4.yml --out_dir /content/drive/.../hls_cls
    python train_hls_cls.py --branch lung  --h5 /content/data_hls_cls.h5 ...

Cross-validation (khuyến nghị để BÁO CÁO, vì một lần chia chỉ cho ~5 bản ghi test)::

    python train_hls_cls.py --branch heart --fold 0   # ... tới --fold 4

Chọn mô hình tốt nhất theo trung bình macro-F1 của hai head, đo **ở mức bản ghi**
trên tập val, không phải mức cửa sổ: mục tiêu cuối cùng là kết luận trên cả bản
ghi, và mức cửa sổ luôn cho con số lạc quan hơn.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from hls_cls_metrics import (
    balanced_accuracy,
    binary_report,
    format_binary_report,
    format_multiclass_report,
    macro_f1,
)
from look2hear.datas import HLSClsDataModule, aggregate_by_record
from look2hear.models import HLSClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Gộp đệ quy; ``override`` thắng ở lá.

    Phải đệ quy chứ không gộp nông: config có khối lồng 3 tầng
    (``model.specaugment.time_mask_width``), gộp nông sẽ thay nguyên khối
    ``specaugment`` bằng đúng một khoá và âm thầm đưa các khoá còn lại về mặc
    định của mô hình.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_branch_config(config: Dict, branch: str) -> Dict:
    """Gộp phần cấu hình chung với phần ghi đè riêng của nhánh."""
    base = {k: v for k, v in config.items() if k not in ("heart", "lung")}
    return deep_merge(base, config.get(branch) or {})


# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: HLSClassifier,
    loader: DataLoader,
    device: str,
    n_classes: int,
) -> Dict:
    """Suy luận trên toàn bộ cửa sổ rồi gộp lên mức bản ghi."""
    model.eval()
    probs_multi: List[np.ndarray] = []
    probs_bin: List[np.ndarray] = []
    record_idx: List[np.ndarray] = []
    win_label: List[np.ndarray] = []
    win_abnormal: List[np.ndarray] = []

    for wav, label, abnormal, ridx in loader:
        wav = wav.to(device, non_blocking=True)
        logits_multi, logit_bin = model(wav)
        probs_multi.append(torch.softmax(logits_multi.float(), dim=-1).cpu().numpy())
        probs_bin.append(torch.sigmoid(logit_bin.float()).cpu().numpy())
        record_idx.append(ridx.numpy())
        win_label.append(label.numpy())
        win_abnormal.append(abnormal.numpy())

    probs_multi = np.concatenate(probs_multi)
    probs_bin = np.concatenate(probs_bin)
    record_idx = np.concatenate(record_idx)
    win_label = np.concatenate(win_label)
    win_abnormal = np.concatenate(win_abnormal)

    dataset = loader.dataset
    n_records = dataset.n_records

    # Nhãn ở mức bản ghi lấy thẳng từ dataset, không suy từ cửa sổ.
    rec_label = dataset.label_idx.astype(int)
    rec_abnormal = dataset.is_abnormal.astype(int)

    rec_probs_multi = aggregate_by_record(probs_multi, record_idx, n_records)
    rec_probs_bin = aggregate_by_record(probs_bin, record_idx, n_records)
    rec_pred_multi = rec_probs_multi.argmax(axis=1)

    return dict(
        # mức cửa sổ (để theo dõi quá trình học)
        window_multi_f1=macro_f1(win_label, probs_multi.argmax(axis=1), n_classes),
        window_bin_f1=macro_f1(win_abnormal.astype(int), (probs_bin >= 0.5).astype(int), 2),
        # mức bản ghi (dùng để chọn mô hình và báo cáo)
        multi_f1=macro_f1(rec_label, rec_pred_multi, n_classes),
        multi_balanced_acc=balanced_accuracy(rec_label, rec_pred_multi, n_classes),
        binary=binary_report(rec_abnormal, rec_probs_bin),
        # Nhị phân suy ra từ head đa lớp: đối chứng cho head nhị phân riêng.
        # Quyết định cứng lấy từ argmax để nhất quán với ma trận nhầm lẫn đa lớp;
        # 1 - P(Normal) chỉ dùng làm điểm xếp hạng cho AUROC.
        binary_from_multi=binary_report(
            rec_abnormal,
            1.0 - rec_probs_multi[:, 0],
            y_pred=(rec_pred_multi != 0).astype(int),
        ),
        rec_label=rec_label,
        rec_abnormal=rec_abnormal,
        rec_pred_multi=rec_pred_multi,
        rec_probs_bin=rec_probs_bin,
        record_ids=dataset.record_ids,
    )


def selection_score(metrics: Dict) -> float:
    """Tiêu chí chọn checkpoint: trung bình macro-F1 hai head ở mức bản ghi.

    Không dùng riêng F1 nhị phân vì tập val có thể chỉ chứa một lớp (xảy ra thật
    với split kế thừa từ ``data_hls.h5``), khi đó con số nhị phân mất ý nghĩa và
    việc chọn mô hình sẽ thành ngẫu nhiên.
    """
    return 0.5 * (metrics["multi_f1"] + metrics["binary"]["macro_f1"])


# --------------------------------------------------------------------------- #
# Huấn luyện
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> Dict:
    config = merge_branch_config(load_config(args.conf_dir), args.branch)
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    optim_cfg = dict(config.get("optim", {}))

    epochs = int(args.epochs or optim_cfg.get("epochs", 60))
    lr = float(optim_cfg.get("lr", 3e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
    patience = int(optim_cfg.get("early_stop_patience", 20))
    lambda_bin = float(optim_cfg.get("lambda_binary", 0.5))
    label_smoothing = float(optim_cfg.get("label_smoothing", 0.05))

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = HLSClsDataModule(
        h5_path=args.h5,
        branch=args.branch,
        fold=args.fold,
        seed=args.seed,
        num_workers=args.num_workers,
        **{k: v for k, v in data_cfg.items() if k != "num_workers"},
    )
    dm.setup()
    classes = dm.classes
    n_classes = len(classes)
    n_folds = dm.n_folds

    model = HLSClassifier(
        n_classes=n_classes, branch=args.branch, classes=classes, **model_cfg
    ).to(device)
    print(f"[model] {args.branch}: {model.num_parameters():,} tham so, {n_classes} lop")

    class_weights = dm.class_weights().to(device)
    pos_weight = dm.pos_weight().to(device)
    print(f"[loss] trong so lop = {np.round(class_weights.cpu().numpy(), 3).tolist()}")
    print(f"[loss] pos_weight (nhi phan) = {float(pos_weight):.3f}")

    criterion_multi = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    criterion_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project or "spmamba",
            name=args.wandb_run_name or f"HLS-cls-{args.branch}"
            + (f"-fold{args.fold}" if args.fold is not None else ""),
            config=dict(
                branch=args.branch,
                fold=args.fold,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                lambda_binary=lambda_bin,
                n_classes=n_classes,
                n_parameters=model.num_parameters(),
                **{f"data_{k}": v for k, v in data_cfg.items()},
            ),
            mode="offline" if args.wandb_offline else "online",
        )

    out_dir = Path(args.out_dir)
    if args.fold is not None:
        out_dir = out_dir / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        totals = dict(loss=0.0, multi=0.0, binary=0.0)
        n_batches = 0

        for wav, label, abnormal, _ in train_loader:
            wav = wav.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            abnormal = abnormal.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits_multi, logit_bin = model(wav)
                loss_multi = criterion_multi(logits_multi, label)
                loss_bin = criterion_bin(logit_bin, abnormal)
                loss = loss_multi + lambda_bin * loss_bin

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += float(loss.detach())
            totals["multi"] += float(loss_multi.detach())
            totals["binary"] += float(loss_bin.detach())
            n_batches += 1

        scheduler.step()
        for key in totals:
            totals[key] /= max(n_batches, 1)

        val = evaluate(model, val_loader, device, n_classes)
        score = selection_score(val)
        improved = score > best_score

        print(
            f"[{args.branch}] epoch {epoch:3d}/{epochs}  loss={totals['loss']:.4f} "
            f"(multi={totals['multi']:.4f} bin={totals['binary']:.4f})  "
            f"val: multiF1={val['multi_f1']:.4f} binF1={val['binary']['macro_f1']:.4f} "
            f"score={score:.4f}{'  *' if improved else ''}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.1f}s",
            flush=True,
        )

        if run is not None:
            log = {
                "epoch": epoch,
                "train/loss": totals["loss"],
                "train/loss_multi": totals["multi"],
                "train/loss_binary": totals["binary"],
                "train/lr": scheduler.get_last_lr()[0],
                "val/multi_f1": val["multi_f1"],
                "val/multi_balanced_acc": val["multi_balanced_acc"],
                "val/binary_f1": val["binary"]["macro_f1"],
                "val/binary_balanced_acc": val["binary"]["balanced_accuracy"],
                "val/window_multi_f1": val["window_multi_f1"],
                "val/window_binary_f1": val["window_bin_f1"],
                "val/score": score,
            }
            for key in ("sensitivity", "specificity", "auroc"):
                if val["binary"][key] is not None:
                    log[f"val/binary_{key}"] = val["binary"][key]
            run.log(log)

        checkpoint = dict(
            state_dict=model.state_dict(),
            model_config=model.config(),
            branch=args.branch,
            fold=args.fold,
            classes=classes,
            epoch=epoch,
            score=score,
            data_config=data_cfg,
            # Nguồn gốc huấn luyện: lưu thẳng danh sách bản ghi đã train để bước
            # đánh giá kiểm được checkpoint có từng nhìn thấy bản ghi đang chấm
            # hay không. Chỉ so chỉ số fold là chưa đủ - hai file H5 dựng bằng
            # hạt giống khác nhau có cùng chỉ số fold nhưng khác bản ghi.
            train_record_ids=list(dm.data_train.record_ids),
            split_source=dm.split_source,
            taxonomy=dm.taxonomy,
            h5_path=args.h5,
        )
        torch.save(checkpoint, out_dir / "last.pt")
        if improved:
            best_score, best_epoch = score, epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, out_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{args.branch}] dung som o epoch {epoch} (patience={patience})")
                break

    print(f"\n[{args.branch}] tot nhat: epoch {best_epoch}, score val={best_score:.4f}")

    # Báo cáo bằng checkpoint tốt nhất, đo trên TẬP TEST - tập chưa hề tham gia
    # chọn epoch hay dừng sớm. Nếu đo lại trên chính tập val thì con số bị thiên
    # lệch lạc quan: ta đã chọn ra epoch trông đẹp nhất trên đúng tập đó.
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    test_loader = dm.test_dataloader()
    final = evaluate(model, test_loader, device, n_classes)
    biased = dm.selection_set_is_eval_set

    subset_name = (
        f"fold {args.fold} (CV long: chon tren fold {(args.fold + 1) % n_folds})"
        if args.fold is not None
        else "split co dinh"
    )
    print(f"\n{'=' * 62}\n  {args.branch.upper()} - tap TEST: {subset_name}\n{'=' * 62}")
    if biased:
        print("  [CANH BAO] tap chon checkpoint TRUNG tap bao cao -> so lieu thien lech")
    print(format_binary_report(final["binary"], "head nhi phan"))
    print(format_binary_report(final["binary_from_multi"], "nhi phan suy tu head da lop"))
    print()
    print(format_multiclass_report(final["rec_label"], final["rec_pred_multi"], classes))

    summary = dict(
        branch=args.branch,
        fold=args.fold,
        best_epoch=best_epoch,
        val_score_at_best=best_score,
        # Cờ này đi kèm mọi con số để người đọc biết ngay có bị thiên lệch do chọn
        # epoch hay không, thay vì phải đọc lại mã nguồn.
        selection_set_equals_eval_set=bool(biased),
        n_eval_records=int(len(final["rec_label"])),
        eval_record_ids=list(final["record_ids"]),
        multi_f1=final["multi_f1"],
        multi_balanced_acc=final["multi_balanced_acc"],
        binary={k: v for k, v in final["binary"].items()},
        binary_from_multi={k: v for k, v in final["binary_from_multi"].items()},
        n_parameters=model.num_parameters(),
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n[done] checkpoint + summary.json o {out_dir}")

    if run is not None:
        run.summary.update(
            {
                "best/epoch": best_epoch,
                "best/score": best_score,
                "best/multi_f1": final["multi_f1"],
                "best/binary_f1": final["binary"]["macro_f1"],
            }
        )
        run.finish()

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train bo phan loai am tim/phoi bat thuong")
    ap.add_argument("--branch", required=True, choices=("heart", "lung"))
    ap.add_argument("--h5", default="/content/data_hls_cls.h5")
    ap.add_argument("--conf_dir", default="configs/hls-cls-t4.yml")
    ap.add_argument("--out_dir", default="/content/hls_cls_out")
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Chay cross-validation tren fold nay (bo qua cot split). "
        "Khong dat = dung split co dinh trong file H5.",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Ghi de so epoch trong config")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true", help="Tat mixed precision")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_offline", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
