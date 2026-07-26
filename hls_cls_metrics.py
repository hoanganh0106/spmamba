"""Chỉ số đánh giá cho phân loại âm tim/phổi, dùng chung cho train và test.

Cài bằng numpy thuần thay vì scikit-learn: môi trường Colab của repo (Cell 2 của
``train_hls_colab.ipynb``) không cài scikit-learn, nên phụ thuộc vào nó sẽ làm
script chết giữa chừng sau khi đã train xong.

Với dữ liệu chỉ 50 bản ghi mỗi nhánh và lớp nhỏ nhất có 2 bản ghi, **accuracy là
chỉ số gây hiểu nhầm**: đoán "bất thường" cho mọi mẫu đã cho 82% accuracy ở nhánh
tim. Vì vậy mọi hàm ở đây đều ưu tiên macro-F1 và balanced accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``cm[i, j]`` = số mẫu nhãn thật ``i`` bị đoán thành ``j``."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        cm[t, p] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Precision / recall / F1 / support cho từng lớp."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return dict(precision=precision, recall=recall, f1=f1, support=support)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Trung bình F1, **chỉ tính trên các lớp thực sự có mặt** trong y_true.

    Lớp vắng mặt sẽ luôn có F1 = 0 và kéo tụt trung bình một cách vô nghĩa. Với
    tập test 5 bản ghi trên 10 lớp thì phần lớn lớp sẽ vắng mặt, nên bỏ qua chúng
    là cách duy nhất để con số còn đọc được.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    stats = per_class_prf(cm)
    present = stats["support"] > 0
    return float(stats["f1"][present].mean()) if present.any() else 0.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Trung bình recall trên các lớp có mặt."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    stats = per_class_prf(cm)
    present = stats["support"] > 0
    return float(stats["recall"][present].mean()) if present.any() else 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """AUROC nhị phân theo thống kê Mann-Whitney U (xử lý đúng trường hợp hoà).

    Trả ``None`` khi chỉ có một lớp — lúc đó AUROC không xác định. Trường hợp này
    xảy ra thật với split kế thừa từ ``data_hls.h5`` (tập test nhánh tim không có
    bản ghi Normal nào), nên phải trả None thay vì một con số vô nghĩa.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0   # hạng trung bình cho nhóm hoà
        i = j + 1

    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Chỉ số cho head nhị phân (1 = bất thường).

    ``scores`` luôn dùng để xếp hạng (AUROC). ``y_pred`` cho phép truyền thẳng
    quyết định cứng thay vì cắt ``scores`` ở ``threshold``.

    Cần tham số này khi điểm số suy ra từ head đa lớp: ``1 - P(Normal) >= 0.5``
    **không** tương đương với ``argmax != Normal``. Với 10 lớp, một mẫu có
    P(Normal)=0.4 và phần còn lại rải đều sẽ bị coi là bất thường theo ngưỡng
    nhưng vẫn là Normal theo argmax - hai kết luận trái nhau in cạnh nhau trong
    cùng một báo cáo.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_pred is None:
        y_pred = (np.asarray(scores) >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, 2)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    return dict(
        macro_f1=macro_f1(y_true, y_pred, 2),
        balanced_accuracy=balanced_accuracy(y_true, y_pred, 2),
        accuracy=accuracy(y_true, y_pred),
        # Trong bối cảnh y tế: sensitivity = tỉ lệ ca bất thường được bắt đúng
        # (bỏ sót nguy hiểm hơn báo động nhầm), specificity = tỉ lệ ca bình
        # thường không bị báo nhầm.
        sensitivity=float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        specificity=float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        auroc=roc_auc(y_true, scores),
        n_normal=int((y_true == 0).sum()),
        n_abnormal=int((y_true == 1).sum()),
    )


def format_confusion(cm: np.ndarray, class_names: Sequence[str], max_width: int = 14) -> str:
    """Ma trận nhầm lẫn dạng bảng chữ, hàng = nhãn thật."""
    names = [n[:max_width] for n in class_names]
    label_w = max(len(n) for n in names) if names else 4
    cell_w = max(4, max(len(str(int(v))) for v in cm.flat) + 1) if cm.size else 4

    header = " " * (label_w + 2) + "".join(f"{i:>{cell_w}d}" for i in range(len(names)))
    lines = [header, " " * (label_w + 2) + "-" * (cell_w * len(names))]
    for i, name in enumerate(names):
        row = "".join(f"{int(v):>{cell_w}d}" for v in cm[i])
        lines.append(f"{name:<{label_w}s} |{row}   ({int(cm[i].sum())})")
    lines.append("(cot = du doan theo chi so lop o hang tren)")
    return "\n".join(lines)


def format_multiclass_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> str:
    """Bảng precision/recall/F1 từng lớp + ma trận nhầm lẫn."""
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    stats = per_class_prf(cm)

    lines = [
        f"  {'lop':<24s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'n':>4s}",
        f"  {'-' * 49}",
    ]
    for i, name in enumerate(class_names):
        if stats["support"][i] == 0:
            continue  # lớp vắng mặt trong tập này -> không có gì để báo cáo
        lines.append(
            f"  {name[:24]:<24s} {stats['precision'][i]:>6.3f} {stats['recall'][i]:>7.3f} "
            f"{stats['f1'][i]:>6.3f} {int(stats['support'][i]):>4d}"
        )
    absent = [class_names[i] for i in range(n_classes) if stats["support"][i] == 0]
    if absent:
        lines.append(f"  (khong co mau trong tap nay: {', '.join(absent)})")

    lines.append("")
    lines.append(f"  macro-F1 = {macro_f1(y_true, y_pred, n_classes):.4f}   "
                 f"balanced-acc = {balanced_accuracy(y_true, y_pred, n_classes):.4f}   "
                 f"acc = {accuracy(y_true, y_pred):.4f}")
    lines.append("")
    lines.append(format_confusion(cm, class_names))
    return "\n".join(lines)


def format_binary_report(report: Dict[str, Optional[float]], title: str = "nhi phan") -> str:
    def fmt(key: str) -> str:
        v = report.get(key)
        return "  n/a" if v is None else f"{v:.4f}"

    lines = [
        f"  [{title}] Normal={report['n_normal']}  Abnormal={report['n_abnormal']}",
        f"    macro-F1={fmt('macro_f1')}  balanced-acc={fmt('balanced_accuracy')}  "
        f"acc={fmt('accuracy')}",
        f"    sensitivity={fmt('sensitivity')}  specificity={fmt('specificity')}  "
        f"AUROC={fmt('auroc')}",
    ]
    if report.get("auroc") is None:
        lines.append(
            "    [canh bao] tap nay chi co MOT lop -> AUROC va F1 nhi phan khong "
            "dung duoc. Dung cross-validation de bao cao."
        )
    return "\n".join(lines)
