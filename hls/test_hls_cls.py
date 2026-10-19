"""Đánh giá bộ phân loại âm tim/phổi bất thường, gồm cả cascade end-to-end.

Ba chế độ, xếp theo độ sạch giảm dần:

``clean``
    Phân loại trực tiếp bản ghi tim/phổi thuần (tập test của ``data_hls_cls.h5``).
    Đây là **trần trên** của bộ phân loại khi đầu vào hoàn hảo.

``cascade-synth`` — kết quả end-to-end CHÍNH
    Test split của ``data_hls.h5`` -> SPMamba -> hai bộ phân loại. Nhãn lấy từ
    thuộc tính ``heart``/``lung`` của từng hỗn hợp. Không rò rỉ, vì các bản ghi
    nguồn ở test split cũng chính là những bản ghi mà cả bộ tách âm lẫn bộ phân
    loại đều chưa từng thấy (với điều kiện ``data_hls_cls.h5`` được dựng bằng
    ``--align-split-with data_hls.h5``).

``cascade-real`` — chỉ để tham khảo
    145 file ``M####.wav`` thu thật -> SPMamba -> phân loại, nhãn từ ``Mix.csv``.
    Thực tế nhất, nhưng **có rò rỉ**: các take ``H####``/``L####`` dựng nên những
    hỗn hợp này trùng lặp nặng với 50 bản ghi gốc dùng để huấn luyện (53/145 file
    tim và 57/145 file phổi trùng byte-for-byte). Không được trích dẫn như kết quả
    chính.

Ví dụ::

    python test_hls_cls.py --mode clean \
        --heart_ckpt out/heart/best.pt --lung_ckpt out/lung/best.pt \
        --cls_h5 /content/data_hls_cls.h5

    python test_hls_cls.py --mode cascade-synth \
        --heart_ckpt out/heart/best.pt --lung_ckpt out/lung/best.pt \
        --sep_ckpt /content/drive/.../SPMamba-HLS-T4 --h5 /content/data_hls.h5

    python test_hls_cls.py --mode all --heart_ckpt ... --lung_ckpt ... --sep_ckpt ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from hls_cls_metrics import (
    balanced_accuracy,
    binary_report,
    format_binary_report,
    format_multiclass_report,
    macro_f1,
)
from look2hear.datas.hls_labels import (
    BRANCHES,
    TAXONOMIES,
    label_for,
    map_label,
    taxonomy_classes,
)
from look2hear.models import load_checkpoint

DEFAULT_DATASET_ROOT = (
    "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+"
    "using+digital+stethoscope/HLS-CMDS"
)

# Thứ tự kênh do FixedOrderLossWrapper quy định khi huấn luyện tách âm:
# s1 = tim -> ch0, s2 = phổi -> ch1 (xem configs/spmamba-hls-t4.yml).
CHANNEL_OF = {"heart": 0, "lung": 1}


# --------------------------------------------------------------------------- #
# Suy luận
# --------------------------------------------------------------------------- #
class BranchClassifier:
    """Bọc một bộ phân loại + tham số cắt cửa sổ đúng như lúc huấn luyện."""

    def __init__(self, ckpt_path: str, device: str, hop_ratio: float = 0.5):
        model, meta = load_checkpoint(ckpt_path, map_location=device)
        self.model = model.to(device).eval()
        self.device = device
        self.branch = meta.get("branch") or model.branch
        self.classes: List[str] = meta.get("classes") or model.classes
        self.sample_rate = int(model.frontend_config["sample_rate"])

        segment = float((meta.get("data_config") or {}).get("segment", 5.0))
        self.seg_len = int(round(segment * self.sample_rate))
        self.hop = max(1, int(round(self.seg_len * hop_ratio)))

        # Phân loại (full / public) quyết định cách quy nhãn thật về chỉ số lớp ở
        # các chế độ cascade, nên phải lấy từ chính checkpoint.
        self.taxonomy = str(meta.get("taxonomy", "full"))
        if self.taxonomy not in TAXONOMIES:
            raise ValueError(
                f"Checkpoint ghi taxonomy={self.taxonomy!r}, khong thuoc {TAXONOMIES}"
            )
        expected = taxonomy_classes(self.branch, self.taxonomy)
        if list(self.classes) != list(expected):
            raise ValueError(
                f"Bang lop trong checkpoint khac bang lop hien tai cho nhanh "
                f"'{self.branch}' (taxonomy={self.taxonomy}).\n"
                f"  checkpoint: {self.classes}\n  hien tai : {expected}\n"
                "  Nhan se bi gan sai. Train lai hoac kiem tra hls_labels.py."
            )

        # Nguồn gốc huấn luyện, dùng để chặn việc chấm điểm trên chính bản ghi đã
        # huấn luyện (xem assert_unseen).
        self.fold = meta.get("fold")
        self.train_record_ids = set(meta.get("train_record_ids") or ())
        if not self.train_record_ids:
            print(
                f"[cls:{self.branch}][canh bao] checkpoint khong luu train_record_ids "
                "(checkpoint cu) -> KHONG the kiem tra ro ri du lieu. Train lai de co "
                "canh bao tu dong."
            )
        print(
            f"[cls:{self.branch}] {ckpt_path} | epoch={meta.get('epoch')} "
            f"segment={segment}s ({self.seg_len} mau) | {len(self.classes)} lop"
        )

    def assert_unseen(self, record_ids: Sequence[str], context: str) -> None:
        """Dừng nếu checkpoint đã từng huấn luyện trên các bản ghi sắp chấm.

        Đây là chốt chặn cuối cùng chống rò rỉ. Không có nó, một checkpoint
        cross-validation đem chấm nhầm tập sẽ cho điểm rất cao và trông hoàn toàn
        hợp lệ - không có dấu hiệu nào trong báo cáo cho thấy con số là vô nghĩa.
        """
        if not self.train_record_ids:
            return
        overlap = sorted(self.train_record_ids & set(record_ids))
        if overlap:
            raise RuntimeError(
                f"[{self.branch}] RO RI DU LIEU o che do '{context}': "
                f"{len(overlap)}/{len(set(record_ids))} ban ghi dang cham diem "
                f"NAM TRONG tap huan luyen cua checkpoint: {overlap[:8]}"
                f"{' ...' if len(overlap) > 8 else ''}\n"
                f"  Checkpoint nay train voi fold={self.fold}. Hay cham bang dung "
                "fold cua no (--fold), hoac dung checkpoint train tren split co dinh."
            )

    @torch.no_grad()
    def predict(self, wav: np.ndarray) -> Tuple[np.ndarray, float]:
        """Sóng âm 1 chiều -> ``(prob_multi [n_classes], prob_abnormal)``.

        Cắt cửa sổ trượt rồi lấy trung bình xác suất, đúng như lúc đánh giá trong
        huấn luyện, để số liệu hai bên so sánh được với nhau.
        """
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if len(wav) < self.seg_len:
            wav = np.pad(wav, (0, self.seg_len - len(wav)))

        starts = list(range(0, len(wav) - self.seg_len + 1, self.hop))
        last = len(wav) - self.seg_len
        if not starts:
            starts = [0]
        elif starts[-1] != last:
            starts.append(last)

        windows = np.stack([wav[s : s + self.seg_len] for s in starts])
        batch = torch.from_numpy(windows).to(self.device)
        logits_multi, logit_bin = self.model(batch)
        prob_multi = torch.softmax(logits_multi.float(), dim=-1).cpu().numpy().mean(axis=0)
        prob_bin = float(torch.sigmoid(logit_bin.float()).cpu().numpy().mean())
        return prob_multi, prob_bin


def load_separator(ckpt: str, device: str):
    """Nạp SPMamba; chấp nhận đường dẫn file hoặc thư mục chứa checkpoint."""
    from test_evaluate import load_model

    if os.path.isdir(ckpt):
        cands = glob.glob(os.path.join(ckpt, "*best*.ckpt")) or glob.glob(
            os.path.join(ckpt, "*.ckpt")
        )
        if not cands:
            raise FileNotFoundError(f"Khong thay .ckpt trong {ckpt}")
        ckpt = max(cands, key=os.path.getmtime)
        print("[sep] tu dong chon checkpoint:", ckpt)
    return load_model(ckpt, device)


@torch.no_grad()
def separate(model, mixture: np.ndarray, device: str) -> np.ndarray:
    """Hỗn hợp 1 chiều -> ``[n_src, T]`` (ch0 = tim, ch1 = phổi)."""
    x = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    return model(x)[0].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Gom kết quả
# --------------------------------------------------------------------------- #
class BranchResults:
    """Tích luỹ dự đoán của một nhánh rồi kết xuất báo cáo."""

    def __init__(self, branch: str, class_names: Sequence[str]):
        self.branch = branch
        self.classes = list(class_names)
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_abnormal: List[int] = []
        self.score_abnormal: List[float] = []
        self.score_from_multi: List[float] = []
        self.names: List[str] = []

    def add(self, name: str, true_idx: int, prob_multi: np.ndarray, prob_bin: float) -> None:
        self.names.append(name)
        self.y_true.append(int(true_idx))
        self.y_pred.append(int(np.argmax(prob_multi)))
        self.y_abnormal.append(int(true_idx != 0))       # chỉ số 0 luôn là Normal
        self.score_abnormal.append(float(prob_bin))
        self.score_from_multi.append(float(1.0 - prob_multi[0]))

    def summary(self) -> Dict:
        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)
        n_classes = len(self.classes)
        return dict(
            branch=self.branch,
            n=len(y_true),
            multi_f1=macro_f1(y_true, y_pred, n_classes),
            multi_balanced_acc=balanced_accuracy(y_true, y_pred, n_classes),
            binary=binary_report(np.asarray(self.y_abnormal), np.asarray(self.score_abnormal)),
            # Quyết định cứng lấy đúng từ argmax để không mâu thuẫn với ma trận
            # nhầm lẫn đa lớp in ngay bên dưới; điểm số vẫn dùng cho AUROC.
            binary_from_multi=binary_report(
                np.asarray(self.y_abnormal),
                np.asarray(self.score_from_multi),
                y_pred=(y_pred != 0).astype(int),
            ),
        )

    def report(self) -> str:
        s = self.summary()
        lines = [
            f"\n{'-' * 62}",
            f"  {self.branch.upper()}  (n = {s['n']} ban ghi)",
            f"{'-' * 62}",
            format_binary_report(s["binary"], "head nhi phan"),
            format_binary_report(s["binary_from_multi"], "nhi phan suy tu head da lop"),
            "",
            format_multiclass_report(
                np.asarray(self.y_true), np.asarray(self.y_pred), self.classes
            ),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Chế độ 1: clean
# --------------------------------------------------------------------------- #
def run_clean(
    cls_h5: str, classifiers: Dict[str, BranchClassifier], split: str, fold: Optional[int]
) -> Dict[str, BranchResults]:
    results: Dict[str, BranchResults] = {}
    with h5py.File(cls_h5, "r") as f:
        for branch, clf in classifiers.items():
            grp = f[branch]

            # File H5 phải dùng đúng bảng lớp mà checkpoint được train. Lệch bảng
            # lớp thì label_idx trong file trỏ sang lớp khác và toàn bộ báo cáo bị
            # gán nhãn sai mà không có dấu hiệu nào.
            h5_classes = json.loads(grp.attrs["classes"])
            if list(h5_classes) != list(clf.classes):
                raise ValueError(
                    f"[{branch}] bang lop trong {cls_h5} khac bang lop cua checkpoint.\n"
                    f"  H5        : {h5_classes}\n  checkpoint: {list(clf.classes)}\n"
                    "  label_idx trong H5 se tro sai lop. Dung lai."
                )
            h5_normal = str(grp.attrs["normal_class"])
            if h5_classes[0] != h5_normal:
                raise ValueError(
                    f"[{branch}] H5 co classes[0]={h5_classes[0]!r} nhung "
                    f"normal_class={h5_normal!r}; ca pipeline gia dinh chi so 0 la Normal."
                )

            splits = [s.decode() if isinstance(s, bytes) else s for s in grp["split"][:]]
            folds = grp["fold"][:]
            if fold is None:
                keep = [i for i, s in enumerate(splits) if s == split]
                desc = f"split={split}"
            else:
                keep = [i for i, k in enumerate(folds) if int(k) == fold]
                desc = f"fold={fold}"
            if not keep:
                raise RuntimeError(f"[{branch}] khong co ban ghi nao voi {desc}")

            ids = [
                x.decode() if isinstance(x, bytes) else x for x in grp["record_ids"][keep]
            ]
            clf.assert_unseen(ids, f"clean/{desc}")
            labels = grp["label_idx"][keep].astype(int)
            audio = grp["audio"][keep]

            res = BranchResults(branch, clf.classes)
            for name, label, wav in zip(ids, labels, audio):
                prob_multi, prob_bin = clf.predict(wav)
                res.add(name, int(label), prob_multi, prob_bin)
            results[branch] = res
            print(f"[clean:{branch}] {len(keep)} ban ghi ({desc})")
    return results


# --------------------------------------------------------------------------- #
# Chế độ 2: cascade trên hỗn hợp tổng hợp (kết quả chính)
# --------------------------------------------------------------------------- #
def run_cascade_synth(
    sep_h5: str,
    separator,
    classifiers: Dict[str, BranchClassifier],
    device: str,
    split: str = "test",
    limit: Optional[int] = None,
) -> Dict[str, BranchResults]:
    results = {b: BranchResults(b, c.classes) for b, c in classifiers.items()}
    class_index = {b: {name: i for i, name in enumerate(c.classes)} for b, c in classifiers.items()}

    with h5py.File(sep_h5, "r") as f:
        if "split" not in f:
            raise ValueError(
                f"{sep_h5} khong co /split. Can file build bang build_hls_h5.py "
                "che do dynamic (co source-disjoint split) de danh gia cascade."
            )
        ids = [x.decode() if isinstance(x, bytes) else x for x in f["mixture_ids"][:]]
        splits = [x.decode() if isinstance(x, bytes) else x for x in f["split"][:]]
        keep = [i for i, s in enumerate(splits) if s == split]
        if limit:
            keep = keep[:limit]
        print(f"[cascade-synth] {len(keep)} hon hop o split '{split}'")

        # Các bản ghi nguồn đứng sau những hỗn hợp này không được nằm trong tập
        # huấn luyện của bộ phân loại, nếu không thí nghiệm cascade mất ý nghĩa.
        for branch, clf in classifiers.items():
            stems = set()
            for i in keep:
                name = f["mixtures"][ids[i]].attrs[branch]
                stems.add(str(name.decode() if isinstance(name, bytes) else name))
            clf.assert_unseen(sorted(stems), "cascade-synth")

        for n_done, i in enumerate(keep, 1):
            mid = ids[i]
            grp = f["mixtures"][mid]
            mixture = grp["mix"][:]
            est = separate(separator, mixture, device)

            for branch, clf in classifiers.items():
                stem = grp.attrs[branch]
                if isinstance(stem, bytes):
                    stem = stem.decode()
                true_name = map_label(label_for(str(stem), branch), branch, clf.taxonomy)
                prob_multi, prob_bin = clf.predict(est[CHANNEL_OF[branch]])
                results[branch].add(
                    f"{mid}:{stem}", class_index[branch][true_name], prob_multi, prob_bin
                )

            if n_done % 20 == 0 or n_done == len(keep):
                print(f"  ... {n_done}/{len(keep)}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Chế độ 3: cascade trên hỗn hợp thu thật (tham khảo, có rò rỉ)
# --------------------------------------------------------------------------- #
def run_cascade_real(
    dataset_root: str,
    separator,
    classifiers: Dict[str, BranchClassifier],
    device: str,
    limit: Optional[int] = None,
) -> Dict[str, BranchResults]:
    import soundfile as sf

    root = Path(dataset_root)
    csv_path = root / "Mix.csv"
    mix_dir = root / "Mix" / "Mix"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Khong thay {csv_path}")

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit:
        rows = rows[:limit]

    type_col = {"heart": "Heart Sound Type", "lung": "Lung Sound Type"}
    results = {b: BranchResults(b, c.classes) for b, c in classifiers.items()}
    class_index = {b: {name: i for i, name in enumerate(c.classes)} for b, c in classifiers.items()}

    print(f"[cascade-real] {len(rows)} hon hop thu that tu Mix.csv")
    missing = 0
    for n_done, row in enumerate(rows, 1):
        mid = row["Mixed Sound ID"].strip()
        path = mix_dir / f"{mid}.wav"
        if not path.is_file():
            missing += 1
            continue
        mixture, _ = sf.read(path, dtype="float32", always_2d=False)
        if mixture.ndim > 1:
            mixture = mixture[:, 0]
        est = separate(separator, mixture, device)

        for branch, clf in classifiers.items():
            true_name = map_label(row[type_col[branch]].strip(), branch, clf.taxonomy)
            if true_name not in class_index[branch]:
                raise ValueError(
                    f"{mid}: Mix.csv ghi loai '{row[type_col[branch]].strip()}' -> "
                    f"'{true_name}' khong co trong bang lop {branch}: "
                    f"{list(class_index[branch])}"
                )
            prob_multi, prob_bin = clf.predict(est[CHANNEL_OF[branch]])
            results[branch].add(mid, class_index[branch][true_name], prob_multi, prob_bin)

        if n_done % 20 == 0 or n_done == len(rows):
            print(f"  ... {n_done}/{len(rows)}", flush=True)

    if missing:
        print(f"[canh bao] thieu {missing} file WAV, da bo qua")
    return results


# --------------------------------------------------------------------------- #
# In kết quả
# --------------------------------------------------------------------------- #
LEAKAGE_WARNING = (
    "  [!] CANH BAO RO RI DU LIEU: cac take H####/L#### dung de tao nhung hon hop\n"
    "      nay trung lap nang voi 50 ban ghi goc dung de HUAN LUYEN bo phan loai\n"
    "      (53/145 file tim va 57/145 file phoi trung byte-for-byte). Con so duoi\n"
    "      day LAC QUAN HON thuc te - chi dung de tham khao dinh tinh, KHONG dung\n"
    "      lam ket qua chinh. Ket qua chinh la che do 'cascade-synth'."
)

MODE_TITLE = {
    "clean": ("CHE DO 1/3 - CLEAN (tran tren, dau vao hoan hao)", None),
    "cascade-synth": ("CHE DO 2/3 - CASCADE tren hon hop TONG HOP  [KET QUA CHINH]", None),
    "cascade-real": ("CHE DO 3/3 - CASCADE tren hon hop THU THAT  [tham khao]", LEAKAGE_WARNING),
}


def print_mode(mode: str, results: Dict[str, BranchResults]) -> Dict[str, Dict]:
    title, warning = MODE_TITLE[mode]
    print(f"\n\n{'=' * 62}\n  {title}\n{'=' * 62}")
    if warning:
        print(warning)
    summaries = {}
    for branch in BRANCHES:
        if branch in results:
            print(results[branch].report())
            summaries[branch] = results[branch].summary()
    return summaries


def print_gap(all_summaries: Dict[str, Dict[str, Dict]]) -> None:
    """Bảng so sánh Clean vs Cascade: đây chính là phần chất lượng mất do tách âm."""
    if "clean" not in all_summaries or "cascade-synth" not in all_summaries:
        return
    print(f"\n\n{'=' * 62}\n  CHENH LECH DO BUOC TACH AM GAY RA\n{'=' * 62}")
    print(f"  {'nhanh':<8s} {'chi so':<22s} {'clean':>8s} {'cascade':>9s} {'chenh':>8s}")
    print(f"  {'-' * 58}")
    for branch in BRANCHES:
        clean = all_summaries["clean"].get(branch)
        casc = all_summaries["cascade-synth"].get(branch)
        if not clean or not casc:
            continue
        pairs = [
            ("macro-F1 da lop", clean["multi_f1"], casc["multi_f1"]),
            ("balanced-acc da lop", clean["multi_balanced_acc"], casc["multi_balanced_acc"]),
            ("macro-F1 nhi phan", clean["binary"]["macro_f1"], casc["binary"]["macro_f1"]),
        ]
        for name, a, b in pairs:
            print(f"  {branch:<8s} {name:<22s} {a:>8.4f} {b:>9.4f} {b - a:>+8.4f}")
    print(
        "\n  Chenh lech am = chat luong mat di khi phan loai tren dau ra tach am\n"
        "  thay vi ban ghi thuan. Neu chenh lech qua lon, buoc tiep theo la tang\n"
        "  cuong du lieu huan luyen bang chinh dau ra cua SPMamba."
    )


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Danh gia bo phan loai am tim/phoi bat thuong")
    ap.add_argument(
        "--mode",
        default="all",
        choices=("clean", "cascade-synth", "cascade-real", "all"),
        help="'all' chay ca ba che do",
    )
    ap.add_argument("--heart_ckpt", required=True, help="best.pt cua nhanh tim")
    ap.add_argument("--lung_ckpt", required=True, help="best.pt cua nhanh phoi")
    ap.add_argument("--cls_h5", default="/content/data_hls_cls.h5", help="[clean]")
    ap.add_argument("--split", default="test", help="[clean] split can danh gia")
    ap.add_argument("--fold", type=int, default=None, help="[clean] danh gia mot fold thay vi split")
    ap.add_argument("--sep_ckpt", default=None, help="[cascade] checkpoint SPMamba (file hoac thu muc)")
    ap.add_argument("--h5", default="/content/data_hls.h5", help="[cascade-synth]")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT, help="[cascade-real]")
    ap.add_argument("--limit", type=int, default=None, help="Gioi han so mau (de chay thu nhanh)")
    ap.add_argument("--out_json", default=None, help="Ghi toan bo so lieu ra file JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    classifiers = {
        "heart": BranchClassifier(args.heart_ckpt, device),
        "lung": BranchClassifier(args.lung_ckpt, device),
    }
    for branch, clf in classifiers.items():
        if clf.branch != branch:
            raise ValueError(
                f"Checkpoint truyen vao --{branch}_ckpt lai la cua nhanh "
                f"'{clf.branch}'. Da hoan doi hai checkpoint?"
            )

    modes = ["clean", "cascade-synth", "cascade-real"] if args.mode == "all" else [args.mode]
    needs_separator = any(m.startswith("cascade") for m in modes)
    separator = None
    if needs_separator:
        if not args.sep_ckpt:
            raise ValueError("Che do cascade can --sep_ckpt (checkpoint SPMamba)")
        separator, sep_sr = load_separator(args.sep_ckpt, device)
        cls_sr = classifiers["heart"].sample_rate
        if int(sep_sr) != int(cls_sr):
            raise ValueError(
                f"Tan so lay mau khong khop: bo tach am {sep_sr} Hz, bo phan loai "
                f"{cls_sr} Hz. Dau ra tach am se bi hieu sai o frontend log-mel."
            )

    def write_json() -> None:
        if not args.out_json:
            return
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, indent=2, ensure_ascii=False)

    all_summaries: Dict[str, Dict[str, Dict]] = {}
    failed: Dict[str, str] = {}
    for mode in modes:
        try:
            if mode == "clean":
                results = run_clean(args.cls_h5, classifiers, args.split, args.fold)
            elif mode == "cascade-synth":
                results = run_cascade_synth(
                    args.h5, separator, classifiers, device, limit=args.limit
                )
            else:
                results = run_cascade_real(
                    args.dataset_root, separator, classifiers, device, limit=args.limit
                )
        except Exception as exc:  # noqa: BLE001
            # Một chế độ hỏng không được kéo theo các chế độ khác. Trước đây lỗi ở
            # 'cascade-real' - chế độ CHỈ để tham khảo - sẽ xoá sạch cả kết quả
            # chính đã tính xong.
            failed[mode] = f"{type(exc).__name__}: {exc}"
            print(f"\n[loi] che do '{mode}' that bai: {failed[mode]}")
            if len(modes) > 1:
                print("      cac che do khac van duoc chay va bao cao binh thuong.")
            continue
        all_summaries[mode] = print_mode(mode, results)
        write_json()   # ghi dần, mất một chế độ vẫn còn kết quả các chế độ trước

    print_gap(all_summaries)
    write_json()
    if args.out_json and all_summaries:
        print(f"\n[done] da ghi so lieu ra {args.out_json}")

    if failed:
        print(f"\n{'=' * 62}")
        for mode, err in failed.items():
            print(f"  [that bai] {mode}: {err}")
        print("=" * 62)
        if not all_summaries:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
