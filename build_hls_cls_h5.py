"""Dựng file HDF5 cho bài toán PHÂN LOẠI âm tim/phổi HLS-CMDS.

Khác với ``build_hls_h5.py`` (dựng hỗn hợp để huấn luyện tách âm), script này chỉ
đóng gói 50 bản ghi tim thuần + 50 bản ghi phổi thuần kèm nhãn bệnh lý.

Schema đầu ra::

    attrs: sample_rate, target_length, created_at, split_source, n_folds
    /heart/audio        float32 [50, 60000]
    /heart/record_ids   str     [50]      # "F_MSM_LUSB"
    /heart/label_idx    int64   [50]      # chỉ số trong attrs["classes"]
    /heart/is_abnormal  int64   [50]      # 0 = Normal, 1 = bất thường
    /heart/split        str     [50]      # "train" | "val" | "test"
    /heart/fold         int64   [50]      # 0..K-1, dùng cho cross-validation
    /heart.attrs: classes (JSON), normal_class, n_records
    /lung/  ... cấu trúc y hệt

Chống rò rỉ dữ liệu — điểm quan trọng nhất
------------------------------------------
Chỉ có 50 bản ghi độc lập mỗi nhánh, mỗi bản ghi dài 15 s sẽ được cắt thành nhiều
cửa sổ lúc huấn luyện. Chia ngẫu nhiên theo cửa sổ sẽ đặt các cửa sổ của cùng một
bản ghi vào cả train lẫn test và cho accuracy ảo rất cao. Vì vậy **mọi phép chia ở
đây đều theo bản ghi**.

Với ``--align-split-with data_hls.h5``, split được kế thừa đúng phân hoạch nguồn mà
``build_hls_h5.py`` đã dùng cho bộ tách âm (đọc từ thuộc tính ``heart``/``lung``
của từng hỗn hợp và mảng ``/split``). Nhờ vậy bản ghi dùng để test bộ phân loại
cũng là bản ghi mà SPMamba chưa từng thấy — điều kiện cần để thí nghiệm cascade
end-to-end có ý nghĩa.

Ví dụ::

    python build_hls_cls_h5.py \
        --dataset-root "hls-cmds_.../HLS-CMDS" \
        --align-split-with /content/data_hls.h5 \
        --out /content/data_hls_cls.h5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

# ``look2hear/datas/__init__.py`` kéo theo torch + pytorch_lightning, nhưng dựng
# dữ liệu thì không cần tới chúng. Nạp thẳng module nhãn theo đường dẫn để script
# này chạy được ở môi trường chỉ có numpy + h5py (ví dụ máy cá nhân, trước khi
# upload H5 lên Colab). ``hls_labels`` cố ý chỉ dùng thư viện chuẩn.
def _load_hls_labels():
    import importlib.util

    path = Path(__file__).resolve().parent / "look2hear" / "datas" / "hls_labels.py"
    spec = importlib.util.spec_from_file_location("hls_labels", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Khong nap duoc bang nhan tu {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_labels = _load_hls_labels()
BRANCHES = _labels.BRANCHES
NORMAL_CLASS = _labels.NORMAL_CLASS
TAXONOMIES = _labels.TAXONOMIES
classes = _labels.classes
taxonomy_classes = _labels.taxonomy_classes
map_label = _labels.map_label
label_for = _labels.label_for
list_records = _labels.list_records
verify_against_csv = _labels.verify_against_csv
wav_dir = _labels.wav_dir


DEFAULT_DATASET_ROOT = (
    "hls-cmds_+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+"
    "using+digital+stethoscope/HLS-CMDS"
)


# --------------------------------------------------------------------------- #
# Đọc WAV
# --------------------------------------------------------------------------- #
def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Đọc WAV thành mono float32 trong [-1, 1]."""
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except ImportError:
        from scipy.io import wavfile

        sample_rate, raw = wavfile.read(path)
        if np.issubdtype(raw.dtype, np.floating):
            data = raw.astype(np.float32)
        elif np.issubdtype(raw.dtype, np.signedinteger):
            data = raw.astype(np.float32) / float(np.iinfo(raw.dtype).max)
        else:
            info = np.iinfo(raw.dtype)
            data = (raw.astype(np.float32) - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1, dtype=np.float32)
    return data.astype(np.float32, copy=False), int(sample_rate)


def _fit_length(data: np.ndarray, target_length: int, label: str) -> np.ndarray:
    if len(data) == target_length:
        return data.astype(np.float32, copy=False)
    if len(data) > target_length:
        print(f"[warn] {label}: cat {len(data)} -> {target_length} mau")
        return data[:target_length].astype(np.float32, copy=False)
    print(f"[warn] {label}: dem {len(data)} -> {target_length} mau")
    return np.pad(data, (0, target_length - len(data))).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Gán split
# --------------------------------------------------------------------------- #
def split_from_separation_h5(sep_h5: Path) -> Dict[str, Dict[str, str]]:
    """Truy ra phân hoạch nguồn mà ``build_hls_h5.py`` đã dùng.

    Trả ``{"heart": {ten_ban_ghi: split}, "lung": {...}}``.

    ``build_hls_h5.py --mode dynamic`` (mặc định) chia nguồn thành các pool rời
    nhau rồi ghi tên bản ghi vào thuộc tính ``heart``/``lung`` của từng hỗn hợp,
    kèm mảng ``/split``. Vì các pool rời nhau nên mỗi bản ghi chỉ thuộc đúng một
    split; nếu không đúng như vậy thì file H5 được build bằng ``--no-source-split``
    và không dùng để căn split được.
    """
    sep_h5 = Path(sep_h5)
    if not sep_h5.is_file():
        raise FileNotFoundError(f"Khong thay H5 tach am: {sep_h5}")

    seen: Dict[str, Dict[str, set]] = {b: defaultdict(set) for b in BRANCHES}
    with h5py.File(sep_h5, "r") as f:
        if "split" not in f:
            raise ValueError(
                f"{sep_h5} khong co dataset /split. File nay duoc build bang "
                "--no-source-split nen khong the ke thua phan hoach nguon. "
                "Build lai data_hls.h5 khong dung co do, hoac bo --align-split-with."
            )
        ids = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f["mixture_ids"][:]]
        splits = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f["split"][:]]
        for mid, split in zip(ids, splits):
            attrs = f["mixtures"][mid].attrs
            if "heart" not in attrs or "lung" not in attrs:
                raise ValueError(
                    f"{sep_h5}: hon hop {mid} thieu thuoc tinh heart/lung. "
                    "Chi H5 build o che do 'dynamic' moi luu duoc nguon."
                )
            for branch in BRANCHES:
                name = attrs[branch]
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                seen[branch][str(name)].add(str(split))

    result: Dict[str, Dict[str, str]] = {}
    for branch in BRANCHES:
        leaked = {k: sorted(v) for k, v in seen[branch].items() if len(v) > 1}
        if leaked:
            detail = "; ".join(f"{k} -> {v}" for k, v in list(leaked.items())[:5])
            raise ValueError(
                f"[{branch}] {len(leaked)} ban ghi xuat hien o nhieu split trong "
                f"{sep_h5}: {detail}. Phan hoach nguon khong roi nhau nen khong "
                "the dung de can split cho bo phan loai."
            )
        result[branch] = {k: next(iter(v)) for k, v in seen[branch].items()}
    return result


def stratified_group_split(
    labels: Sequence[int],
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> List[str]:
    """Chia theo bản ghi, phân tầng theo lớp.

    Duyệt từng lớp và rải các bản ghi của lớp đó lần lượt vào test / val / train
    để lớp hiếm (S4 chỉ có 2 bản ghi) không bị dồn hết vào một split.
    """
    n = len(labels)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)

    # Phân bổ theo phần dư lớn nhất (Hamilton) thay vì làm tròn từng lớp.
    # Làm tròn từng lớp cho ra 0 với mọi lớp có ≤5 bản ghi ở test_frac=0.10
    # (round(5*0.1)=round(0.5)=0 theo quy tắc làm tròn về số chẵn), nên đúng những
    # lớp hiếm nhất - S4 chỉ có 2 bản ghi - luôn vắng mặt khỏi tập test, và tổng
    # số mẫu test cũng không đạt tỉ lệ yêu cầu.
    def apportion(frac: float) -> Dict[int, int]:
        quotas = {lab: len(members) * frac for lab, members in by_class.items()}
        base = {lab: int(q) for lab, q in quotas.items()}
        target = int(round(n * frac))
        remaining = target - sum(base.values())
        order = sorted(by_class, key=lambda lab: (-(quotas[lab] - base[lab]), lab))
        for lab in order:
            if remaining <= 0:
                break
            base[lab] += 1
            remaining -= 1
        return base

    n_test_by_class = apportion(test_frac)
    n_val_by_class = apportion(val_frac)

    assigned = ["train"] * n
    for lab in sorted(by_class):
        members = by_class[lab]
        rng.shuffle(members)
        n_test = n_test_by_class.get(lab, 0)
        n_val = n_val_by_class.get(lab, 0)
        # Luôn chừa ít nhất 1 bản ghi cho train ở mỗi lớp.
        while n_test + n_val >= len(members) and (n_test + n_val) > 0:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        for i in members[:n_test]:
            assigned[i] = "test"
        for i in members[n_test : n_test + n_val]:
            assigned[i] = "val"
    return assigned


FOLD_RESERVED = -1
"""Giá trị ``fold`` cho bản ghi bị giữ lại, không tham gia cross-validation."""


def assign_folds(
    labels: Sequence[int],
    n_folds: int,
    rng: np.random.Generator,
    reserved: Sequence[bool] | None = None,
) -> np.ndarray:
    """Grouped stratified K-fold: mỗi bản ghi là một nhóm, rải đều theo lớp.

    Điểm bắt đầu được xoay sau mỗi lớp; nếu lớp nào cũng rải từ fold 0 thì với
    nhiều lớp nhỏ (S4 chỉ 2 bản ghi) fold 0 sẽ phình ra còn fold cuối gần rỗng.

    ``reserved[i] = True`` thì bản ghi ``i`` nhận ``fold = -1`` và bị loại khỏi
    mọi fold. Dùng cho các bản ghi thuộc pool test của bộ tách âm: nếu để chúng
    lọt vào tập train của một fold nào đó thì bộ phân loại đã nhìn thấy nguồn của
    hỗn hợp dùng trong thí nghiệm cascade, đúng thứ mà ``--align-split-with``
    sinh ra để ngăn.
    """
    folds = np.full(len(labels), FOLD_RESERVED, dtype=np.int64)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        if reserved is not None and reserved[idx]:
            continue
        by_class[int(lab)].append(idx)
    offset = 0
    for lab in sorted(by_class):
        members = by_class[lab]
        rng.shuffle(members)
        for pos, idx in enumerate(members):
            folds[idx] = (pos + offset) % n_folds
        offset = (offset + len(members)) % n_folds
    return folds


# --------------------------------------------------------------------------- #
# Dựng file
# --------------------------------------------------------------------------- #
def build(
    dataset_root: Path,
    out: Path,
    sample_rate: int,
    target_length: int,
    align_with: Path | None,
    val_frac: float,
    test_frac: float,
    n_folds: int,
    seed: int,
    allow_unaligned: bool = False,
    taxonomy: str = "full",
) -> None:
    dataset_root = Path(dataset_root).expanduser().resolve()
    out = Path(out).expanduser().resolve()

    aligned: Dict[str, Dict[str, str]] | None = None
    if align_with is not None:
        aligned = split_from_separation_h5(align_with)
        print(f"[split] ke thua phan hoach nguon tu {align_with}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    string_dtype = h5py.string_dtype(encoding="utf-8")
    rng = np.random.default_rng(seed)

    with h5py.File(out, "w") as h5:
        h5.attrs["sample_rate"] = sample_rate
        h5.attrs["target_length"] = target_length
        h5.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["dataset_root"] = str(dataset_root)
        h5.attrs["split_source"] = str(align_with) if align_with else "stratified"
        h5.attrs["n_folds"] = n_folds
        h5.attrs["seed"] = seed
        h5.attrs["taxonomy"] = taxonomy

        for branch in BRANCHES:
            print(f"\n{'=' * 60}\n[{branch}]\n{'=' * 60}")
            verify_against_csv(dataset_root, branch)

            branch_classes = taxonomy_classes(branch, taxonomy)
            cls_to_idx = {name: i for i, name in enumerate(branch_classes)}
            stems = list_records(dataset_root, branch)
            directory = wav_dir(dataset_root, branch)

            audio = np.zeros((len(stems), target_length), dtype=np.float32)
            label_idx = np.zeros(len(stems), dtype=np.int64)
            abnormal = np.zeros(len(stems), dtype=np.int64)

            for i, stem in enumerate(stems):
                wav, sr = _read_wav(directory / f"{stem}.wav")
                if sr != sample_rate:
                    raise ValueError(f"{stem}.wav: mong doi {sample_rate} Hz, nhan {sr} Hz")
                audio[i] = _fit_length(wav, target_length, stem)
                # Luôn gán nhãn đầy đủ trước rồi mới rút gọn: phép đối chiếu chéo
                # với CSV chạy trên bảng nhãn đầy đủ, nên rút gọn sớm sẽ làm mất
                # khả năng phát hiện ghép nhãn sai.
                full_name = label_for(stem, branch)
                name = map_label(full_name, branch, taxonomy)
                label_idx[i] = cls_to_idx[name]
                # Nhãn nhị phân suy từ nhãn ĐẦY ĐỦ, nên không đổi theo phân loại.
                abnormal[i] = int(full_name != NORMAL_CLASS)

            if aligned is not None:
                table = aligned[branch]
                missing = [s for s in stems if s not in table]
                if missing and not allow_unaligned:
                    raise RuntimeError(
                        f"[{branch}] {len(missing)} ban ghi khong xuat hien trong H5 tach am: "
                        f"{missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
                        "  Khong doan bua split cho chung. Mac dinh 'train' la huong duy "
                        "nhat co the am tham co loi cho tap huan luyen, va neu that ra "
                        "chung thuoc pool test cua bo tach am thi thi nghiem cascade mat "
                        "gia tri.\n"
                        "  Kiem tra lai data_hls.h5 co dung la file da dung de train bo "
                        "tach am khong, hoac chay lai voi --allow-unaligned-records de "
                        "xep chung vao 'unused' (bi loai khoi ca train lan eval)."
                    )
                # 'unused' chứ không phải 'train': bản ghi không truy được nguồn thì
                # loại hẳn, an toàn hơn là đưa vào huấn luyện.
                splits = [table.get(s, "unused") for s in stems]
                if missing:
                    print(
                        f"[split][canh bao] {len(missing)} ban ghi khong co trong H5 tach am "
                        f"-> xep vao 'unused', bi loai khoi ca train lan eval: {missing[:5]}"
                        f"{' ...' if len(missing) > 5 else ''}"
                    )
            else:
                splits = stratified_group_split(label_idx, val_frac, test_frac, rng)
                print(f"[split] chia phan tang theo ban ghi (seed={seed})")

            # Bản ghi thuộc pool test của bộ tách âm (và bản ghi 'unused') được giữ
            # lại khỏi mọi fold, để cross-validation không bao giờ huấn luyện trên
            # nguồn của các hỗn hợp dùng cho thí nghiệm cascade.
            reserved = [s in ("test", "unused") for s in splits]
            folds = assign_folds(label_idx, n_folds, rng, reserved=reserved)
            if any(reserved):
                print(
                    f"[fold] giu lai {sum(reserved)} ban ghi (split=test/unused) khoi "
                    f"cross-validation -> fold = {FOLD_RESERVED}"
                )

            grp = h5.create_group(branch)
            grp.attrs["classes"] = json.dumps(branch_classes, ensure_ascii=False)
            grp.attrs["normal_class"] = NORMAL_CLASS
            grp.attrs["n_records"] = len(stems)
            grp.attrs["n_classes"] = len(branch_classes)
            grp.create_dataset("audio", data=audio)
            grp.create_dataset(
                "record_ids", data=np.asarray(stems, dtype=object), dtype=string_dtype
            )
            grp.create_dataset("label_idx", data=label_idx)
            grp.create_dataset("is_abnormal", data=abnormal)
            grp.create_dataset(
                "split", data=np.asarray(splits, dtype=object), dtype=string_dtype
            )
            grp.create_dataset("fold", data=folds)

            _report(branch, stems, branch_classes, label_idx, abnormal, splits, folds)

    mib = out.stat().st_size / (1024 * 1024)
    print(f"\n[done] da ghi {out}  ({mib:.1f} MiB)")


def _report(
    branch: str,
    stems: List[str],
    branch_classes: List[str],
    label_idx: np.ndarray,
    abnormal: np.ndarray,
    splits: List[str],
    folds: np.ndarray,
) -> None:
    """In tóm tắt + chốt chặn rò rỉ."""
    counts = Counter(splits)
    print(
        f"\n[{branch}] tong {len(stems)} ban ghi | "
        f"train={counts.get('train', 0)} val={counts.get('val', 0)} "
        f"test={counts.get('test', 0)} unused={counts.get('unused', 0)}"
    )
    print(f"  {'lop':<24s} {'train':>6s} {'val':>5s} {'test':>5s} {'unus':>5s}   {'tong':>5s}")
    for ci, name in enumerate(branch_classes):
        row = [
            sum(1 for i in range(len(stems)) if label_idx[i] == ci and splits[i] == s)
            for s in ("train", "val", "test", "unused")
        ]
        print(
            f"  {name:<24s} {row[0]:>6d} {row[1]:>5d} {row[2]:>5d} {row[3]:>5d}   {sum(row):>5d}"
        )

    n_abn = int(abnormal.sum())
    print(f"  nhi phan: Normal={len(stems) - n_abn}  Abnormal={n_abn}")
    fold_counts = dict(sorted(Counter(folds.tolist()).items()))
    n_reserved = fold_counts.pop(FOLD_RESERVED, 0)
    print(f"  fold (dung cho cross-validation): {fold_counts}")
    if n_reserved:
        print(
            f"  fold={FOLD_RESERVED}: {n_reserved} ban ghi giu lai cho thi nghiem "
            "cascade, khong tham gia cross-validation"
        )

    # Mỗi bản ghi là một nhóm và chỉ được gán đúng một split; khẳng định lại ở đây
    # để tiêu chí chấp nhận trong prd.md kiểm được bằng chính đầu ra của script.
    per_record = defaultdict(set)
    for stem, split in zip(stems, splits):
        per_record[stem].add(split)
    leaked = sum(1 for v in per_record.values() if len(v) > 1)
    dup = len(stems) - len(set(stems))
    print(f"  [check] ban ghi o nhieu split: {leaked}")
    print(f"  [check] ten ban ghi trung lap: {dup}")
    if leaked or dup:
        raise RuntimeError(f"[{branch}] phat hien ro ri/trung lap ban ghi - dung lai")

    empty = [s for s in ("train", "val", "test") if counts.get(s, 0) == 0]
    if empty:
        print(f"  [canh bao] split rong: {empty} - kiem tra lai --align-split-with")

    # Chỉ có 9 bản ghi Normal (tim) / 12 (phổi) trên tổng 50, nên một split nhỏ
    # rất dễ không nhận được bản ghi Normal nào. Khi đó F1 nhị phân trên split ấy
    # là vô nghĩa (chỉ còn một lớp) - phải báo ngay thay vì để đọc nhầm kết quả.
    for split_name in ("val", "test"):
        members = [i for i in range(len(stems)) if splits[i] == split_name]
        if not members:
            continue
        n_normal = sum(1 for i in members if abnormal[i] == 0)
        if n_normal == 0 or n_normal == len(members):
            only = "Abnormal" if n_normal == 0 else "Normal"
            print(
                f"  [canh bao] split '{split_name}' chi co lop {only} "
                f"({len(members)} ban ghi) -> F1/AUROC nhi phan tren split nay "
                "KHONG dung duoc. Dung cross-validation (cot 'fold') de bao cao."
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Thu muc HLS-CMDS")
    ap.add_argument("--out", default="data_hls_cls.h5", help="File HDF5 dau ra")
    ap.add_argument("--sample-rate", type=int, default=4000)
    ap.add_argument("--target-length", type=int, default=60000, help="So mau moi ban ghi (15s@4k)")
    ap.add_argument(
        "--align-split-with",
        default=None,
        help="data_hls.h5 de ke thua dung phan hoach nguon cua bo tach am "
        "(khuyen nghi: bat buoc neu muon chay danh gia cascade)",
    )
    ap.add_argument("--val-frac", type=float, default=0.15, help="Chi dung khi khong align")
    ap.add_argument("--test-frac", type=float, default=0.10, help="Chi dung khi khong align")
    ap.add_argument(
        "--taxonomy",
        choices=TAXONOMIES,
        default="full",
        help="full: giu nguyen 10 lop tim / 6 lop phoi. "
        "public: chi giu cac lop CO nguon du lieu cong khai de pretrain "
        "(tim: Normal + Early/Mid/Late Systolic Murmur; phoi: Normal + Coarse/Fine "
        "Crackles + Rhonchi + Wheezing), phan con lai gom vao 'Other Abnormal'. "
        "Xem DATASETS.md.",
    )
    ap.add_argument("--n-folds", type=int, default=5, help="So fold cho cross-validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow-unaligned-records",
        action="store_true",
        help="Cho phep ban ghi khong co trong H5 tach am; chung se bi xep vao "
        "'unused' va loai khoi ca train lan eval (mac dinh: dung voi loi)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    build(
        dataset_root=Path(args.dataset_root),
        out=Path(args.out),
        sample_rate=args.sample_rate,
        target_length=args.target_length,
        align_with=Path(args.align_split_with) if args.align_split_with else None,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        n_folds=args.n_folds,
        seed=args.seed,
        allow_unaligned=args.allow_unaligned_records,
        taxonomy=args.taxonomy,
    )


if __name__ == "__main__":
    main()
