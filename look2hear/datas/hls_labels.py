"""Bảng nhãn cho HLS-CMDS (nhận dạng âm tim/phổi bất thường).

Tên file dạng ``{Gender}_{Abbr}_{Location}.wav`` là **nguồn chân lý** của nhãn,
không phải cột ID trong CSV.

Lý do: ``LS.csv`` đặt mã viết tắt khác với tên file — CSV ghi ``C`` cho Fine
Crackles và ``G`` cho Coarse Crackles, trong khi tên file dùng ``FC`` và ``CC``.
Đối chiếu theo cặp (giới tính, vị trí) xác nhận ``FC``↔``C`` (5 bản ghi) và
``CC``↔``G`` (9 bản ghi), nên tra nhãn theo mã viết tắt của CSV sẽ **đảo ngược
Coarse ↔ Fine**. Phần tim không có vấn đề này (mã CSV và tên file trùng nhau).

CSV vẫn được dùng để kiểm tra chéo: :func:`verify_against_csv` đếm số bản ghi mỗi
lớp theo tên file rồi so với số đếm từ CSV, và dừng nếu lệch.

Module này cố ý chỉ dùng thư viện chuẩn (không numpy/torch) để chạy được cả trên
máy chưa cài môi trường huấn luyện.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


NORMAL_CLASS = "Normal"

# Mã viết tắt trong TÊN FILE -> tên lớp đầy đủ.
HEART_ABBR: Dict[str, str] = {
    "N": "Normal",
    "AF": "Atrial Fibrillation",
    "AVB": "AV Block",
    "T": "Tachycardia",
    "S3": "S3",
    "S4": "S4",
    "ESM": "Early Systolic Murmur",
    "MSM": "Mid Systolic Murmur",
    "LSM": "Late Systolic Murmur",
    "LDM": "Late Diastolic Murmur",
}

LUNG_ABBR: Dict[str, str] = {
    "N": "Normal",
    "CC": "Coarse Crackles",
    "FC": "Fine Crackles",
    "PR": "Pleural Rub",
    "R": "Rhonchi",
    "W": "Wheezing",
}

# Thứ tự lớp cố định: Normal luôn ở chỉ số 0, phần còn lại xếp theo bảng chữ cái.
# Cố định thứ tự để chỉ số lớp trong checkpoint cũ vẫn đúng ở lần chạy sau.
HEART_CLASSES: List[str] = [NORMAL_CLASS] + sorted(
    v for v in HEART_ABBR.values() if v != NORMAL_CLASS
)
LUNG_CLASSES: List[str] = [NORMAL_CLASS] + sorted(
    v for v in LUNG_ABBR.values() if v != NORMAL_CLASS
)

BRANCHES = ("heart", "lung")

# (thư mục WAV, tên file CSV, cột loại âm, cột ID) theo từng nhánh.
_BRANCH_SPEC = {
    "heart": ("HS/HS", "HS.csv", "Heart Sound Type", "Heart Sound ID"),
    "lung": ("LS/LS", "LS.csv", "Lung Sound Type", "Lung Sound ID"),
}


def _check_branch(branch: str) -> str:
    branch = branch.lower()
    if branch not in BRANCHES:
        raise ValueError(f"branch phai la 'heart' hoac 'lung', nhan duoc {branch!r}")
    return branch


def abbr_table(branch: str) -> Dict[str, str]:
    return HEART_ABBR if _check_branch(branch) == "heart" else LUNG_ABBR


def classes(branch: str) -> List[str]:
    return HEART_CLASSES if _check_branch(branch) == "heart" else LUNG_CLASSES


def wav_dir(dataset_root: Path | str, branch: str) -> Path:
    return Path(dataset_root) / _BRANCH_SPEC[_check_branch(branch)][0]


def parse_stem(stem: str) -> Tuple[str, str, str]:
    """``'F_MSM_LUSB'`` -> ``('F', 'MSM', 'LUSB')``."""
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Ten ban ghi {stem!r} khong dung dang {{Gender}}_{{Abbr}}_{{Location}}"
        )
    return parts[0], parts[1], parts[2]


def label_for(stem: str, branch: str) -> str:
    """Trả tên lớp đầy đủ cho một tên bản ghi."""
    table = abbr_table(branch)
    _, abbr, _ = parse_stem(stem)
    if abbr not in table:
        raise ValueError(
            f"Ma viet tat {abbr!r} trong {stem!r} khong co trong bang {branch}. "
            f"Ma hop le: {sorted(table)}"
        )
    return table[abbr]


def is_abnormal(stem: str, branch: str) -> bool:
    return label_for(stem, branch) != NORMAL_CLASS


def list_records(dataset_root: Path | str, branch: str) -> List[str]:
    """Danh sách tên bản ghi (đã sắp xếp) của một nhánh."""
    directory = wav_dir(dataset_root, branch)
    if not directory.is_dir():
        raise FileNotFoundError(f"Khong thay thu muc WAV: {directory}")
    stems = sorted(p.stem for p in directory.glob("*.wav") if p.is_file())
    if not stems:
        raise RuntimeError(f"Khong co file .wav nao trong {directory}")
    return stems


def counts_from_files(dataset_root: Path | str, branch: str) -> Counter:
    return Counter(label_for(s, branch) for s in list_records(dataset_root, branch))


def _read_csv_rows(dataset_root: Path | str, branch: str) -> Tuple[List[dict], str, str]:
    _, csv_name, type_col, id_col = _BRANCH_SPEC[_check_branch(branch)]
    csv_path = Path(dataset_root) / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Khong thay CSV metadata: {csv_path}")
    # utf-8-sig: các CSV này có BOM, đọc bằng utf-8 thường sẽ làm hỏng tên cột đầu.
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} khong co dong nao")
    for col in (type_col, id_col):
        if col not in rows[0]:
            raise ValueError(f"{csv_path} thieu cot {col!r}")
    return rows, type_col, id_col


def counts_from_csv(dataset_root: Path | str, branch: str) -> Counter:
    rows, type_col, _ = _read_csv_rows(dataset_root, branch)
    return Counter(r[type_col].strip() for r in rows)


def _signature_from_files(dataset_root: Path | str, branch: str) -> Dict[str, Counter]:
    """``{tên lớp: Counter{(giới tính, vị trí): số lượng}}`` suy từ tên file."""
    sig: Dict[str, Counter] = {}
    for stem in list_records(dataset_root, branch):
        gender, _, location = parse_stem(stem)
        sig.setdefault(label_for(stem, branch), Counter())[(gender, location)] += 1
    return sig


def _signature_from_csv(dataset_root: Path | str, branch: str) -> Dict[str, Counter]:
    """Cùng dạng như trên nhưng lấy từ CSV.

    Chỉ dùng phần giới tính và vị trí của cột ID — hai phần này trùng với tên file
    ở cả hai nhánh. Phần viết tắt ở giữa thì KHÔNG dùng, vì LS.csv đặt mã khác
    (``C``/``G`` thay cho ``FC``/``CC``).
    """
    rows, type_col, id_col = _read_csv_rows(dataset_root, branch)
    sig: Dict[str, Counter] = {}
    for row in rows:
        gender, _, location = parse_stem(row[id_col].strip())
        sig.setdefault(row[type_col].strip(), Counter())[(gender, location)] += 1
    return sig


def verify_against_csv(dataset_root: Path | str, branch: str, verbose: bool = True) -> Counter:
    """Đối chiếu nhãn suy từ tên file với metadata trong CSV.

    Chốt chặn cho lỗi ghép nhãn kiểu Coarse↔Fine. Phép so **không** dựa vào số
    lượng mỗi lớp, mà dựa vào "dấu vân tay" của từng lớp: đa tập các cặp
    ``(giới tính, vị trí)``.

    Lý do phải mạnh hơn phép đếm: nhiều lớp có cùng số bản ghi, nên hoán đổi hai
    lớp trong bảng viết tắt sẽ lọt qua phép đếm. Nhánh phổi có Pleural Rub và
    Coarse Crackles cùng 9 bản ghi; nhánh tim có ba cặp như vậy (Late Diastolic
    Murmur / Early Systolic Murmur = 6, Late Systolic Murmur / S3 = 5,
    Tachycardia / AV Block = 3). Trong khi đó không có hai lớp nào trùng dấu vân
    tay ``(giới tính, vị trí)``, nên phép so này bắt được mọi hoán đổi.
    """
    branch = _check_branch(branch)
    from_files = counts_from_files(dataset_root, branch)
    sig_files = _signature_from_files(dataset_root, branch)
    sig_csv = _signature_from_csv(dataset_root, branch)

    known = set(classes(branch))
    unknown_csv = sorted(set(sig_csv) - known)
    if unknown_csv:
        raise ValueError(
            f"[{branch}] CSV co loai am khong nam trong bang nhan: {unknown_csv}. "
            f"Bang hien tai: {sorted(known)}"
        )

    mismatched = sorted(
        name for name in set(sig_files) | set(sig_csv)
        if sig_files.get(name, Counter()) != sig_csv.get(name, Counter())
    )
    if mismatched:
        lines = []
        for name in mismatched:
            a = sig_files.get(name, Counter())
            b = sig_csv.get(name, Counter())
            only_file = sorted((a - b).elements())
            only_csv = sorted((b - a).elements())
            lines.append(
                f"    {name}: ten_file={sum(a.values())} ban ghi, CSV={sum(b.values())}\n"
                f"       chi co o ten file: {only_file}\n"
                f"       chi co o CSV     : {only_csv}"
            )
        raise ValueError(
            f"[{branch}] Nhan suy tu ten file khong khop CSV:\n" + "\n".join(lines) + "\n"
            "  Rat co the bang viet tat trong hls_labels.py da sai (vi du dao "
            "Coarse Crackles voi Fine Crackles). Sua bang truoc khi huan luyen."
        )

    # Nếu có hai lớp trùng dấu vân tay thì phép kiểm trên mất khả năng phân biệt
    # đúng cặp lớp đó - báo để không tưởng nhầm là đã kiểm hết.
    names = sorted(sig_files)
    ambiguous = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if sig_files[a] == sig_files[b]
    ]
    if ambiguous:
        print(
            f"[nhan][canh bao] {branch}: cac cap lop co dau van tay giong het nhau "
            f"{ambiguous} - hoan doi giua chung se KHONG bi phat hien."
        )

    if verbose:
        total = sum(from_files.values())
        print(
            f"[nhan] {branch}: {total} ban ghi, {len(from_files)} lop - khop CSV "
            "(doi chieu theo gioi tinh + vi tri, khong chi theo so luong)"
        )
        for name in classes(branch):
            n = from_files.get(name, 0)
            flag = "" if name == NORMAL_CLASS else "  (bat thuong)"
            print(f"        {name:<24s} {n:>3d}{flag}")
    return from_files
