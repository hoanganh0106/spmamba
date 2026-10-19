"""Dataset/DataModule cho bài toán phân loại âm tim/phổi HLS-CMDS.

Đọc file do ``build_hls_cls_h5.py`` sinh ra (xem schema trong docstring của script
đó). Mỗi bản ghi dài 15 s được cắt thành cửa sổ ngắn:

* **train** — mỗi lần lấy mẫu cắt ngẫu nhiên một cửa sổ, kèm tăng cường dữ liệu.
  ``__len__`` = ``số bản ghi × repeat`` để một epoch nhìn đủ nhiều biến thể.
* **eval** — cửa sổ trượt tất định (hop = segment/2), giữ ``record_idx`` để gộp
  dự đoán lên mức bản ghi.

Ranh giới train/eval luôn nằm ở **mức bản ghi**, không phải mức cửa sổ: chỉ có 50
bản ghi độc lập mỗi nhánh, nếu các cửa sổ của cùng một bản ghi rơi vào cả hai bên
thì mô hình chỉ cần nhớ bản ghi là đạt accuracy rất cao mà không học được gì.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HLSClsDataset(Dataset):
    """Cửa sổ audio + nhãn cho một nhánh (``heart`` hoặc ``lung``).

    Trả về ``(wav, label_idx, is_abnormal, record_idx)``:

    * ``wav``        — float32 ``[seg_len]``
    * ``label_idx``  — int64, chỉ số lớp bệnh lý
    * ``is_abnormal``— float32 0.0/1.0, nhãn cho head nhị phân
    * ``record_idx`` — int64, chỉ số bản ghi **trong dataset này**, dùng để gộp
      dự đoán cửa sổ lên mức bản ghi lúc đánh giá

    Tham số chọn tập con — dùng đúng một trong hai:

    * ``split``      — "train" / "val" / "test" theo cột ``/{branch}/split``
    * ``fold`` + ``fold_role`` — cross-validation **lồng** theo cột
      ``/{branch}/fold``, với ``fold_role`` là ``"test"`` / ``"val"`` / ``"train"``:

      - ``test``  = fold ``k``           → tập BÁO CÁO, không bao giờ dùng để chọn
      - ``val``   = fold ``(k+1) % K``   → chọn epoch tốt nhất và dừng sớm
      - ``train`` = các fold còn lại

      Bản ghi có ``fold < 0`` bị loại khỏi cả ba vai trò: đó là các bản ghi thuộc
      pool test của bộ tách âm, giữ lại riêng cho thí nghiệm cascade.
    """

    def __init__(
        self,
        h5_path: str,
        branch: str,
        split: Optional[str] = None,
        fold: Optional[int] = None,
        fold_role: str = "train",
        segment: float = 5.0,
        train: bool = False,
        repeat: int = 8,
        hop_ratio: float = 0.5,
        augment: bool = True,
        gain_db: float = 6.0,
        noise_snr_db: Tuple[float, float] = (20.0, 40.0),
        seed: int = 0,
    ):
        super().__init__()
        if (split is None) == (fold is None):
            raise ValueError("Chi dinh dung MOT trong hai: split hoac fold")
        if fold is not None and fold_role not in ("train", "val", "test"):
            raise ValueError("fold_role phai la 'train', 'val' hoac 'test'")

        self.h5_path = h5_path
        self.branch = branch
        self.train = train
        self.augment = augment and train
        self.gain_db = gain_db
        self.noise_snr_db = noise_snr_db
        self.repeat = max(1, repeat) if train else 1

        with h5py.File(h5_path, "r") as f:
            if branch not in f:
                raise KeyError(f"{h5_path} khong co nhanh '{branch}' (co: {list(f.keys())})")
            grp = f[branch]
            self.sample_rate = int(f.attrs["sample_rate"])
            self.classes: List[str] = json.loads(grp.attrs["classes"])
            self.normal_class: str = str(grp.attrs["normal_class"])

            all_splits = [
                s.decode("utf-8") if isinstance(s, bytes) else s for s in grp["split"][:]
            ]
            all_folds = [int(k) for k in grp["fold"][:]]
            n_folds = int(f.attrs.get("n_folds", 5))
            self.n_folds = n_folds
            self.split_source = str(f.attrs.get("split_source", "unknown"))
            self.taxonomy = str(f.attrs.get("taxonomy", "full"))
            if split is not None:
                keep = [i for i, s in enumerate(all_splits) if s == split]
                self.subset_desc = f"split={split}"
            else:
                # Cross-validation LỒNG. Nếu dùng chính fold k vừa để dừng sớm/chọn
                # checkpoint vừa để báo cáo thì con số báo cáo bị thiên lệch lạc
                # quan: ta đã chọn ra epoch trông đẹp nhất trên đúng tập đó.
                inner = (fold + 1) % n_folds
                if fold_role == "test":
                    keep = [i for i, k in enumerate(all_folds) if k == fold]
                elif fold_role == "val":
                    keep = [i for i, k in enumerate(all_folds) if k == inner]
                else:
                    keep = [
                        i for i, k in enumerate(all_folds)
                        if k >= 0 and k != fold and k != inner
                    ]
                self.subset_desc = f"fold={fold}/{fold_role}"

            if not keep:
                raise RuntimeError(
                    f"[{branch}] tap con rong voi {self.subset_desc}. "
                    "Kiem tra lai file H5 hoac tham so chon tap con."
                )

            # Toàn bộ đường đánh giá coi chỉ số 0 là "Normal" (nhãn nhị phân, điểm
            # anomaly, cột đầu ma trận nhầm lẫn). Khẳng định ngay tại chỗ đọc dữ
            # liệu, nếu không một file H5 dựng bằng bảng lớp khác sẽ đảo ngược
            # toàn bộ kết quả nhị phân mà không có dấu hiệu nào.
            if self.classes[0] != self.normal_class:
                raise ValueError(
                    f"[{branch}] classes[0]={self.classes[0]!r} nhung normal_class="
                    f"{self.normal_class!r}. Ca pipeline gia dinh chi so 0 la lop binh "
                    "thuong; dung lai thay vi bao cao nguoc."
                )

            self.audio = grp["audio"][keep].astype(np.float32)
            self.label_idx = grp["label_idx"][keep].astype(np.int64)
            self.is_abnormal = grp["is_abnormal"][keep].astype(np.float32)
            ids = grp["record_ids"][keep]
            self.record_ids = [x.decode("utf-8") if isinstance(x, bytes) else x for x in ids]

        self.n_records, self.record_length = self.audio.shape
        self.seg_len = int(round(segment * self.sample_rate))
        if self.seg_len > self.record_length:
            raise ValueError(
                f"segment={segment}s ({self.seg_len} mau) dai hon ban ghi "
                f"({self.record_length} mau)"
            )

        # Chỉ mục cửa sổ tất định cho chế độ đánh giá.
        self.windows: List[Tuple[int, int]] = []
        if not train:
            hop = max(1, int(round(self.seg_len * hop_ratio)))
            for r in range(self.n_records):
                starts = list(range(0, self.record_length - self.seg_len + 1, hop))
                if not starts:
                    starts = [0]
                # Bảo đảm phủ tới cuối bản ghi.
                last = self.record_length - self.seg_len
                if starts[-1] != last:
                    starts.append(last)
                self.windows.extend((r, s) for s in starts)

        # RNG được tạo lười ở lần lấy mẫu đầu tiên, xem _get_rng().
        self._base_seed = seed
        self._rng: Optional[np.random.Generator] = None

        n_items = len(self)
        print(
            f"[HLSClsDataset] {branch} {self.subset_desc}: {self.n_records} ban ghi "
            f"-> {n_items} cua so ({'train' if train else 'eval'}), "
            f"segment={segment}s, augment={self.augment}"
        )

    # -- thống kê phục vụ hàm mất mát ------------------------------------- #
    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for lab in self.label_idx:
            counts[int(lab)] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """``n_total / (n_classes * n_lop)``; lớp vắng mặt nhận trọng số 0."""
        counts = self.class_counts()
        total = counts.sum()
        weights = np.zeros(len(counts), dtype=np.float32)
        present = counts > 0
        weights[present] = total / (present.sum() * counts[present])
        return torch.from_numpy(weights)

    def pos_weight(self) -> torch.Tensor:
        """Tỉ lệ Normal/Abnormal cho ``BCEWithLogitsLoss``."""
        n_pos = float(self.is_abnormal.sum())
        n_neg = float(len(self.is_abnormal) - n_pos)
        return torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32)

    # -- tăng cường dữ liệu ------------------------------------------------ #
    def _get_rng(self) -> np.random.Generator:
        """RNG riêng cho từng DataLoader worker và từng epoch.

        Tạo RNG trong ``__init__`` là sai: mỗi worker nhận một bản sao dataset với
        cùng hạt giống, nên tất cả worker sẽ cắt cùng vị trí và tăng cường giống
        hệt nhau, và lặp lại y nguyên ở mọi epoch. ``worker_info.seed`` được
        PyTorch đổi theo từng epoch và từng worker, nên trộn nó vào là đủ để có
        đa dạng thật mà vẫn tái lập được khi cố định hạt giống toàn cục.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = self._base_seed if info is None else (self._base_seed + info.seed) % (2**32)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _apply_augment(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # KHÔNG dịch vòng (np.roll) cửa sổ ở đây. Dịch vòng một đoạn đã cắt sẽ dán
        # cuối đoạn vào đầu đoạn, tạo một bước nhảy biên độ tại chỗ nối - tức một
        # tiếng "click" băng thông rộng nhân tạo. Với bài toán này thì đó là tăng
        # cường phản tác dụng: tiếng click chính là đặc trưng của Fine/Coarse
        # Crackles, nên dạy mô hình coi click là chuyện bình thường sẽ làm giảm
        # đúng cái khả năng ta cần. Tính bất biến theo thời gian đã có sẵn nhờ cắt
        # ngẫu nhiên vị trí cửa sổ trong bản ghi 15 s ở __getitem__.

        # Đổi biên độ. Lưu ý: frontend chuẩn hoá log-mel theo từng mẫu nên phần
        # lớn tác dụng của phép này bị triệt tiêu (đó là chủ ý - xem tính bất biến
        # âm lượng trong hls_classifier.py). Giữ lại vì nó vẫn đổi vị trí tương
        # đối của tín hiệu so với sàn ``log(mel + eps)``, tức thay đổi nhẹ hình
        # dạng phổ ở các dải năng lượng thấp.
        wav = wav * float(10.0 ** (rng.uniform(-self.gain_db, self.gain_db) / 20.0))

        # Nhiễu Gauss ở SNR ngẫu nhiên. Đây là phép tăng cường có tác dụng thật
        # trong nhóm này: nó đổi tỉ lệ tín hiệu/nhiễu nên đổi hẳn hình dạng phổ.
        rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)
        snr = float(rng.uniform(*self.noise_snr_db))
        wav = wav + rng.normal(0.0, rms / (10.0 ** (snr / 20.0)), size=wav.shape).astype(np.float32)

        # KHÔNG đảo dấu. Đảo cực ống nghe không đổi nhãn, nhưng frontend dùng phổ
        # công suất ``|STFT(x)|²`` mà ``|STFT(-x)|² = |STFT(x)|²`` đúng bằng nhau,
        # nên phép này chứng minh được là không đổi một bit nào ở đầu vào mô hình.
        # Giữ lại chỉ tạo ảo giác đã tăng cường dữ liệu.
        return wav.astype(np.float32, copy=False)

    # -- giao diện Dataset -------------------------------------------------- #
    def __len__(self) -> int:
        return self.n_records * self.repeat if self.train else len(self.windows)

    def __getitem__(self, idx: int):
        if self.train:
            rng = self._get_rng()
            r = idx % self.n_records
            start = int(rng.integers(0, self.record_length - self.seg_len + 1))
        else:
            rng = None
            r, start = self.windows[idx]

        wav = self.audio[r, start : start + self.seg_len].copy()
        if self.augment:
            wav = self._apply_augment(wav, rng)

        return (
            torch.from_numpy(wav),
            torch.tensor(int(self.label_idx[r]), dtype=torch.long),
            torch.tensor(float(self.is_abnormal[r]), dtype=torch.float32),
            torch.tensor(r, dtype=torch.long),
        )


def aggregate_by_record(
    probs: np.ndarray, record_idx: np.ndarray, n_records: int
) -> np.ndarray:
    """Gộp xác suất mức cửa sổ lên mức bản ghi bằng trung bình.

    Trung bình xác suất ổn định hơn bỏ phiếu đa số khi mỗi bản ghi chỉ có vài
    cửa sổ, và giữ được thông tin độ tự tin để tính AUROC.

    Bản ghi không nhận được cửa sổ nào sẽ làm hàm này **báo lỗi** thay vì trả về
    hàng toàn 0. Hàng toàn 0 đi qua ``argmax`` sẽ cho lớp 0, tức ``"Normal"``:
    một bản ghi bị mất dữ liệu sẽ âm thầm được báo là bình thường. Với công cụ
    sàng lọc y tế thì đó là kiểu hỏng nguy hiểm nhất, nên phải dừng và báo.
    """
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs[:, None]
    record_idx = np.asarray(record_idx).astype(int)
    if len(record_idx) != len(probs):
        raise ValueError(
            f"So cua so khong khop: probs={len(probs)}, record_idx={len(record_idx)}"
        )
    if len(record_idx) and (record_idx.min() < 0 or record_idx.max() >= n_records):
        raise ValueError(
            f"record_idx ngoai pham vi [0, {n_records - 1}]: "
            f"min={record_idx.min()}, max={record_idx.max()}"
        )

    out = np.zeros((n_records, probs.shape[1]), dtype=np.float64)
    counts = np.zeros(n_records, dtype=np.int64)
    for p, r in zip(probs, record_idx):
        out[r] += p
        counts[r] += 1

    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(
            f"{empty.size}/{n_records} ban ghi khong co cua so nao "
            f"(chi so: {empty[:10].tolist()}). Neu tra ve hang toan 0 thi argmax se "
            "cho lop 0 = 'Normal', tuc bao cao SAI la binh thuong. Kiem tra lai "
            "cach sinh cua so trong HLSClsDataset."
        )

    out /= counts[:, None]
    return np.squeeze(out, axis=1) if out.shape[1] == 1 else out


class HLSClsDataModule:
    """Gom 3 dataloader + thống kê lớp cho một nhánh.

    Dùng ``fold=None`` cho chế độ split cố định (train/val/test đọc từ file), hoặc
    ``fold=k`` cho cross-validation (val = fold k, train = phần còn lại; lúc này
    test dùng lại chính val của fold đó).
    """

    def __init__(
        self,
        h5_path: str,
        branch: str,
        segment: float = 5.0,
        repeat: int = 8,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
        augment: bool = True,
        fold: Optional[int] = None,
        seed: int = 0,
        **kwargs,
    ):
        # Không nuốt khoá lạ: cả khối `data:` trong YAML được truyền thẳng vào đây,
        # nên một khoá gõ sai (hay một tham số tưởng là có thật) sẽ bị bỏ qua âm
        # thầm và người dùng tin rằng mình đã bật một thứ chưa hề được bật.
        if kwargs:
            raise TypeError(
                f"HLSClsDataModule: khoa cau hinh khong nhan ra: {sorted(kwargs)}. "
                f"Cac khoa hop le: augment, batch_size, branch, fold, h5_path, "
                f"num_workers, pin_memory, repeat, seed, segment."
            )

        self.h5_path = h5_path
        self.branch = branch
        self.segment = segment
        self.repeat = repeat
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment = augment
        self.fold = fold
        self.seed = seed

        self.data_train: Optional[HLSClsDataset] = None
        self.data_val: Optional[HLSClsDataset] = None
        self.data_test: Optional[HLSClsDataset] = None

    def setup(self) -> None:
        common = dict(
            h5_path=self.h5_path,
            branch=self.branch,
            segment=self.segment,
            seed=self.seed,
        )
        if self.fold is None:
            self.data_train = HLSClsDataset(
                **common, split="train", train=True, repeat=self.repeat, augment=self.augment
            )
            self.data_val = HLSClsDataset(**common, split="val", train=False)
            self.data_test = HLSClsDataset(**common, split="test", train=False)
        else:
            self.data_train = HLSClsDataset(
                **common,
                fold=self.fold,
                fold_role="train",
                train=True,
                repeat=self.repeat,
                augment=self.augment,
            )
            self.data_val = HLSClsDataset(**common, fold=self.fold, fold_role="val", train=False)
            self.data_test = HLSClsDataset(**common, fold=self.fold, fold_role="test", train=False)

        self._assert_disjoint()

    @property
    def selection_set_is_eval_set(self) -> bool:
        """True nếu tập dùng chọn checkpoint trùng tập đem báo cáo.

        Phải luôn False. Giữ lại như một cờ để ghi vào ``summary.json``, nhờ đó
        người đọc kết quả biết được con số có bị thiên lệch do chọn epoch hay
        không, thay vì phải đọc lại mã nguồn.
        """
        return self.data_test is self.data_val or (
            set(self.data_test.record_ids) == set(self.data_val.record_ids)
        )

    def _assert_disjoint(self) -> None:
        """Chốt chặn: tập bản ghi train không được giao với val/test."""
        train_ids = set(self.data_train.record_ids)
        val_ids = set(self.data_val.record_ids)
        test_ids = set(self.data_test.record_ids)
        overlap_vt = val_ids & test_ids
        if overlap_vt:
            raise RuntimeError(
                f"[{self.branch}] {len(overlap_vt)} ban ghi vua o val vua o test: "
                f"{sorted(overlap_vt)[:5]}. Tap chon checkpoint va tap bao cao phai "
                "roi nhau, neu khong con so bao cao se bi thien lech lac quan."
            )
        for name, ds in (("val", self.data_val), ("test", self.data_test)):
            overlap = train_ids & set(ds.record_ids)
            if overlap:
                raise RuntimeError(
                    f"[{self.branch}] {len(overlap)} ban ghi vua o train vua o {name}: "
                    f"{sorted(overlap)[:5]}. Day la ro ri du lieu - dung lai."
                )
        print(
            f"[HLSClsDataModule] {self.branch}: train/val/test roi nhau o muc ban ghi "
            f"({len(train_ids)}/{len(self.data_val.record_ids)}/"
            f"{len(self.data_test.record_ids)} ban ghi)"
        )

    @property
    def classes(self) -> List[str]:
        return self.data_train.classes

    @property
    def n_folds(self) -> int:
        return self.data_train.n_folds

    @property
    def split_source(self) -> str:
        return self.data_train.split_source

    @property
    def taxonomy(self) -> str:
        return self.data_train.taxonomy

    def class_weights(self) -> torch.Tensor:
        return self.data_train.class_weights()

    def pos_weight(self) -> torch.Tensor:
        return self.data_train.pos_weight()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def _eval_loader(self, dataset: HLSClsDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return self._eval_loader(self.data_val)

    def test_dataloader(self) -> DataLoader:
        return self._eval_loader(self.data_test)
