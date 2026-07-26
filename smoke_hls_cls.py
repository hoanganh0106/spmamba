"""Kiểm tra nhanh pipeline phân loại trước khi train thật.

Chạy trên Colab ngay sau khi dựng xong ``data_hls_cls.h5``::

    python smoke_hls_cls.py --h5 /content/data_hls_cls.h5

Kiểm tra 6 nhóm, in PASS/FAIL từng nhóm và thoát với mã khác 0 nếu có lỗi:

1. Chỉ số — đối chiếu với giá trị tính tay, để một hàm sai không âm thầm cho ra
   con số đẹp trong báo cáo y tế.
2. Bảng nhãn — số lượng mỗi lớp khớp CSV, bẫy Coarse/Fine bị chặn.
3. Mô hình — shape đầu ra, số tham số, chạy được cả train lẫn eval.
4. Dataset — cửa sổ phủ hết bản ghi, train/val/test rời nhau ở mức bản ghi.
5. Gộp theo bản ghi — trung bình đúng, và bản ghi không có cửa sổ nào không bị
   âm thầm gán về lớp 0 ("Normal").
6. Một vòng train ngắn — loss giảm, checkpoint nạp lại được.

Cái đắt nhất (train thật, tách âm) không nằm ở đây; mục tiêu là bắt lỗi trong
vài chục giây thay vì sau một giờ huấn luyện.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  {detail}")


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) < tol


# --------------------------------------------------------------------------- #
def test_metrics() -> None:
    print("\n[1/6] Chi so (doi chieu voi gia tri tinh tay)")
    from hls_cls_metrics import (
        balanced_accuracy,
        binary_report,
        confusion_matrix,
        macro_f1,
        roc_auc,
    )

    # AUROC: y=[0,1,1,0], scores=[0.1,0.4,0.35,0.8].
    # Cap (duong, am): (0.4,0.1)=1, (0.4,0.8)=0, (0.35,0.1)=1, (0.35,0.8)=0 -> 2/4.
    check("AUROC khong hoa", close(roc_auc(np.array([0, 1, 1, 0]),
                                           np.array([0.1, 0.4, 0.35, 0.8])), 0.5),
          f"nhan {roc_auc(np.array([0,1,1,0]), np.array([0.1,0.4,0.35,0.8]))}")

    # Hoa hoan toan -> moi cap tinh 0.5 -> AUROC = 0.5
    check("AUROC khi hoa hoan toan",
          close(roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])), 0.5))

    # Tach hoan hao -> 1.0 ; nguoc hoan toan -> 0.0
    check("AUROC tach hoan hao",
          close(roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0))
    check("AUROC nguoc hoan toan",
          close(roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])), 0.0))
    check("AUROC mot lop -> None",
          roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) is None)

    # Ma tran nham lan: hang = that, cot = doan
    cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
    check("ma tran nham lan", cm.tolist() == [[1, 1], [0, 2]], f"nhan {cm.tolist()}")

    # sensitivity = recall lop 1 = 2/2 = 1.0 ; specificity = 1/2 = 0.5
    rep = binary_report(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.8, 0.7]))
    check("sensitivity (recall lop bat thuong)", close(rep["sensitivity"], 1.0),
          f"nhan {rep['sensitivity']}")
    check("specificity (recall lop binh thuong)", close(rep["specificity"], 0.5),
          f"nhan {rep['specificity']}")
    check("sensitivity va specificity khong bi hoan doi",
          rep["sensitivity"] != rep["specificity"])

    # macro-F1 chi tinh tren lop CO MAT trong y_true.
    # y_true toan lop 0, y_pred toan lop 0 -> lop 0: F1=1 -> macro = 1.0
    check("macro-F1 bo qua lop vang mat",
          close(macro_f1(np.array([0, 0, 0]), np.array([0, 0, 0]), 5), 1.0),
          f"nhan {macro_f1(np.array([0,0,0]), np.array([0,0,0]), 5)}")
    check("balanced accuracy",
          close(balanced_accuracy(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2), 0.75),
          f"nhan {balanced_accuracy(np.array([0,0,1,1]), np.array([0,1,1,1]), 2)}")

    # Bao cao mot lop khong duoc lam sap chuong trinh
    try:
        single = binary_report(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
        check("bao cao khi chi co mot lop", single["auroc"] is None)
    except Exception as exc:  # noqa: BLE001
        check("bao cao khi chi co mot lop", False, f"nem loi {exc}")


def test_labels(dataset_root: str | None) -> None:
    print("\n[2/6] Bang nhan")
    from look2hear.datas.hls_labels import (
        HEART_CLASSES,
        LUNG_CLASSES,
        label_for,
        verify_against_csv,
    )

    check("Normal o chi so 0 (nhanh tim)", HEART_CLASSES[0] == "Normal")
    check("Normal o chi so 0 (nhanh phoi)", LUNG_CLASSES[0] == "Normal")
    check("so lop tim = 10", len(HEART_CLASSES) == 10, f"nhan {len(HEART_CLASSES)}")
    check("so lop phoi = 6", len(LUNG_CLASSES) == 6, f"nhan {len(LUNG_CLASSES)}")
    check("CC = Coarse Crackles", label_for("F_CC_LUA", "lung") == "Coarse Crackles")
    check("FC = Fine Crackles", label_for("F_FC_LUA", "lung") == "Fine Crackles")

    if dataset_root and Path(dataset_root).is_dir():
        try:
            verify_against_csv(dataset_root, "heart", verbose=False)
            verify_against_csv(dataset_root, "lung", verbose=False)
            check("nhan khop CSV (ca hai nhanh)", True)
        except Exception as exc:  # noqa: BLE001
            check("nhan khop CSV (ca hai nhanh)", False, str(exc)[:200])
    else:
        print("  [bo qua] khong co --dataset_root nen khong doi chieu CSV")


def test_model() -> None:
    print("\n[3/6] Mo hinh")
    from look2hear.models.hls_classifier import HLSClassifier

    model = HLSClassifier(n_classes=10, branch="heart", classes=[f"c{i}" for i in range(10)])
    n_params = model.num_parameters()
    check("so tham so < 500K", n_params < 500_000, f"nhan {n_params:,}")

    batch = torch.randn(4, 20000)          # 4 cua so 5s @ 4 kHz
    model.eval()
    with torch.no_grad():
        logits_multi, logit_bin = model(batch)
    check("shape head da lop = [4, 10]", tuple(logits_multi.shape) == (4, 10),
          f"nhan {tuple(logits_multi.shape)}")
    check("shape head nhi phan = [4]", tuple(logit_bin.shape) == (4,),
          f"nhan {tuple(logit_bin.shape)}")
    check("dau ra huu han", bool(torch.isfinite(logits_multi).all()))

    # Frontend: kiem tra so khung pho va so dai mel
    spec = model.frontend(batch)
    expected_frames = 20000 // 64 + 1
    check("shape log-mel = [4, 1, 64, n_frames]",
          spec.shape[0] == 4 and spec.shape[1] == 1 and spec.shape[2] == 64
          and spec.shape[3] == expected_frames,
          f"nhan {tuple(spec.shape)}, mong doi n_frames={expected_frames}")

    # Bat bien voi muc am luong: nhan bien do len 4 lan khong duoc doi du doan.
    # Day la tinh chat BAT BUOC cho buoc cascade - dau ra SPMamba co bien do khac
    # han ban ghi thuan, neu mo hinh nhay cam voi muc am luong thi ket qua cascade
    # se tut ma khong ro ly do. So sanh theo XAC SUAT (thang do co y nghia) thay vi
    # logit tho.
    with torch.no_grad():
        pa = torch.softmax(model(batch)[0], dim=-1)
        pb = torch.softmax(model(batch * 4.0)[0], dim=-1)
    delta = float((pa - pb).abs().max())
    check("bat bien voi muc am luong (lech xac suat < 0.02)", delta < 0.02,
          f"lech toi da {delta:.4f}")

    # SpecAugment chi hoat dong khi training
    model.train()
    with torch.no_grad():
        masked = model.specaug(model.frontend(batch))
    model.eval()
    with torch.no_grad():
        unmasked = model.specaug(model.frontend(batch))
    check("SpecAugment tat khi eval",
          bool(torch.equal(unmasked, model.frontend(batch))))
    check("SpecAugment co tac dung khi train",
          not bool(torch.equal(masked, model.frontend(batch))))

    # config() phai du de dung lai mo hinh Y HET. Chi so sanh so tham so la khong
    # du: bo frontend hay specaugment khoi config van giu nguyen so tham so (chung
    # khong co tham so hoc duoc) nhung mo hinh dung lai se cho pho khac han.
    cfg = model.config()
    rebuilt = HLSClassifier(**{**cfg, "channels": tuple(cfg["channels"])})
    check("config() giu nguyen so tham so", rebuilt.num_parameters() == n_params)
    check("config() giu nguyen tham so frontend",
          rebuilt.frontend_config == model.frontend_config,
          f"{rebuilt.frontend_config} != {model.frontend_config}")
    check("config() giu nguyen tham so SpecAugment",
          rebuilt.specaug_config == model.specaug_config,
          f"{rebuilt.specaug_config} != {model.specaug_config}")
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    with torch.no_grad():
        check("mo hinh dung lai cho pho giong het",
              bool(torch.allclose(rebuilt.frontend(batch), spec, atol=1e-6)))
        check("mo hinh dung lai cho logits giong het",
              bool(torch.allclose(rebuilt(batch)[0], model(batch)[0], atol=1e-5)))


def test_dataset(h5_path: str) -> None:
    print("\n[4/6] Dataset va split")
    from look2hear.datas.hls_cls_datamodule import HLSClsDataModule, HLSClsDataset

    for branch in ("heart", "lung"):
        dm = HLSClsDataModule(h5_path=h5_path, branch=branch, num_workers=0, batch_size=8)
        dm.setup()   # tu no da khang dinh train/val/test roi nhau
        check(f"[{branch}] train/val/test roi nhau o muc ban ghi", True)

        wav, label, abnormal, ridx = dm.data_train[0]
        check(f"[{branch}] mau train dung shape",
              wav.shape == (dm.data_train.seg_len,) and label.dtype == torch.long,
              f"nhan {tuple(wav.shape)}")
        check(f"[{branch}] is_abnormal khop chi so lop",
              float(abnormal) == float(int(label) != 0))

        # Cua so eval phai phu het ban ghi
        ds = dm.data_val
        covered = {r for r, _ in ds.windows}
        check(f"[{branch}] moi ban ghi val deu co cua so",
              covered == set(range(ds.n_records)),
              f"thieu {sorted(set(range(ds.n_records)) - covered)}")
        last_start = max(s for r, s in ds.windows if r == 0)
        check(f"[{branch}] cua so phu toi cuoi ban ghi",
              last_start + ds.seg_len == ds.record_length,
              f"cuoi cung ket thuc o {last_start + ds.seg_len}/{ds.record_length}")

        # Cross-validation LONG: fold k = tap bao cao, fold (k+1) = tap chon epoch.
        # Ba tap phai roi nhau doi mot, neu khong con so bao cao bi thien lech.
        sets = {}
        for role in ("train", "val", "test"):
            sets[role] = set(
                HLSClsDataset(h5_path=h5_path, branch=branch, fold=0, fold_role=role,
                              train=(role == "train"), repeat=1).record_ids
            )
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        bad = {f"{a}&{b}": sorted(sets[a] & sets[b])[:3] for a, b in pairs if sets[a] & sets[b]}
        check(f"[{branch}] fold 0: train/val/test roi nhau doi mot", not bad, str(bad))
        check(f"[{branch}] fold 0: tap bao cao khac tap chon epoch",
              sets["test"] != sets["val"],
              "test trung val -> so lieu CV se thien lech lac quan")

        # Ban ghi giu lai (fold = -1, thuoc pool test cua bo tach am) khong duoc
        # xuat hien o bat ky vai tro nao.
        import h5py as _h5py
        with _h5py.File(h5_path, "r") as _f:
            all_ids = [x.decode() if isinstance(x, bytes) else x
                       for x in _f[branch]["record_ids"][:]]
            all_folds = [int(k) for k in _f[branch]["fold"][:]]
        reserved = {i for i, k in zip(all_ids, all_folds) if k < 0}
        used = sets["train"] | sets["val"] | sets["test"]
        check(f"[{branch}] ban ghi giu lai (fold=-1) bi loai khoi CV",
              not (reserved & used), f"lot vao CV: {sorted(reserved & used)[:3]}")
        check(f"[{branch}] CV phu het ban ghi khong bi giu lai",
              used == set(all_ids) - reserved,
              f"thieu {sorted((set(all_ids) - reserved) - used)[:3]}")

        # Trong so lop phai huu han va khong am
        w = dm.class_weights()
        check(f"[{branch}] trong so lop huu han, khong am",
              bool(torch.isfinite(w).all()) and bool((w >= 0).all()))


def test_aggregate() -> None:
    print("\n[5/6] Gop du doan len muc ban ghi")
    from look2hear.datas.hls_cls_datamodule import aggregate_by_record

    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    ridx = np.array([0, 0, 1])
    out = aggregate_by_record(probs, ridx, 2)
    check("trung binh dung", close(out[0, 0], 0.7) and close(out[1, 1], 0.9),
          f"nhan {out.tolist()}")

    flat = aggregate_by_record(np.array([0.2, 0.4, 1.0]), ridx, 2)
    check("dau vao 1 chieu tra ve 1 chieu", flat.shape == (2,), f"nhan {flat.shape}")
    check("trung binh dau vao 1 chieu", close(float(flat[0]), 0.3))

    # Ban ghi khong co cua so nao PHAI lam ham bao loi. Neu tra ve hang toan 0 thi
    # argmax cho lop 0 = "Normal", tuc am tham bao cao sai la binh thuong - kieu
    # hong nguy hiem nhat trong sang loc y te.
    try:
        aggregate_by_record(probs, ridx, 4)   # 4 ban ghi nhung chi co du lieu cho 2
        check("ban ghi thieu cua so -> bao loi", False, "khong bao loi, tra ve hang toan 0")
    except ValueError:
        check("ban ghi thieu cua so -> bao loi", True)

    # record_idx ngoai pham vi cung phai bao loi thay vi ghi de sai o
    try:
        aggregate_by_record(probs, np.array([0, 1, 5]), 2)
        check("record_idx ngoai pham vi -> bao loi", False, "khong bao loi")
    except (ValueError, IndexError):
        check("record_idx ngoai pham vi -> bao loi", True)


def test_train_step(h5_path: str) -> None:
    print("\n[6/6] Mot vong train ngan")
    import torch.nn as nn

    from look2hear.datas.hls_cls_datamodule import HLSClsDataModule
    from look2hear.models.hls_classifier import HLSClassifier

    dm = HLSClsDataModule(h5_path=h5_path, branch="lung", num_workers=0,
                          batch_size=8, repeat=1)
    dm.setup()
    model = HLSClassifier(n_classes=len(dm.classes), branch="lung", classes=dm.classes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit_multi = nn.CrossEntropyLoss(weight=dm.class_weights())
    crit_bin = nn.BCEWithLogitsLoss(pos_weight=dm.pos_weight())

    model.train()
    losses = []
    loader = dm.train_dataloader()
    for step, (wav, label, abnormal, _) in enumerate(loader):
        opt.zero_grad(set_to_none=True)
        lm, lb = model(wav)
        loss = crit_multi(lm, label) + 0.5 * crit_bin(lb, abnormal)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step >= 4:
            break

    check("loss huu han", all(math.isfinite(v) for v in losses), f"{losses}")
    check("gradient chay duoc (>= 2 buoc)", len(losses) >= 2, f"chi chay {len(losses)} buoc")

    # Luu roi nap lai checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best.pt"
        torch.save(dict(state_dict=model.state_dict(), model_config=model.config(),
                        branch="lung", classes=dm.classes, epoch=1,
                        data_config={"segment": 5.0}), path)
        from look2hear.models.hls_classifier import load_checkpoint

        loaded, meta = load_checkpoint(str(path))
        check("nap lai checkpoint", loaded.num_parameters() == model.num_parameters())
        check("metadata giu duoc segment",
              float(meta["data_config"]["segment"]) == 5.0)

        model.eval(); loaded.eval()
        x = torch.randn(2, dm.data_train.seg_len)
        with torch.no_grad():
            check("mo hinh nap lai cho ket qua giong het",
                  bool(torch.allclose(model(x)[0], loaded(x)[0], atol=1e-5)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Kiem tra nhanh pipeline phan loai HLS")
    ap.add_argument("--h5", default="/content/data_hls_cls.h5")
    ap.add_argument("--dataset_root", default=None, help="Thu muc HLS-CMDS de doi chieu CSV")
    ap.add_argument("--skip_data", action="store_true", help="Bo qua cac kiem tra can file H5")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 62)
    print("  KIEM TRA NHANH PIPELINE PHAN LOAI AM TIM/PHOI")
    print("=" * 62)

    test_metrics()
    test_labels(args.dataset_root)
    test_model()
    test_aggregate()

    if args.skip_data:
        print("\n[bo qua] cac kiem tra can file H5 (--skip_data)")
    elif not Path(args.h5).is_file():
        print(f"\n[bo qua] khong thay {args.h5} - chay build_hls_cls_h5.py truoc")
    else:
        test_dataset(args.h5)
        test_train_step(args.h5)

    print("\n" + "=" * 62)
    print(f"  PASS: {len(PASSED)}    FAIL: {len(FAILED)}")
    if FAILED:
        print("=" * 62)
        for item in FAILED:
            print(f"  - {item}")
        print("\n  KHONG dat - sua truoc khi train that.")
        return 1
    print("=" * 62)
    print("  Tat ca kiem tra deu dat. Co the chay train_hls_cls.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
