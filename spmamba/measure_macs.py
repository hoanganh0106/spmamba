#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
measure_macs.py
===============
Đo MACs cho SPMamba MỘT CÁCH TRUNG THỰC, có bù phần Mamba mà các tool đếm FLOPs
(ptflops / thop / fvcore) thường BỎ SÓT.

VÌ SAO BỊ SÓT?
--------------
mamba_ssm.modules.mamba_simple.Mamba dùng "fast path": khi inference_params is None
và đang chạy trên CUDA, nó gọi 1 kernel hợp nhất `mamba_inner_fn(...)` thực hiện
in_proj + conv1d + selective_scan + out_proj TRONG MỘT kernel CUDA tuỳ biến.
=> các submodule nn.Linear / nn.Conv1d KHÔNG được gọi qua .__call__,
   nên forward-hook của ptflops/thop/fvcore KHÔNG kích hoạt => báo 0 MACs cho Mamba.
Đây chính là lý do `cal_flops.py` (bản gốc) phải CỘNG TAY selective_scan + in_proj + dt_proj.

Script này:
  (A) In kích thước (batch, seq_len, d_model) thực tế đi vào TỪNG lớp Mamba.
  (B) Chạy ptflops (tái lập con số hiện tại) + bảng per-module + chứng minh Mamba=0.
  (C) Tính lại MACs Mamba bằng giải tích (đọc shape thật của in_proj/conv1d/x_proj/
      dt_proj/out_proj + công thức selective_scan) => tổng "đã bù".
  (D) thop (profile) — đối chiếu (đơn vị: MAC).
  (E) fvcore (FlopCountAnalysis) — in danh sách "unsupported / skipped ops"
      (chỗ lộ ra scan/einsum/cumsum bị bỏ qua). (đơn vị fvcore: fma = 1 => xấp xỉ MAC).
  (F) Ước lượng tay riêng phần selective_scan.
  (G) Bảng tổng hợp.

QUY ƯỚC ĐƠN VỊ
--------------
  - ptflops  -> MACs (1 MAC = 1 mul + 1 add).
  - thop     -> "FLOPs" nhưng thực chất đếm MACs (fma = 1). => coi là MAC.
  - fvcore   -> "flops" nhưng đếm 1 fma = 1 flop => xấp xỉ MAC.
  => CẢ BA cùng đơn vị MAC-tương-đương, so sánh trực tiếp được.
  - selective_scan: theo công thức chuẩn 9*B*L*D*N (mamba issue #110, cũng là cách
    SPMamba/VMamba dùng). Đây là số "op" thô; ta cộng vào tổng MAC y như cal_flops.py
    để SO SÁNH ĐƯỢC với 238.69 G của bài gốc. (Phần này chỉ ~1/4 chi phí Mamba,
    phần lớn còn lại là in_proj/out_proj là MAC thật, không nhập nhằng.)

CHẠY (Colab, 1 GPU):
  python measure_macs.py --conf_dir configs/spmamba-3speakers.yml
  python measure_macs.py --conf_dir configs/spmamba-3speakers.yml --seconds 1.0
  # so sánh với cấu hình gốc (sửa H/emb_ks/n_srcs trong 1 file yml riêng rồi trỏ vào)
"""
import os
import sys
import argparse
import math
import yaml
import torch
import torch.nn as nn

import look2hear.models

# ----------------------------------------------------------------------------- #
# tiện ích
# ----------------------------------------------------------------------------- #
def count_params_M(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def get_mamba_class():
    from mamba_ssm.modules.mamba_simple import Mamba
    return Mamba


def fmt(g):
    return f"{g:10.4f} G"


# ----------------------------------------------------------------------------- #
# Giải tích MACs cho MỘT module Mamba — đọc SHAPE THẬT của các trọng số.
# tokens = batch * L  (số token đưa vào scan)
# ----------------------------------------------------------------------------- #
def mamba_macs_breakdown(m, batch, L):
    """Trả về dict {op: macs} cho 1 lần gọi Mamba với (batch, L)."""
    tokens = batch * L
    d_inner = m.d_inner
    d_state = m.d_state
    d_conv = m.d_conv

    # mỗi Linear: MAC/token = out_features * in_features
    in_proj = tokens * m.in_proj.weight.shape[0] * m.in_proj.weight.shape[1]
    x_proj = tokens * m.x_proj.weight.shape[0] * m.x_proj.weight.shape[1]
    dt_proj = tokens * m.dt_proj.weight.shape[0] * m.dt_proj.weight.shape[1]
    out_proj = tokens * m.out_proj.weight.shape[0] * m.out_proj.weight.shape[1]
    # conv1d depthwise (groups = d_inner): MAC/token = d_inner * d_conv
    conv1d = tokens * d_inner * d_conv
    # selective_scan core (mamba issue #110): 9*B*L*D*N + skip-D (B*L*D) + z-gate (B*L*D)
    scan = 9 * tokens * d_inner * d_state + 2 * tokens * d_inner
    return {
        "in_proj": in_proj,
        "conv1d": conv1d,
        "x_proj": x_proj,
        "dt_proj": dt_proj,
        "selective_scan": scan,
        "out_proj": out_proj,
        "_total": in_proj + conv1d + x_proj + dt_proj + scan + out_proj,
        "_scan_only": scan,
    }


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_dir", required=True, help="đường dẫn file yml cấu hình")
    ap.add_argument("--seconds", type=float, default=1.0, help="độ dài input (giây)")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!!! CẢNH BÁO: không có CUDA. selective_scan của Mamba cần CUDA; "
              "ptflops vẫn chạy nhưng forward Mamba có thể lỗi/không đại diện.")

    # --- build model từ yml ---
    with open(args.conf_dir) as f:
        conf = yaml.safe_load(f)
    net_conf = conf["audionet"]["audionet_config"]
    sr = conf.get("datamodule", {}).get("data_config", {}).get("sample_rate", args.sr)
    Model = getattr(look2hear.models, conf["audionet"]["audionet_name"])
    model = Model(sample_rate=sr, **net_conf).to(device).eval()

    n_samples = int(round(args.sr * args.seconds))
    print("=" * 78)
    print("CẤU HÌNH ĐO")
    print("=" * 78)
    for k, v in net_conf.items():
        print(f"  {k:22s}: {v}")
    print(f"  input shape           : [1, {n_samples}]  ({args.seconds}s @ {args.sr} Hz, mono)")
    print(f"  device                : {device}")
    print(f"  Params                : {count_params_M(model):.4f} M")

    Mamba = get_mamba_class()

    # ---- QUAN TRỌNG: bỏ gradient-checkpointing để hook chạy & đếm đúng 1 lần ----
    import torch.utils.checkpoint as _ckpt
    _orig_ckpt = _ckpt.checkpoint

    def _passthrough(fn, *a, **k):
        # bỏ các kwargs như use_reentrant; gọi thẳng block(tensor)
        tens = [x for x in a if torch.is_tensor(x)]
        return fn(tens[0]) if tens else fn(*a[:1])
    _ckpt.checkpoint = _passthrough

    # =========================================================================
    # (A) Bắt kích thước (batch, L, d_model) đi vào TỪNG Mamba
    # =========================================================================
    print("\n" + "=" * 78)
    print("(A) SEQ_LEN THỰC TẾ ĐI VÀO TỪNG LỚP MAMBA")
    print("=" * 78)
    shapes = []  # list of (name, module, batch, L)
    handles = []

    def make_hook(name):
        def hook(module, inp):
            x = inp[0]
            b, L, dm = x.shape
            shapes.append((name, module, b, L))
            print(f"  {name:46s} batch={b:5d}  L={L:5d}  d_model={dm:4d}  "
                  f"(d_inner={module.d_inner}, d_state={module.d_state})")
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, Mamba):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))

    x = torch.randn(1, n_samples, device=device)
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    print(f"\n  Tổng số lần gọi Mamba: {len(shapes)}  "
          f"(= n_layers * 2(intra/inter) * 2(bi-directional))")

    # =========================================================================
    # (B) ptflops — tái lập con số hiện tại + bảng per-module
    # =========================================================================
    print("\n" + "=" * 78)
    print("(B) PTFLOPS (bảng per-module ở dưới; chú ý Mamba sẽ ~0)")
    print("=" * 78)
    ptflops_macs = None
    try:
        from ptflops import get_model_complexity_info
        with torch.no_grad():
            macs, params = get_model_complexity_info(
                model, (n_samples,), as_strings=False,
                print_per_layer_stat=True, verbose=False,
            )
        ptflops_macs = macs
        print(f"\n  >>> ptflops TỔNG: {macs/1e9:.4f} G MACs | Params: {params/1e6:.4f} M")
    except Exception as e:
        print(f"  ptflops lỗi: {e}")

    # Chứng minh Mamba báo 0: chạy ptflops trên 1 lớp Mamba đơn lẻ
    print("\n  --- Kiểm chứng: ptflops trên MỘT lớp Mamba đơn lẻ ---")
    try:
        from ptflops import get_model_complexity_info
        name0, m0, b0, L0 = shapes[0]
        probe = m0
        def ctor(res):
            return {"hidden_states": torch.randn(b0, L0, m0.d_model, device=device)}
        with torch.no_grad():
            mm, _ = get_model_complexity_info(
                probe, (1,), input_constructor=ctor,
                as_strings=False, print_per_layer_stat=False, verbose=False,
            )
        ana = mamba_macs_breakdown(m0, b0, L0)["_total"]
        print(f"  Mamba '{name0}' input=({b0},{L0},{m0.d_model}):")
        print(f"     ptflops báo : {mm/1e9:.6f} G  <-- gần 0 => BỊ SÓT")
        print(f"     giải tích   : {ana/1e9:.6f} G  <-- giá trị thật")
    except Exception as e:
        print(f"  (probe lỗi: {e})")

    # =========================================================================
    # (C) MACs Mamba bằng giải tích -> tổng "đã bù"
    # =========================================================================
    print("\n" + "=" * 78)
    print("(C) BÙ MAMBA BẰNG GIẢI TÍCH (đọc shape thật + công thức scan)")
    print("=" * 78)
    agg = {"in_proj": 0, "conv1d": 0, "x_proj": 0, "dt_proj": 0,
           "selective_scan": 0, "out_proj": 0, "_total": 0, "_scan_only": 0}
    for name, mod, b, L in shapes:
        bd = mamba_macs_breakdown(mod, b, L)
        for k in agg:
            agg[k] += bd[k]
    print("  Tổng MACs theo từng phép trong Mamba (cộng dồn 24 lần gọi):")
    for k in ["in_proj", "conv1d", "x_proj", "dt_proj", "selective_scan", "out_proj"]:
        print(f"     {k:16s}: {agg[k]/1e9:9.4f} G")
    print(f"     {'TỔNG Mamba':16s}: {agg['_total']/1e9:9.4f} G "
          f"(trong đó scan = {agg['_scan_only']/1e9:.4f} G)")

    if ptflops_macs is not None:
        print(f"\n  ptflops (đếm được, không Mamba)      : {ptflops_macs/1e9:9.4f} G")
        print(f"  + scan-only                          : {(ptflops_macs+agg['_scan_only'])/1e9:9.4f} G  (vẫn thiếu in/out_proj!)")
        print(f"  + TOÀN BỘ Mamba (CON SỐ ĐÚNG)        : {(ptflops_macs+agg['_total'])/1e9:9.4f} G")

    # =========================================================================
    # (D) thop
    # =========================================================================
    print("\n" + "=" * 78)
    print("(D) THOP (đơn vị: MAC). Dùng custom_ops cho Mamba để KHÔNG bỏ sót.")
    print("=" * 78)
    try:
        from thop import profile

        def thop_mamba(m, x, y):
            inp = x[0]
            b, L, _ = inp.shape
            m.total_ops += torch.DoubleTensor([mamba_macs_breakdown(m, b, L)["_total"]])

        xin = torch.randn(1, n_samples, device=device)
        macs_thop, params_thop = profile(
            model, inputs=(xin,), custom_ops={Mamba: thop_mamba}, verbose=False
        )
        print(f"  thop (CÓ custom Mamba): {macs_thop/1e9:.4f} G MACs | {params_thop/1e6:.4f} M")
        print("  (chạy lại KHÔNG custom_ops sẽ thấy tụt xuống ~ ptflops => Mamba bị sót)")
    except Exception as e:
        print(f"  thop lỗi/không cài: {e}")

    # =========================================================================
    # (E) fvcore — lộ unsupported ops (tắt fast_path để scan hiện ra)
    # =========================================================================
    print("\n" + "=" * 78)
    print("(E) FVCORE — danh sách 'unsupported / skipped ops' (đơn vị fma≈MAC)")
    print("=" * 78)
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        # tắt fast path để path tường minh chạy -> in_proj/out_proj thành aten,
        # và selective_scan_cuda lộ ra trong unsupported_ops
        for mod in model.modules():
            if isinstance(mod, Mamba):
                mod.use_fast_path = False

        xin = torch.randn(1, n_samples, device=device)
        flops = FlopCountAnalysis(model, xin)
        total = flops.total()
        print(f"  fvcore TỔNG (đếm được): {total/1e9:.4f} G (fma≈MAC)")
        print("\n  --- UNSUPPORTED / SKIPPED OPS (đây là chỗ Mamba bị bỏ qua) ---")
        uns = flops.unsupported_ops()
        if len(uns) == 0:
            print("  (rỗng — fvcore có thể đã không trace được vào kernel scan)")
        for op, cnt in sorted(uns.items(), key=lambda kv: -kv[1]):
            print(f"     {op:40s} x{cnt}")
        # bật lại fast path
        for mod in model.modules():
            if isinstance(mod, Mamba):
                mod.use_fast_path = True
    except Exception as e:
        print(f"  fvcore lỗi/không cài (hoặc không trace được scan): {e}")
        print("  => Bản thân việc fvcore không trace được scan cũng CHÍNH LÀ bằng chứng bị sót.")

    # =========================================================================
    # (F) Ước lượng tay riêng selective_scan
    # =========================================================================
    print("\n" + "=" * 78)
    print("(F) ƯỚC LƯỢNG TAY: selective_scan = 9 * batch * L * d_inner * d_state")
    print("=" * 78)
    by_kind = {}
    for name, mod, b, L in shapes:
        key = (b, L, mod.d_inner, mod.d_state)
        by_kind.setdefault(key, 0)
        by_kind[key] += 1
    scan_total = 0.0
    for (b, L, D, N), cnt in by_kind.items():
        per = 9 * b * L * D * N
        scan_total += per * cnt
        print(f"  {cnt:2d} lần gọi: 9*{b}*{L}*{D}*{N} = {per/1e9:.4f} G  => {per*cnt/1e9:.4f} G")
    print(f"  -> selective_scan tổng (chỉ phần scan): {scan_total/1e9:.4f} G")

    # =========================================================================
    # (G) Bảng tổng hợp
    # =========================================================================
    print("\n" + "=" * 78)
    print("(G) TỔNG HỢP (config 3 nguồn này)")
    print("=" * 78)
    pf = ptflops_macs / 1e9 if ptflops_macs else float("nan")
    print(f"  ptflops (thiếu Mamba)             : {pf:9.4f} G")
    print(f"  scan-only (ước lượng tay)         : {scan_total/1e9:9.4f} G")
    print(f"  Mamba đầy đủ (giải tích)          : {agg['_total']/1e9:9.4f} G")
    if ptflops_macs:
        print(f"  >>> CON SỐ ĐÚNG = ptflops + Mamba : {(ptflops_macs+agg['_total'])/1e9:9.4f} G  <<<")
    print(f"  Params                            : {count_params_M(model):9.4f} M")

    _ckpt.checkpoint = _orig_ckpt


if __name__ == "__main__":
    main()
