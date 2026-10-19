#!/bin/bash
# ==============================================================================
# setup_env_hust.sh
# Khởi tạo môi trường Conda cho dự án SPMamba -> K210 Distillation
# Hệ thống: NVIDIA DGX A100 — ĐH Bách Khoa Hà Nội (172.16.3.8)
#
# Cách chạy trên Head Node (chỉ cần chạy 1 lần đầu tiên):
#   bash setup_env_hust.sh
# ==============================================================================
set -e

ENV_NAME="spmamba_k210"

echo "=== [1/4] Kích hoạt Miniconda trên hệ thống HUST ==="
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Không tìm thấy conda.sh mặc định, thử dùng lệnh conda trực tiếp..."
fi

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Môi trường Conda '${ENV_NAME}' đã tồn tại."
else
    echo "=== [2/4] Tạo môi trường Conda '${ENV_NAME}' (Python 3.10) ==="
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

echo "=== [3/4] Cài đặt PyTorch (CUDA 12.1 cho A100) và các gói phụ thuộc ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning rich pyyaml h5py soundfile scipy numpy wandb onnx onnxsim torch-complex

echo "=== [4/4] Tạo cấu trúc thư mục logs và Experiments ==="
mkdir -p logs Experiments/checkpoint wandb_logs data

echo "=============================================================================="
echo "HOÀN TẤT CÀI ĐẶT MÔI TRƯỜNG TRÊN HEAD NODE!"
echo "Lưu ý: Gói CUDA 'causal-conv1d' và 'mamba-ssm' (cho Teacher SPMamba) sẽ được"
echo "kiểm tra và cài tự động trên GPU Compute Node khi bạn chạy:"
echo "    sbatch run_distill_a100.sh"
echo "=============================================================================="
