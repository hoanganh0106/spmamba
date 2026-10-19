#!/bin/bash
#SBATCH --job-name=k210_distill
#SBATCH --output=logs/k210_output_%j.log
#SBATCH --error=logs/k210_error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --requeue

# ==============================================================================
# Script SLURM chạy Online Knowledge Distillation trên NVIDIA DGX A100 80GB (HUST)
#
# Cách submit job (từ trong ~/my_project):
#   cd ~/my_project && mkdir -p logs
#   sbatch run_distill_a100.sh
# Theo dõi log:
#   tail -f logs/k210_output_<job_id>.log
# ==============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/my_project}"
cd "$PROJECT_DIR"

echo "============================================================"
echo "SLURM Job ID      : ${SLURM_JOB_ID:-local}"
echo "SLURM Node        : ${SLURM_NODELIST:-$(hostname)}"
echo "CUDA Visible Dev  : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working Directory : $(pwd)"
echo "Start Time        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 1. Kích hoạt môi trường Conda trên máy chủ HUST
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate spmamba

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HYDRA_FULL_ERROR=1

# 2. Kiểm tra GPU bẩn (Zombie VRAM từ job của người dùng khác trên node SLURM)
echo "=== Kiểm tra trạng thái GPU A100 ==="
nvidia-smi

USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
echo "VRAM đang bị chiếm trước khi chạy: ${USED_MEM} MiB / ${TOTAL_MEM} MiB"

if [ "${USED_MEM}" -gt 15000 ]; then
    echo "[CẢNH BÁO] GPU trên node ${SLURM_NODELIST:-unknown} đang bị tiến trình rác chiếm ${USED_MEM} MiB VRAM!"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "Đang yêu cầu SLURM requeue job ${SLURM_JOB_ID} sang GPU sạch..."
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
fi

# 3. Kiểm tra dữ liệu HDF5 và Checkpoint Teacher SPMamba
CONFIG_FILE="${CONFIG_FILE:-configs/distill-k210-a100.yml}"
H5_PATH="${H5_PATH:-data/data_30h.h5}"
TEACHER_CKPT="${TEACHER_CKPT:-output/epoch=91-best.ckpt}"
CKPT_DIR="Experiments/checkpoint/K210-SepUNet-Attention-A100"

if [ ! -f "$H5_PATH" ]; then
    echo "[LỖI] Không tìm thấy file dữ liệu HDF5 tại: $H5_PATH"
    exit 1
fi

# 4. Chạy Smoke Test nhanh trước khi huấn luyện
echo "=== Chạy Smoke Test kiểm tra toàn bộ pipeline K210 ==="
python smoke_k210_distill.py

# 5. Tự động nối lại (Auto-resume) nếu có checkpoint last.ckpt từ lần chạy trước
RESUME_ARGS=()
if [ -f "${CKPT_DIR}/last.ckpt" ]; then
    echo "=== Phát hiện checkpoint '${CKPT_DIR}/last.ckpt' -> Tự động tiếp tục (Resume) ==="
    RESUME_ARGS=(--resume_ckpt "${CKPT_DIR}/last.ckpt")
fi

# Cờ WandB (mặc định bật online; đặt WANDB_OFFLINE=1 nếu compute node chặn mạng)
WANDB_ARGS=(--wandb)
if [ "${WANDB_OFFLINE:-0}" = "1" ]; then
    WANDB_ARGS=(--wandb --wandb_offline)
fi

# 6. Chạy huấn luyện Hybrid K210 (KPU Conv2D + RISC-V Dual-Path Attention & BiGRU)
echo "=== Bắt đầu huấn luyện Hybrid K210 (KPU + RISC-V Attention/BiGRU) ==="
python distill_train.py \
    --conf_dir "$CONFIG_FILE" \
    --h5_path "$H5_PATH" \
    "${RESUME_ARGS[@]}" \
    "${WANDB_ARGS[@]}"

# 7. Đánh giá tự động SI-SNR / SI-SNRi / SDR / SDRi trên tập Test
BEST_CKPT="${CKPT_DIR}/best_k210.ckpt"
if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT="${CKPT_DIR}/last.ckpt"
fi

if [ -f "$BEST_CKPT" ]; then
    echo "=== Đánh giá chất lượng Student trên tập Test (${BEST_CKPT}) ==="
    python eval_k210.py \
        --ckpt "$BEST_CKPT" \
        --conf_dir "$CONFIG_FILE" \
        --h5_path "$H5_PATH" \
        --save_csv "${CKPT_DIR}/metrics_k210.csv"

    echo "=== Xuất mô hình ONNX (Opset 11) và dữ liệu hiệu chuẩn INT8 cho K210 ==="
    python export_k210.py \
        --ckpt "$BEST_CKPT" \
        --out_onnx "${CKPT_DIR}/k210_sep3.onnx" \
        --h5_path "$H5_PATH" \
        --calib_dir "${CKPT_DIR}/k210_calib" \
        --calib_samples 100
fi

echo "============================================================"
echo "HOÀN TẤT! Thời gian kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"
echo " - Checkpoint tốt nhất : ${BEST_CKPT}"
echo " - Bảng điểm SDRi/SI-SNRi: ${CKPT_DIR}/metrics_k210.csv"
echo " - File ONNX cho K210  : ${CKPT_DIR}/k210_sep3.onnx"
echo "============================================================"
