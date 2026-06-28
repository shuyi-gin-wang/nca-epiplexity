#!/usr/bin/env bash
#SBATCH --job-name=epx-obs-hm-g32
#SBATCH --account=hpc_core
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/scratch/user/shuywang/slurm_logs/%x-%j.out

set -euo pipefail

module load nvidia/cuda/12.9.1

export MPLBACKEND=Agg
export MPLCONFIGDIR=/mnt/scratch/user/shuywang/nca-epiplexity/.cache/matplotlib
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/scratch/user/shuywang/nca-epiplexity
mkdir -p "$MPLCONFIGDIR" /mnt/scratch/user/shuywang/slurm_logs

nvidia-smi

/mnt/scratch/user/shuywang/envs/nca-epx/bin/python scripts/epiplexity/train.py \
  --run-name epx_observer_balanced_hmean_gain015_directK16_floor03_grid32_g1500_h100_chunk32 \
  --gzip-mode threshold \
  --threshold 0.3 \
  --p-drop 0.0 \
  --probe-archs linear,mlp_wide,deep_mlp,transformer \
  --probe-combine hmean \
  --learnability-floor 0.15 \
  --horizon-mode direct \
  --probe-horizon 16 \
  --probe-lr 0.01 \
  --grid 32 \
  --pop-size 128 \
  --eval-batch-size 32 \
  --generations 1500 \
  --sigma-decay 0.998 \
  --checkpoint-every 100
