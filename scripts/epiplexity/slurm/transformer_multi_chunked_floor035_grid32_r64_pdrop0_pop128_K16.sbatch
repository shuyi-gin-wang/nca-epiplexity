#!/usr/bin/env bash
#SBATCH --job-name=epx-cmul16-g32r64-p128
#SBATCH --account=hpc_core
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/scratch/user/shuywang/slurm_logs/%x-%j.out

set -euo pipefail

module load nvidia/cuda/12.9.1

export MPLBACKEND=Agg
export MPLCONFIGDIR=/mnt/scratch/user/shuywang/nca-epiplexity-grid32/.cache/matplotlib
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/scratch/user/shuywang/nca-epiplexity-grid32
mkdir -p "$MPLCONFIGDIR" /mnt/scratch/user/shuywang/slurm_logs

/mnt/scratch/user/shuywang/envs/nca-epx/bin/python scripts/epiplexity/trainers/evolve_gzip_chunked.py \
  --run-name runs/epiplexity/epx_transformer_multi_K16_pdrop0_floor035_grid32_r64_pop128_chunk1_g1500 \
  --gzip-mode threshold \
  --gzip-threshold 0.35 \
  --grid 32 \
  --rollout-steps 64 \
  --p-drop 0.0 \
  --probe-arch transformer \
  --horizon-mode multi \
  --multi-ks 1,2,4,8,16 \
  --probe-horizon 16 \
  --pop-size 128 \
  --eval-batch-size 1 \
  --n-generations 1500 \
  --sigma-decay 0.998 \
  --checkpoint-every 25
