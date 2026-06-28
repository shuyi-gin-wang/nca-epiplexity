#!/usr/bin/env bash
#SBATCH --job-name=epx-tr-k16-floor
#SBATCH --account=hpc_core
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/scratch/user/shuywang/slurm_logs/%x-%j.out

set -euo pipefail

module load nvidia/cuda/12.9.1

export MPLBACKEND=Agg
export MPLCONFIGDIR=/mnt/scratch/user/shuywang/nca-epiplexity/.cache/matplotlib
export PYTHONUNBUFFERED=1

cd /mnt/scratch/user/shuywang/nca-epiplexity
mkdir -p "$MPLCONFIGDIR" /mnt/scratch/user/shuywang/slurm_logs

/mnt/scratch/user/shuywang/envs/nca-epx/bin/python scripts/epiplexity/train.py \
  --run-name epx_transformer_direct_K16_pdrop0_floor03_g1500 \
  --gzip-mode threshold \
  --threshold 0.3 \
  --p-drop 0.0 \
  --probe-arch transformer \
  --horizon-mode direct \
  --probe-horizon 16 \
  --pop-size 128 \
  --generations 1500 \
  --sigma-decay 0.998 \
  --checkpoint-every 100
