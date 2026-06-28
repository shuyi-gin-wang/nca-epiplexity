#!/usr/bin/env bash
#SBATCH --job-name=epx-tr-k16-g32
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

cd /mnt/scratch/user/shuywang/nca-epiplexity-grid32
mkdir -p "$MPLCONFIGDIR" /mnt/scratch/user/shuywang/slurm_logs

/mnt/scratch/user/shuywang/envs/nca-epx/bin/python scripts/epiplexity/train.py \
  --run-name epx_transformer_direct_K16_pdrop0_floor03_grid32_g1500 \
  --gzip-mode threshold \
  --threshold 0.3 \
  --grid 32 \
  --p-drop 0.0 \
  --probe-arch transformer \
  --horizon-mode direct \
  --probe-horizon 16 \
  --pop-size 32 \
  --generations 1500 \
  --sigma-decay 0.998 \
  --checkpoint-every 100
