#!/usr/bin/env bash
#SBATCH --job-name=epx-ens-floor
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
export MPLCONFIGDIR=/mnt/scratch/user/shuywang/nca-epiplexity/.cache/matplotlib
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/scratch/user/shuywang/nca-epiplexity
mkdir -p "$MPLCONFIGDIR" /mnt/scratch/user/shuywang/slurm_logs

nvidia-smi

/mnt/scratch/user/shuywang/envs/nca-epx/bin/python scripts/epiplexity/train.py \
  --run-name epx_probe_ensemble_mean_directK16_floor03_g1500_rtxpro6000 \
  --gzip-mode threshold \
  --threshold 0.3 \
  --p-drop 0.0 \
  --probe-archs linear,mlp_wide,deep_mlp,transformer \
  --probe-combine mean \
  --horizon-mode direct \
  --probe-horizon 16 \
  --probe-lr 0.01 \
  --pop-size 128 \
  --generations 1500 \
  --sigma-decay 0.998 \
  --checkpoint-every 100