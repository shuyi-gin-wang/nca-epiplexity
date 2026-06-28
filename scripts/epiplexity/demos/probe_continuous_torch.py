"""
PyTorch / CUDA quick check for the continuous-NCA prequential probe.

Mirrors probe_continuous_jax.py but uses utils/nca_torch.py:
  - ASAL-style continuous NCA (grid=128, d_state=3, p_drop=0.5, dt=0.01)
  - probe: Conv3x3 -> d_state, MSE loss
  - fixed normalization: preq_gain = preq_length / ((initial_loss - floor) * steps)
    bounded in [0,1]

Quick config: 16 rules, 64 rollout steps, 2 ICs.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/demos/probe_continuous_torch.py
"""
import os
import sys
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT, add_repo_root_to_path

add_repo_root_to_path()

from utils.nca_torch import (
    NCAContinuousTorch,
    describe_torch_device,
    generate_rule_rollouts_continuous,
    resolve_torch_device,
    train_probe_preq_continuous,
)

GRID = 128
D_STATE = 3
P_DROP = 0.5
DT = 0.01
NUM_RULES = 16
N_IC = 2
ROLLOUT_STEPS = 64
PROBE_STEPS = 200
LR = 1e-2
RENDER_STEPS = 8
SEED = 0

OUT_DIR = str(DEMO_OUT)
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_torch_device(args.device, allow_cpu=args.allow_cpu)
    print(f"torch {torch.__version__}  device={describe_torch_device(device)}")
    print(
        f"Continuous NCA preq (torch) quick check: {NUM_RULES} rules, grid={GRID}, "
        f"d_state={D_STATE}, p_drop={P_DROP}, dt={DT}, "
        f"rollout_steps={ROLLOUT_STEPS}, n_ic={N_IC}, probe_steps={PROBE_STEPS}"
    )

    rule_seeds = torch.arange(NUM_RULES, dtype=torch.int64) + SEED

    substrate = NCAContinuousTorch(
        grid_size=GRID, d_state=D_STATE, p_drop=P_DROP, dt=DT, device=device,
    )

    t0 = time.time()
    sims = generate_rule_rollouts_continuous(
        substrate, rule_seeds, n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
        ic_rng_seed=SEED + 1,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"rollouts in {time.time() - t0:.2f}s; sims shape {tuple(sims.shape)}")

    t0 = time.time()
    preq_gain, preq_length, mse_floor, initial_loss, baseline = train_probe_preq_continuous(
        sims, d_state=D_STATE, probe_steps=PROBE_STEPS, lr=LR, rng_seed=SEED,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"preq probe in {time.time() - t0:.2f}s; "
        f"gain [{float(preq_gain.min()):.3f}, {float(preq_gain.max()):.3f}]; "
        f"floor [{float(mse_floor.min()):.4f}, {float(mse_floor.max()):.4f}]; "
        f"initial_loss [{float(initial_loss.min()):.4f}, {float(initial_loss.max()):.4f}]; "
        f"baseline [{float(baseline.min()):.4f}, {float(baseline.max()):.4f}]"
    )

    # Scatter: gain vs floor
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mse_floor.cpu(), preq_gain.cpu(), s=32, alpha=0.75)
    for i in range(NUM_RULES):
        ax.annotate(str(i), (float(mse_floor[i]), float(preq_gain[i])),
                    fontsize=7, alpha=0.6)
    ax.set_xlabel("MSE floor")
    ax.set_ylabel("prequential probe gain (renormalized, in [0,1])")
    ax.set_title(
        f"Continuous NCA (torch, ASAL-style) preq probe\n"
        f"grid={GRID} d={D_STATE} p_drop={P_DROP} T={ROLLOUT_STEPS} n_ic={N_IC}"
    )
    fig.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_scatter.png")
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)
    print(f"saved {scatter_path}")

    # Render top-3 / bottom-3 by gain
    order = torch.argsort(preq_gain)
    picks = order[-3:].flip(0).tolist() + order[:3].tolist()
    labels = ["top1", "top2", "top3", "bot1", "bot2", "bot3"]

    fig, axes = plt.subplots(len(picks), RENDER_STEPS,
                             figsize=(1.6 * RENDER_STEPS, 1.8 * len(picks)))
    t_idx = torch.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).long().tolist()
    for row, (rule_b, name) in enumerate(zip(picks, labels)):
        for col, t in enumerate(t_idx):
            img = sims[rule_b, 0, t].clamp(0, 1).cpu().numpy()  # (H, W, D)
            if D_STATE == 1:
                img = img.repeat(3, axis=-1)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        axes[row, 0].set_title(
            f"{name} rule={rule_b}  "
            f"gain={float(preq_gain[rule_b]):.3f}  "
            f"floor={float(mse_floor[rule_b]):.4f}",
            loc="left", fontsize=9,
        )
    fig.tight_layout()
    grid_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_exemplars.png")
    fig.savefig(grid_path, dpi=110)
    plt.close(fig)
    print(f"saved {grid_path}")

    log_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_scores.tsv")
    with open(log_path, "w") as f:
        f.write("idx\tpreq_gain\tpreq_length\tmse_floor\tinitial_loss\tbaseline\n")
        for i in range(NUM_RULES):
            f.write(
                f"{i}\t{float(preq_gain[i]):.4f}\t{float(preq_length[i]):.4f}\t"
                f"{float(mse_floor[i]):.6f}\t{float(initial_loss[i]):.6f}\t"
                f"{float(baseline[i]):.6f}\n"
            )
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
