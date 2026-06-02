"""
Full sweep of the continuous-NCA prequential probe on GPU.

Config: 64 rules x 4 IC x 256 rollout steps x 128x128 grid x d_state=3
        x 200 probe steps. Streams per-rule rollouts to avoid materializing
        the ~13 GB aggregate tensor on a 8 GB GPU. Re-rolls top/bottom rules
        at the end for rendering.

Run from repo root:
    .venv/Scripts/python.exe scripts/demo_probe_preq_continuous_torch_sweep.py
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca_torch import (
    NCAContinuousTorch,
    rollout_one_rule,
    sweep_rules_streaming,
)

GRID = 128
D_STATE = 3
P_DROP = 0.5
DT = 0.01
NUM_RULES = 64
N_IC = 4
ROLLOUT_STEPS = 256
PROBE_STEPS = 200
LR = 1e-2
RENDER_STEPS = 8
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}  device={device}  "
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(
        f"Continuous NCA preq (torch, streaming): {NUM_RULES} rules, grid={GRID}, "
        f"d_state={D_STATE}, p_drop={P_DROP}, dt={DT}, "
        f"rollout_steps={ROLLOUT_STEPS}, n_ic={N_IC}, probe_steps={PROBE_STEPS}"
    )

    rule_seeds = torch.arange(NUM_RULES, dtype=torch.int64) + SEED
    substrate = NCAContinuousTorch(
        grid_size=GRID, d_state=D_STATE, p_drop=P_DROP, dt=DT, device=device,
    )

    t0 = time.time()
    preq_gain, preq_length, mse_floor, initial_loss, baseline = sweep_rules_streaming(
        substrate, rule_seeds,
        n_ic=N_IC, rollout_steps=ROLLOUT_STEPS, probe_steps=PROBE_STEPS, lr=LR,
        ic_rng_seed=SEED + 1, probe_rng_seed=SEED,
        progress_every=8,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(
        f"sweep done in {elapsed:.1f}s ({elapsed/NUM_RULES:.2f}s/rule); "
        f"gain [{float(preq_gain.min()):.3f}, {float(preq_gain.max()):.3f}]; "
        f"floor [{float(mse_floor.min()):.5f}, {float(mse_floor.max()):.5f}]; "
        f"initial_loss [{float(initial_loss.min()):.4f}, {float(initial_loss.max()):.4f}]"
    )

    # Scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mse_floor.numpy(), preq_gain.numpy(), s=28, alpha=0.7)
    for i in range(NUM_RULES):
        ax.annotate(str(i), (float(mse_floor[i]), float(preq_gain[i])),
                    fontsize=6, alpha=0.5)
    ax.set_xlabel("MSE floor (final probe loss)")
    ax.set_ylabel("prequential probe gain  (in [0,1])")
    ax.set_title(
        f"Continuous NCA preq sweep  N={NUM_RULES}\n"
        f"grid={GRID} d={D_STATE} p_drop={P_DROP} T={ROLLOUT_STEPS} n_ic={N_IC} "
        f"probe_steps={PROBE_STEPS}"
    )
    fig.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_sweep_scatter.png")
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)
    print(f"saved {scatter_path}")

    # Histogram of gain
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(preq_gain.numpy(), bins=20, edgecolor="black")
    ax.set_xlabel("prequential probe gain")
    ax.set_ylabel("count")
    ax.set_title(f"preq_gain histogram  (N={NUM_RULES})")
    fig.tight_layout()
    hist_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_sweep_hist.png")
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)
    print(f"saved {hist_path}")

    # Re-roll top-3 / bottom-3 for rendering
    order = torch.argsort(preq_gain)
    picks = order[-3:].flip(0).tolist() + order[:3].tolist()
    labels = ["top1", "top2", "top3", "bot1", "bot2", "bot3"]

    fig, axes = plt.subplots(len(picks), RENDER_STEPS,
                             figsize=(1.6 * RENDER_STEPS, 1.8 * len(picks)))
    t_idx = torch.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).long().tolist()
    for row, (rule_b, name) in enumerate(zip(picks, labels)):
        sims_one = rollout_one_rule(
            substrate, int(rule_seeds[rule_b].item()), rule_b, N_IC, ROLLOUT_STEPS,
            ic_rng_seed=SEED + 1,
        )
        for col, t in enumerate(t_idx):
            img = sims_one[0, t].clamp(0, 1).cpu().numpy()
            if D_STATE == 1:
                img = img.repeat(3, axis=-1)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
        axes[row, 0].set_title(
            f"{name} rule={rule_b}  "
            f"gain={float(preq_gain[rule_b]):.3f}  "
            f"floor={float(mse_floor[rule_b]):.5f}",
            loc="left", fontsize=9,
        )
        del sims_one
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fig.tight_layout()
    grid_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_sweep_exemplars.png")
    fig.savefig(grid_path, dpi=110)
    plt.close(fig)
    print(f"saved {grid_path}")

    log_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_sweep_scores.tsv")
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
