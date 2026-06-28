"""
Quick check for the continuous-NCA prequential probe.

Mirrors probe_discrete_epiplexity.py but on the ASAL-style continuous substrate
(NCAContinuous: 128 grid, d_state=3, p_drop=0.5, dt=0.01). State is continuous
in [0,1]^D, so the probe is trained with MSE against a variance baseline.

Quick config: 16 rules, 64 rollout steps, 2 ICs.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/demos/probe_continuous_jax.py
"""
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT, add_repo_root_to_path

add_repo_root_to_path()

from utils.nca import (
    NCAContinuous,
    _generate_rule_rollouts_continuous,
    _train_probes_from_rollouts_preq_continuous,
)

GRID = 128
D_STATE = 3
P_DROP = 0.5
DT = 0.01
NUM_RULES = 16
N_IC = 2
ROLLOUT_STEPS = 64
PROBE_STEPS = 200
RENDER_STEPS = 8
SEED = 0

OUT_DIR = str(DEMO_OUT)
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"JAX devices: {jax.devices()}")
    print(
        f"Continuous NCA preq quick check: {NUM_RULES} rules, grid={GRID}, "
        f"d_state={D_STATE}, p_drop={P_DROP}, dt={DT}, "
        f"rollout_steps={ROLLOUT_STEPS}, n_ic={N_IC}, probe_steps={PROBE_STEPS}"
    )

    rng = jax.random.PRNGKey(SEED)
    seeds = jax.random.split(rng, NUM_RULES)

    t0 = time.time()
    sims = _generate_rule_rollouts_continuous(
        seeds, GRID, D_STATE, P_DROP, DT,
        ROLLOUT_STEPS, N_IC, start_step=0, ic_rng_seed=SEED + 1,
    )
    sims.block_until_ready()
    print(f"rollouts in {time.time() - t0:.2f}s; sims shape {sims.shape}")

    t0 = time.time()
    preq_gain, preq_length, mse_floor, baseline = _train_probes_from_rollouts_preq_continuous(
        sims, D_STATE, PROBE_STEPS, lr=1e-2, probe_rng_seed=SEED,
    )
    preq_gain.block_until_ready()
    print(
        f"preq probe in {time.time() - t0:.2f}s; "
        f"gain range [{float(preq_gain.min()):.3f}, {float(preq_gain.max()):.3f}]; "
        f"floor range [{float(mse_floor.min()):.4f}, {float(mse_floor.max()):.4f}]; "
        f"baseline range [{float(baseline.min()):.4f}, {float(baseline.max()):.4f}]"
    )

    # Scatter: prequential gain vs MSE floor (analogue of preq vs gzip)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(jnp.asarray(mse_floor), jnp.asarray(preq_gain), s=32, alpha=0.75)
    for i in range(NUM_RULES):
        ax.annotate(str(i), (float(mse_floor[i]), float(preq_gain[i])),
                    fontsize=7, alpha=0.6)
    ax.set_xlabel("MSE floor  (lower = more predictable next-step)")
    ax.set_ylabel("prequential probe gain")
    ax.set_title(
        f"Continuous NCA (ASAL-style) preq probe\n"
        f"grid={GRID} d={D_STATE} p_drop={P_DROP} T={ROLLOUT_STEPS} n_ic={N_IC}"
    )
    fig.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "probe_preq_continuous_scatter.png")
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)
    print(f"saved {scatter_path}")

    # Render: top-3 by gain and bottom-3 (most/least learnable rules)
    order = jnp.argsort(preq_gain)
    picks = list(order[-3:][::-1].tolist()) + list(order[:3].tolist())
    labels = ["top1", "top2", "top3", "bot1", "bot2", "bot3"]
    nca = NCAContinuous(grid_size=GRID, d_state=D_STATE, p_drop=P_DROP, dt=DT)

    fig, axes = plt.subplots(len(picks), RENDER_STEPS,
                             figsize=(1.6 * RENDER_STEPS, 1.8 * len(picks)))
    t_idx = jnp.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).astype(jnp.int32)
    for row, (rule_b, name) in enumerate(zip(picks, labels)):
        for col, t in enumerate(t_idx.tolist()):
            img = nca.render_state(sims[rule_b, 0, t], params=None)
            axes[row, col].imshow(jnp.asarray(img))
            axes[row, col].axis("off")
        axes[row, 0].set_title(
            f"{name} rule={rule_b}  "
            f"gain={float(preq_gain[rule_b]):.3f}  "
            f"floor={float(mse_floor[rule_b]):.4f}",
            loc="left", fontsize=9,
        )
    fig.tight_layout()
    grid_path = os.path.join(OUT_DIR, "probe_preq_continuous_exemplars.png")
    fig.savefig(grid_path, dpi=110)
    plt.close(fig)
    print(f"saved {grid_path}")

    log_path = os.path.join(OUT_DIR, "probe_preq_continuous_scores.tsv")
    with open(log_path, "w") as f:
        f.write("idx\tpreq_gain\tpreq_length\tmse_floor\tbaseline\n")
        for i in range(NUM_RULES):
            f.write(
                f"{i}\t{float(preq_gain[i]):.4f}\t{float(preq_length[i]):.4f}\t"
                f"{float(mse_floor[i]):.6f}\t{float(baseline[i]):.6f}\n"
            )
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
