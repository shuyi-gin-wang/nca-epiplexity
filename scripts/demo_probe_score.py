"""
Sample N random NCA rules, compute gzip ratio AND probe-gain for each,
scatter them, and render exemplars from each quadrant.

Run from repo root:
    .venv/Scripts/python.exe scripts/demo_probe_score.py
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca import (
    NCA,
    compute_rule_gzip_batch,
    compute_rule_probe_batch,
    generate_nca_dataset,
)
from utils.tokenizers import NCA_Tokenizer

GRID = 12
PATCH = 2
NUM_COLORS = 10
NUM_RULES = 64
K = 8
N_IC = 4
ROLLOUT_STEPS = 24
PROBE_STEPS = 200
GZIP_N_STEPS = 10
RENDER_STEPS = 12
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Sampling {NUM_RULES} rules; k={K}, n_ic={N_IC}, probe_steps={PROBE_STEPS}")

    rng = jax.random.PRNGKey(SEED)
    seeds = jax.random.split(rng, NUM_RULES)

    tokenizer = NCA_Tokenizer(patch=PATCH, num_colors=NUM_COLORS)

    t0 = time.time()
    gzip_scores = compute_rule_gzip_batch(
        seeds, tokenizer,
        grid=GRID, d_state=NUM_COLORS, identity_bias=0.0, temperature=1e-4,
        n_steps=GZIP_N_STEPS, dT=1, start_step=0, mode="gzip",
    )
    print(f"gzip scores in {time.time() - t0:.2f}s; "
          f"range [{float(gzip_scores.min()):.3f}, {float(gzip_scores.max()):.3f}]")

    t0 = time.time()
    gain, final_loss, baseline = compute_rule_probe_batch(
        seeds,
        grid=GRID, d_state=NUM_COLORS, n_groups=1,
        identity_bias=0.0, temperature=1e-4,
        k=K, n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
        probe_steps=PROBE_STEPS, lr=1e-2,
    )
    print(f"probe scores in {time.time() - t0:.2f}s; "
          f"gain range [{float(gain.min()):.3f}, {float(gain.max()):.3f}]; "
          f"baseline range [{float(baseline.min()):.3f}, {float(baseline.max()):.3f}]")

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(jnp.asarray(gzip_scores), jnp.asarray(gain), s=24, alpha=0.7)
    ax.set_xlabel("gzip ratio (entropy floor)")
    ax.set_ylabel("probe gain  (reducible structure)")
    ax.set_title(f"NCA rules: gzip vs probe-gain  (k={K}, n_ic={N_IC})")
    ax.axhline(float(jnp.median(gain)), color="grey", lw=0.5, ls="--")
    ax.axvline(float(jnp.median(gzip_scores)), color="grey", lw=0.5, ls="--")
    # quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[1], ylim[1], "IV (target)", ha="right", va="top", fontsize=9, color="green")
    ax.text(xlim[1], ylim[0], "III chaos",   ha="right", va="bottom", fontsize=9, color="red")
    ax.text(xlim[0], ylim[1], "II ordered",  ha="left",  va="top", fontsize=9, color="orange")
    ax.text(xlim[0], ylim[0], "I fixed",     ha="left",  va="bottom", fontsize=9, color="grey")
    fig.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "probe_vs_gzip_scatter.png")
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)
    print(f"saved {scatter_path}")

    # Pick one exemplar per quadrant for visual sanity
    gz_med = float(jnp.median(gzip_scores))
    ga_med = float(jnp.median(gain))
    quadrants = {
        "IV_target":    (gzip_scores >= gz_med) & (gain >= ga_med),
        "III_chaos":    (gzip_scores >= gz_med) & (gain <  ga_med),
        "II_ordered":   (gzip_scores <  gz_med) & (gain >= ga_med),
        "I_fixed":      (gzip_scores <  gz_med) & (gain <  ga_med),
    }
    selected_idx = {}
    for name, mask in quadrants.items():
        idx = jnp.where(mask, size=NUM_RULES, fill_value=-1)[0]
        idx = idx[idx >= 0]
        if idx.shape[0] > 0:
            # pick the extremal one in each quadrant
            if name == "IV_target":
                pick = idx[jnp.argmax(gain[idx] + gzip_scores[idx])]
            elif name == "III_chaos":
                pick = idx[jnp.argmax(gzip_scores[idx] - gain[idx])]
            elif name == "II_ordered":
                pick = idx[jnp.argmax(gain[idx] - gzip_scores[idx])]
            else:
                pick = idx[jnp.argmin(gzip_scores[idx] + gain[idx])]
            selected_idx[name] = int(pick)

    if selected_idx:
        sel_seeds = jnp.stack([seeds[i] for i in selected_idx.values()])
        sims = generate_nca_dataset(
            seed=rng, num_sims=sel_seeds.shape[0],
            grid=GRID, d_state=NUM_COLORS, n_groups=1,
            identity_bias=0.0, temperature=1e-4,
            num_examples=RENDER_STEPS, num_rules=sel_seeds.shape[0],
            dT=1, rule_seeds=sel_seeds,
        )
        nca = NCA(grid_size=GRID, d_state=NUM_COLORS, n_groups=1, temperature=1e-4)

        fig, axes = plt.subplots(len(selected_idx), RENDER_STEPS,
                                 figsize=(1.6 * RENDER_STEPS, 1.8 * len(selected_idx)))
        if len(selected_idx) == 1:
            axes = axes[None, :]
        for row, (name, rule_b) in enumerate(selected_idx.items()):
            for t in range(RENDER_STEPS):
                img = nca.render_state(sims[row, t], params=None)
                axes[row, t].imshow(jnp.asarray(img))
                axes[row, t].axis("off")
                if t == 0:
                    axes[row, t].set_ylabel(name, fontsize=10)
            axes[row, 0].set_title(
                f"{name}  gzip={float(gzip_scores[rule_b]):.2f}  "
                f"gain={float(gain[rule_b]):.2f}",
                loc="left", fontsize=9,
            )
        fig.tight_layout()
        quad_path = os.path.join(OUT_DIR, "probe_quadrant_exemplars.png")
        fig.savefig(quad_path, dpi=120)
        plt.close(fig)
        print(f"saved {quad_path}")

    # Dump scores to text so we can rank later
    log_path = os.path.join(OUT_DIR, "probe_vs_gzip_scores.tsv")
    with open(log_path, "w") as f:
        f.write("idx\tgzip\tgain\tfinal_loss\tbaseline\n")
        for i in range(NUM_RULES):
            f.write(f"{i}\t{float(gzip_scores[i]):.4f}\t"
                    f"{float(gain[i]):.4f}\t{float(final_loss[i]):.4f}\t"
                    f"{float(baseline[i]):.4f}\n")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
