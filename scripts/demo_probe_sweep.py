"""
Sweep probe horizon k for N sampled NCA rules. Plot gain-vs-k curves
colored by Wolfram quadrant (assigned from gzip + gain at largest k).

Run from repo root:
    .venv/Scripts/python.exe scripts/demo_probe_sweep.py
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
    compute_rule_probe_sweep_batch,
    generate_nca_dataset,
)
from utils.tokenizers import NCA_Tokenizer

GRID = 12
PATCH = 2
NUM_COLORS = 10
NUM_RULES = 64
KS = (1, 2, 4, 8, 16)
N_IC = 4
PROBE_STEPS = 200
GZIP_N_STEPS = 10
SEED = 0

QUADRANTS = {
    "IV_target":  ("tab:green",  "high gzip  &  high gain"),
    "III_chaos":  ("tab:red",    "high gzip  &  low gain"),
    "II_ordered": ("tab:orange", "low gzip   &  high gain"),
    "I_fixed":    ("tab:gray",   "low gzip   &  low gain"),
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def assign_quadrants(gzip_scores, gain_at_largest_k):
    gz_med = float(jnp.median(gzip_scores))
    ga_med = float(jnp.median(gain_at_largest_k))
    labels = []
    for b in range(gzip_scores.shape[0]):
        gz = float(gzip_scores[b]); ga = float(gain_at_largest_k[b])
        if gz >= gz_med and ga >= ga_med:   labels.append("IV_target")
        elif gz >= gz_med and ga <  ga_med: labels.append("III_chaos")
        elif gz <  gz_med and ga >= ga_med: labels.append("II_ordered")
        else:                                labels.append("I_fixed")
    return labels, gz_med, ga_med


def pick_exemplar(idx_list, gain_curve, gzip_scores, name):
    """Pick the most representative rule from a quadrant."""
    if not idx_list:
        return None
    last = gain_curve[:, -1]
    if name == "IV_target":
        scores = [float(last[i] + gzip_scores[i]) for i in idx_list]
    elif name == "III_chaos":
        scores = [float(gzip_scores[i] - last[i]) for i in idx_list]
    elif name == "II_ordered":
        scores = [float(last[i] - gzip_scores[i]) for i in idx_list]
    else:
        scores = [-float(gzip_scores[i] + last[i]) for i in idx_list]
    return idx_list[int(jnp.argmax(jnp.array(scores)))]


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Sampling {NUM_RULES} rules; ks={KS}; n_ic={N_IC}; probe_steps={PROBE_STEPS}")

    rng = jax.random.PRNGKey(SEED)
    seeds = jax.random.split(rng, NUM_RULES)
    tokenizer = NCA_Tokenizer(patch=PATCH, num_colors=NUM_COLORS)

    t0 = time.time()
    gzip_scores = compute_rule_gzip_batch(
        seeds, tokenizer,
        grid=GRID, d_state=NUM_COLORS, identity_bias=0.0, temperature=1e-4,
        n_steps=GZIP_N_STEPS, dT=1, start_step=0, mode="gzip",
    )
    print(f"gzip in {time.time() - t0:.2f}s")

    t0 = time.time()
    gain_curve, fl_curve, bl_curve, ks_out = compute_rule_probe_sweep_batch(
        seeds, ks=KS,
        grid=GRID, d_state=NUM_COLORS, n_groups=1,
        identity_bias=0.0, temperature=1e-4,
        n_ic=N_IC, probe_steps=PROBE_STEPS, lr=1e-2,
    )
    print(f"probe-sweep in {time.time() - t0:.2f}s; "
          f"gain shape {tuple(gain_curve.shape)}; "
          f"gain at k={KS[-1]}: range [{float(gain_curve[:, -1].min()):.3f}, "
          f"{float(gain_curve[:, -1].max()):.3f}]")

    # Assign quadrant using gain at the LARGEST k (long-horizon learnability)
    labels, gz_med, ga_med = assign_quadrants(gzip_scores, gain_curve[:, -1])
    print(f"medians: gzip={gz_med:.3f}, gain@k={KS[-1]}={ga_med:.3f}")
    for name, _ in QUADRANTS.items():
        print(f"  {name}: {labels.count(name)} rules")

    # ---- Figure 1: gain-vs-k curves, colored by quadrant ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ks_arr = jnp.array(KS)
    for b in range(NUM_RULES):
        color, _ = QUADRANTS[labels[b]]
        ax.plot(ks_arr, gain_curve[b], color=color, alpha=0.35, lw=1)

    # Highlight one exemplar per quadrant with a thick line
    by_quad = {name: [i for i, lbl in enumerate(labels) if lbl == name]
               for name in QUADRANTS}
    exemplars = {}
    for name, idx_list in by_quad.items():
        ex = pick_exemplar(idx_list, gain_curve, gzip_scores, name)
        if ex is not None:
            exemplars[name] = ex
            color, _ = QUADRANTS[name]
            ax.plot(ks_arr, gain_curve[ex], color=color, lw=2.5, marker="o",
                    label=f"{name} (rule {ex}, gzip={float(gzip_scores[ex]):.2f})")

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("horizon  k  (steps ahead)")
    ax.set_ylabel("probe gain  (1 - loss / marginal-baseline)")
    ax.set_title(f"Probe gain vs horizon for {NUM_RULES} sampled NCA rules\n"
                 f"colored by Wolfram quadrant (gzip × gain @ k={KS[-1]})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, "probe_gain_vs_k.png")
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)
    print(f"saved {curve_path}")

    # ---- Figure 2: quadrant-averaged gain curves with shaded IQR ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (color, desc) in QUADRANTS.items():
        idx = by_quad[name]
        if not idx: continue
        curves = gain_curve[jnp.array(idx)]  # (n, K)
        med = jnp.median(curves, axis=0)
        q25 = jnp.quantile(curves, 0.25, axis=0)
        q75 = jnp.quantile(curves, 0.75, axis=0)
        ax.fill_between(ks_arr, q25, q75, color=color, alpha=0.18)
        ax.plot(ks_arr, med, color=color, lw=2.2, marker="o",
                label=f"{name}  (n={len(idx)})")

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("horizon  k")
    ax.set_ylabel("probe gain (median; IQR shaded)")
    ax.set_title("Long-horizon learnability by quadrant")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    quad_path = os.path.join(OUT_DIR, "probe_gain_vs_k_by_quadrant.png")
    fig.savefig(quad_path, dpi=140)
    plt.close(fig)
    print(f"saved {quad_path}")

    # ---- Render the exemplars at long horizon for visual confirmation ----
    if exemplars:
        sel_idx = list(exemplars.values())
        sel_seeds = jnp.stack([seeds[i] for i in sel_idx])
        render_T = 12
        sims = generate_nca_dataset(
            seed=rng, num_sims=sel_seeds.shape[0],
            grid=GRID, d_state=NUM_COLORS, n_groups=1,
            identity_bias=0.0, temperature=1e-4,
            num_examples=render_T, num_rules=sel_seeds.shape[0],
            dT=1, rule_seeds=sel_seeds,
        )
        nca = NCA(grid_size=GRID, d_state=NUM_COLORS, n_groups=1, temperature=1e-4)

        # one row per quadrant: leftmost narrow column is the label, rest are timesteps
        nrows = len(exemplars)
        fig, axes = plt.subplots(
            nrows, render_T + 1,
            figsize=(1.4 * (render_T + 1), 1.9 * nrows),
            gridspec_kw={"width_ratios": [2.2] + [1.0] * render_T,
                         "wspace": 0.05, "hspace": 0.25},
        )
        if nrows == 1:
            axes = axes[None, :]

        for row, (name, rule_b) in enumerate(exemplars.items()):
            # label cell
            color, _ = QUADRANTS[name]
            curve_str = "\n".join(
                f"k={k:>2d}: {float(gain_curve[rule_b, j]):.2f}"
                for j, k in enumerate(KS)
            )
            axes[row, 0].axis("off")
            axes[row, 0].text(
                0.02, 0.5,
                f"{name}\nrule {rule_b}\ngzip={float(gzip_scores[rule_b]):.2f}\n\n{curve_str}",
                ha="left", va="center", fontsize=9, color=color,
                family="monospace", transform=axes[row, 0].transAxes,
            )
            # timestep frames
            for t in range(render_T):
                ax = axes[row, t + 1]
                ax.imshow(jnp.asarray(nca.render_state(sims[row, t], params=None)))
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(color); s.set_linewidth(1.5)
                if row == 0:
                    ax.set_title(f"t={t}", fontsize=8)

        fig.suptitle("Quadrant exemplars — same NCA rule rolled out, with its gain-vs-k profile",
                     fontsize=11, y=0.995)
        ex_path = os.path.join(OUT_DIR, "probe_sweep_exemplars.png")
        fig.savefig(ex_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {ex_path}")

    # Dump curves to tsv
    log_path = os.path.join(OUT_DIR, "probe_sweep_scores.tsv")
    with open(log_path, "w") as f:
        f.write("idx\tquadrant\tgzip\t" + "\t".join(f"gain_k{k}" for k in KS) + "\n")
        for i in range(NUM_RULES):
            row = [str(i), labels[i], f"{float(gzip_scores[i]):.4f}"]
            row += [f"{float(gain_curve[i, j]):.4f}" for j in range(len(KS))]
            f.write("\t".join(row) + "\n")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
