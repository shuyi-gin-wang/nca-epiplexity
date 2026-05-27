"""
Sample N random NCA rules, compute gzip ratio + fixed-k probe gain +
prequential probe gain + the new requential probe gain (Finzi et al.
arXiv:2601.03220, distillation against the true NCA teacher distribution),
scatter all three probe scores against gzip, render quadrant exemplars
(selected by the requential score), and dump all scores side-by-side.

Run from repo root:
    .venv/Scripts/python.exe scripts/demo_probe_requential.py
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
    compute_rule_probe_preq_batch,
    compute_rule_probe_requential_batch,
    generate_nca_dataset,
)
from utils.tokenizers import NCA_Tokenizer

GRID = 12
PATCH = 2
NUM_COLORS = 10
NUM_RULES = 64
K_FIXED = 8
N_IC = 4
ROLLOUT_STEPS = 24
PROBE_STEPS = 200
GZIP_N_STEPS = 10
RENDER_STEPS = 12
EXEMPLARS_PER_QUADRANT = 4
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Sampling {NUM_RULES} rules; n_ic={N_IC}, probe_steps={PROBE_STEPS}")

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
    fixed_gain, fixed_loss, _ = compute_rule_probe_batch(
        seeds,
        grid=GRID, d_state=NUM_COLORS, n_groups=1,
        identity_bias=0.0, temperature=1e-4,
        k=K_FIXED, n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
        probe_steps=PROBE_STEPS, lr=1e-2,
    )
    print(f"fixed-k={K_FIXED} probe in {time.time() - t0:.2f}s; "
          f"gain range [{float(fixed_gain.min()):.3f}, {float(fixed_gain.max()):.3f}]")

    t0 = time.time()
    preq_gain, preq_length, preq_floor, preq_baseline = compute_rule_probe_preq_batch(
        seeds,
        grid=GRID, d_state=NUM_COLORS, n_groups=1,
        identity_bias=0.0, temperature=1e-4,
        n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
        probe_steps=PROBE_STEPS, lr=1e-2,
    )
    print(f"preq probe in {time.time() - t0:.2f}s; "
          f"gain range [{float(preq_gain.min()):.3f}, {float(preq_gain.max()):.3f}]")

    t0 = time.time()
    req_gain, req_length, req_floor, req_baseline = compute_rule_probe_requential_batch(
        seeds,
        grid=GRID, d_state=NUM_COLORS, n_groups=1,
        identity_bias=0.0, temperature=1e-4,
        n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
        probe_steps=PROBE_STEPS, lr=1e-2,
    )
    print(f"requential probe in {time.time() - t0:.2f}s; "
          f"gain range [{float(req_gain.min()):.3f}, {float(req_gain.max()):.3f}]; "
          f"floor range [{float(req_floor.min()):.3f}, {float(req_floor.max()):.3f}]")

    # Pearson correlation between the three probe scores
    def pearson(a, b):
        a = jnp.asarray(a); b = jnp.asarray(b)
        am = a - a.mean(); bm = b - b.mean()
        return float((am * bm).sum() / (jnp.sqrt((am * am).sum() * (bm * bm).sum()) + 1e-12))
    print(f"corr(fixed, preq) = {pearson(fixed_gain, preq_gain):.4f}")
    print(f"corr(fixed, req)  = {pearson(fixed_gain, req_gain):.4f}")
    print(f"corr(preq,  req)  = {pearson(preq_gain,  req_gain):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (axes[0], fixed_gain, f"gzip vs fixed-k probe gain  (k={K_FIXED})"),
        (axes[1], preq_gain,  "gzip vs prequential probe gain  (CE on samples)"),
        (axes[2], req_gain,   "gzip vs requential probe gain  (KL distillation)"),
    ]
    for ax, score, title in panels:
        ax.scatter(jnp.asarray(gzip_scores), jnp.asarray(score), s=24, alpha=0.7)
        ax.set_xlabel("gzip ratio")
        ax.set_ylabel("probe gain")
        ax.set_title(title)
        ax.axhline(float(jnp.median(score)), color="grey", lw=0.5, ls="--")
        ax.axvline(float(jnp.median(gzip_scores)), color="grey", lw=0.5, ls="--")
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.text(xlim[1], ylim[1], "IV (target)", ha="right", va="top", fontsize=9, color="green")
        ax.text(xlim[1], ylim[0], "III chaos",   ha="right", va="bottom", fontsize=9, color="red")
        ax.text(xlim[0], ylim[1], "II ordered",  ha="left",  va="top", fontsize=9, color="orange")
        ax.text(xlim[0], ylim[0], "I fixed",     ha="left",  va="bottom", fontsize=9, color="grey")
    fig.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "probe_requential_vs_gzip_scatter.png")
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)
    print(f"saved {scatter_path}")

    # preq-vs-req scatter to visualize their agreement
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(jnp.asarray(preq_gain), jnp.asarray(req_gain), s=24, alpha=0.7)
    lo = float(jnp.minimum(preq_gain.min(), req_gain.min()))
    hi = float(jnp.maximum(preq_gain.max(), req_gain.max()))
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.5, ls="--")
    ax.set_xlabel("prequential gain (CE on samples)")
    ax.set_ylabel("requential gain (KL distillation)")
    ax.set_title(f"preq vs req at temp=1e-4  (corr={pearson(preq_gain, req_gain):.3f})")
    fig.tight_layout()
    agree_path = os.path.join(OUT_DIR, "probe_preq_vs_requential_scatter.png")
    fig.savefig(agree_path, dpi=140)
    plt.close(fig)
    print(f"saved {agree_path}")

    # Quadrant exemplars: bin rules by (gzip, score) median-split, take the
    # top-K most-extreme rules per quadrant, render their NCA rollouts.
    def render_quadrant_exemplars(score, tag):
        gz_med = float(jnp.median(gzip_scores))
        sc_med = float(jnp.median(score))
        quadrants = {
            "IV_target":  (gzip_scores >= gz_med) & (score >= sc_med),
            "III_chaos":  (gzip_scores >= gz_med) & (score <  sc_med),
            "II_ordered": (gzip_scores <  gz_med) & (score >= sc_med),
            "I_fixed":    (gzip_scores <  gz_med) & (score <  sc_med),
        }
        extremity_score = {
            "IV_target":  lambda idx:  score[idx] + gzip_scores[idx],
            "III_chaos":  lambda idx:  gzip_scores[idx] - score[idx],
            "II_ordered": lambda idx:  score[idx] - gzip_scores[idx],
            "I_fixed":    lambda idx: -(gzip_scores[idx] + score[idx]),
        }
        selected_idx = {}
        for name, mask in quadrants.items():
            idx = jnp.where(mask, size=NUM_RULES, fill_value=-1)[0]
            idx = idx[idx >= 0]
            if idx.shape[0] == 0:
                continue
            order = jnp.argsort(-extremity_score[name](idx))
            k = int(min(EXEMPLARS_PER_QUADRANT, idx.shape[0]))
            selected_idx[name] = [int(idx[order[j]]) for j in range(k)]
        if not selected_idx:
            return

        flat_rules, flat_labels = [], []
        for name, rule_list in selected_idx.items():
            for rank, rule_b in enumerate(rule_list):
                flat_rules.append(rule_b)
                flat_labels.append((name, rank))
        sel_seeds = jnp.stack([seeds[i] for i in flat_rules])
        sims = generate_nca_dataset(
            seed=rng, num_sims=sel_seeds.shape[0],
            grid=GRID, d_state=NUM_COLORS, n_groups=1,
            identity_bias=0.0, temperature=1e-4,
            num_examples=RENDER_STEPS, num_rules=sel_seeds.shape[0],
            dT=1, rule_seeds=sel_seeds,
        )
        nca = NCA(grid_size=GRID, d_state=NUM_COLORS, n_groups=1, temperature=1e-4)

        n_rows = len(flat_rules)
        fig, axes = plt.subplots(n_rows, RENDER_STEPS,
                                 figsize=(1.6 * RENDER_STEPS, 1.8 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
        for row, ((name, rank), rule_b) in enumerate(zip(flat_labels, flat_rules)):
            for t in range(RENDER_STEPS):
                img = nca.render_state(sims[row, t], params=None)
                axes[row, t].imshow(jnp.asarray(img))
                axes[row, t].axis("off")
                if t == 0:
                    axes[row, t].set_ylabel(f"{name} #{rank}", fontsize=10)
            axes[row, 0].set_title(
                f"{name} #{rank}  rule={rule_b}  "
                f"gzip={float(gzip_scores[rule_b]):.2f}  "
                f"req={float(req_gain[rule_b]):.2f}  "
                f"preq={float(preq_gain[rule_b]):.2f}  "
                f"fixed={float(fixed_gain[rule_b]):.2f}",
                loc="left", fontsize=9,
            )
        fig.suptitle(f"quadrants split on gzip × {tag}", fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"probe_{tag}_quadrant_exemplars.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"saved {out_path}")

    render_quadrant_exemplars(req_gain,  "requential")
    render_quadrant_exemplars(preq_gain, "prequential")

    log_path = os.path.join(OUT_DIR, "probe_requential_scores.tsv")
    with open(log_path, "w") as f:
        f.write(
            "idx\tgzip\t"
            "fixed_gain\tfixed_loss\t"
            "preq_gain\tpreq_length\tpreq_floor\tpreq_baseline\t"
            "req_gain\treq_length\treq_floor\treq_baseline\n"
        )
        for i in range(NUM_RULES):
            f.write(
                f"{i}\t{float(gzip_scores[i]):.4f}\t"
                f"{float(fixed_gain[i]):.4f}\t{float(fixed_loss[i]):.4f}\t"
                f"{float(preq_gain[i]):.4f}\t{float(preq_length[i]):.4f}\t"
                f"{float(preq_floor[i]):.4f}\t{float(preq_baseline[i]):.4f}\t"
                f"{float(req_gain[i]):.4f}\t{float(req_length[i]):.4f}\t"
                f"{float(req_floor[i]):.4f}\t{float(req_baseline[i]):.4f}\n"
            )
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
