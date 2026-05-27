"""
Companion to demo_probe_requential.py: sweep NCA sampling temperature and show
where the prequential score (CE on samples) collapses while the requential
score (KL against the teacher distribution) keeps a learnability signal.

Run from repo root:
    .venv/Scripts/python.exe scripts/demo_probe_requential_temp.py
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca import (
    compute_rule_probe_preq_batch,
    compute_rule_probe_requential_batch,
)

GRID = 12
NUM_COLORS = 10
NUM_RULES = 64
N_IC = 4
ROLLOUT_STEPS = 24
PROBE_STEPS = 200
TEMPS = (1e-4, 1e-2, 1e-1, 3e-1, 1.0)
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def pearson(a, b):
    a = jnp.asarray(a); b = jnp.asarray(b)
    am = a - a.mean(); bm = b - b.mean()
    return float((am * bm).sum() / (jnp.sqrt((am * am).sum() * (bm * bm).sum()) + 1e-12))


def main():
    print(f"JAX devices: {jax.devices()}")
    seeds = jax.random.split(jax.random.PRNGKey(SEED), NUM_RULES)

    rows = []
    fig, axes = plt.subplots(1, len(TEMPS), figsize=(4 * len(TEMPS), 4), sharex=False, sharey=False)
    for i, T in enumerate(TEMPS):
        t0 = time.time()
        preq_gain, *_ = compute_rule_probe_preq_batch(
            seeds, grid=GRID, d_state=NUM_COLORS, n_groups=1,
            identity_bias=0.0, temperature=float(T),
            n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
            probe_steps=PROBE_STEPS, lr=1e-2,
        )
        req_gain, *_ = compute_rule_probe_requential_batch(
            seeds, grid=GRID, d_state=NUM_COLORS, n_groups=1,
            identity_bias=0.0, temperature=float(T),
            n_ic=N_IC, rollout_steps=ROLLOUT_STEPS,
            probe_steps=PROBE_STEPS, lr=1e-2,
        )
        corr = pearson(preq_gain, req_gain)
        rows.append((float(T), float(preq_gain.min()), float(preq_gain.max()),
                     float(req_gain.min()), float(req_gain.max()), corr))
        print(f"T={T:.0e}  preq=[{float(preq_gain.min()):.3f},{float(preq_gain.max()):.3f}]  "
              f"req=[{float(req_gain.min()):.3f},{float(req_gain.max()):.3f}]  "
              f"corr={corr:.4f}  ({time.time() - t0:.1f}s)")

        ax = axes[i]
        ax.scatter(jnp.asarray(preq_gain), jnp.asarray(req_gain), s=20, alpha=0.7)
        lo = min(float(preq_gain.min()), float(req_gain.min()))
        hi = max(float(preq_gain.max()), float(req_gain.max()))
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.5, ls="--")
        ax.set_xlabel("prequential gain (CE on samples)")
        if i == 0:
            ax.set_ylabel("requential gain (KL distillation)")
        ax.set_title(f"T={T:g}    corr={corr:.3f}", fontsize=10)
    fig.suptitle("preq vs requential probe gain across teacher sampling temperatures",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "probe_requential_vs_preq_temp_sweep.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")

    log_path = os.path.join(OUT_DIR, "probe_requential_temp_sweep.tsv")
    with open(log_path, "w") as f:
        f.write("temperature\tpreq_min\tpreq_max\treq_min\treq_max\tcorr\n")
        for r in rows:
            f.write(f"{r[0]:.4g}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]:.4f}\t{r[5]:.4f}\n")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
