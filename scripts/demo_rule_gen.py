"""
Minimal demo: sample NCA rules, filter by gzip complexity, render one rollout.

Run from repo root:
    .venv/Scripts/python.exe scripts/demo_rule_gen.py
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca import NCA, compute_rule_gzip_batch, generate_nca_dataset, generate_rules_batch
from utils.tokenizers import NCA_Tokenizer

GRID = 12
PATCH = 2
NUM_COLORS = 10
N_STEPS = 10
NUM_RULES = 4
THRESHOLD = 0.5
UPPER_BOUND = 1.0
SEED = 0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Sampling {NUM_RULES} rules with gzip complexity in [{THRESHOLD}, {UPPER_BOUND}] ...")

    tokenizer = NCA_Tokenizer(patch=PATCH, num_colors=NUM_COLORS)
    rng = jax.random.PRNGKey(SEED)

    t0 = time.time()
    current_seed = rng
    accepted = []
    total_tested = 0
    iteration = 0
    while sum(a.shape[0] for a in accepted) < NUM_RULES:
        iteration += 1
        seeds = jax.random.split(current_seed, NUM_RULES)
        scores = compute_rule_gzip_batch(
            seeds, tokenizer,
            grid=GRID, d_state=NUM_COLORS, identity_bias=0.0, temperature=1e-4,
            n_steps=N_STEPS, dT=1, start_step=0, mode="gzip",
        )
        idx = jnp.logical_and(scores > THRESHOLD, scores < UPPER_BOUND)
        passed = int(idx.sum())
        total_tested += NUM_RULES
        print(f"  iter {iteration}: tested {NUM_RULES}, accepted {passed} "
              f"(scores={[round(float(s), 3) for s in scores]})")
        if passed > 0:
            remaining = NUM_RULES - sum(a.shape[0] for a in accepted)
            accepted.append(seeds[idx][:remaining])
        current_seed = jax.random.split(current_seed, 1)[0]
    rule_seeds = jnp.concat(accepted, axis=0)
    print(f"\nGot {rule_seeds.shape[0]} rules from {total_tested} candidates "
          f"({rule_seeds.shape[0] / total_tested:.1%} acceptance) in {time.time() - t0:.2f}s")

    print("\nRolling out all sampled rules ...")
    sims = generate_nca_dataset(
        seed=rng,
        num_sims=NUM_RULES,
        grid=GRID,
        d_state=NUM_COLORS,
        n_groups=1,
        identity_bias=0.0,
        temperature=1e-4,
        num_examples=N_STEPS,
        num_rules=NUM_RULES,
        dT=1,
        rule_seeds=rule_seeds,
    )
    print(f"Sims shape (B, T, H, W, G): {sims.shape}")

    nca = NCA(grid_size=GRID, d_state=NUM_COLORS, n_groups=1, temperature=1e-4)

    for b in range(NUM_RULES):
        fig, axes = plt.subplots(1, N_STEPS, figsize=(2 * N_STEPS, 2.2))
        for t in range(N_STEPS):
            img = nca.render_state(sims[b, t], params=None)
            axes[t].imshow(jnp.asarray(img))
            axes[t].set_title(f"t={t}")
            axes[t].axis("off")
        fig.suptitle(f"Rule {b}  seed={list(map(int, rule_seeds[b]))}")
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"rule_{b}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  saved {out_path}")

    tokens, target = tokenizer.encode_task(sims)
    print(f"\nTokenized shape: tokens={tuple(tokens.shape)}, target={tuple(target.shape)}")
    print(f"Vocab size used: num_colors^(patch^2) + 2 = {NUM_COLORS ** (PATCH ** 2) + 2}")
    print(f"First 30 tokens of rule 0: {tokens[0, :30].tolist()}")


if __name__ == "__main__":
    main()
