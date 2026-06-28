"""
Evolution strategy for NCAContinuous, maximizing identity-init preq_length.

Same metric as train_bilevel_jax.py but search via (mu, lambda)
mutation + elitism instead of bilevel SGD. The SGD attempt converged to the
same 'slow-drift collapse' attractor regardless of starting basin; ES doesn't
follow gradients so it can sample disconnected regions of NCA-param space.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/legacy/evolve_preq_only_jax.py
"""
import os
import sys
import time
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from einops import rearrange
from jax.random import split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT, add_repo_root_to_path

add_repo_root_to_path()

from utils.nca import NCAContinuous, _ProbeConv, rollout_simulation

GRID = 16
D_STATE = 3
P_DROP = 0.5
DT = 0.05
ROLLOUT_STEPS = 32
N_IC = 2
PROBE_STEPS = 50
PROBE_LR = 1e-2

POP_SIZE = 64
N_ELITE = 8         # top-K survive unmutated; rest are mutated offspring
N_GENERATIONS = 100
SIGMA_INIT = 0.1    # mutation stddev (weights are O(0.1-0.5))
SIGMA_DECAY = 0.99  # per-generation; sigma is about 0.037 by gen 100
SEED = 0
RENDER_STEPS = 8

OUT_DIR = str(DEMO_OUT)
os.makedirs(OUT_DIR, exist_ok=True)


def identity_probe_params(d_state):
    kernel = jnp.zeros((3, 3, d_state, d_state))
    kernel = kernel.at[1, 1].set(jnp.eye(d_state))
    bias = jnp.zeros(d_state)
    return {"params": {"Conv_0": {"kernel": kernel, "bias": bias}}}


def probe_preq_identity_init(sims, d_state, probe_steps, lr):
    T = sims.shape[1]
    x_flat = rearrange(sims[:, :T - 1], "I T H W D -> (I T) H W D")
    y_flat = rearrange(sims[:, 1:], "I T H W D -> (I T) H W D")
    probe = _ProbeConv(d_total=d_state)
    params = identity_probe_params(d_state)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def forward(p, batch):
        return jax.vmap(probe.apply, in_axes=(None, 0))(p, batch)

    def loss_fn(p):
        pred = forward(p, x_flat)
        return jnp.mean((pred - y_flat) ** 2)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    id_baseline = loss_fn(params)
    (_, _), losses = lax.scan(step, (params, opt_state), None, length=probe_steps)
    mse_floor = losses[-1]
    preq_length = jnp.sum(jnp.maximum(losses - mse_floor, 0.0))
    return preq_length, mse_floor, id_baseline


def rollout_for_params(nca, net_params, rng, n_ic, rollout_steps):
    params = {"net_params": net_params}

    def one_rollout(_rng):
        return rollout_simulation(
            _rng, params, substrate=nca,
            rollout_steps=rollout_steps, time_sampling='video',
            start_step=0, k_steps=1,
        )

    return jax.vmap(one_rollout)(split(rng, n_ic))


def fitness_one(nca, net_params, rng):
    sims = rollout_for_params(nca, net_params, rng, N_IC, ROLLOUT_STEPS)
    preq_length, mse_floor, id_baseline = probe_preq_identity_init(
        sims, D_STATE, PROBE_STEPS, PROBE_LR,
    )
    return preq_length, id_baseline, mse_floor


def mutate_population(pop_params, rng, sigma):
    """Add iid Gaussian noise to every parameter leaf. Each leaf already has
    a leading POP_SIZE axis, so one normal sample of leaf.shape gives each
    individual its own noise."""
    leaves, treedef = jax.tree.flatten(pop_params)
    rngs = split(rng, len(leaves))
    new_leaves = [
        leaf + sigma * jax.random.normal(r, leaf.shape)
        for leaf, r in zip(leaves, rngs)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def render_rollout_grid(sims, nca, t_idx, title, path):
    fig, axes = plt.subplots(1, len(t_idx), figsize=(1.6 * len(t_idx), 1.8))
    for col, t in enumerate(t_idx.tolist()):
        img = nca.render_state(sims[0, t], params=None)
        axes[col].imshow(jnp.asarray(img))
        axes[col].axis("off")
        axes[col].set_title(f"t={t}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    print(f"JAX devices: {jax.devices()}")
    print(
        f"pop={POP_SIZE} elite={N_ELITE} gens={N_GENERATIONS} "
        f"sigma {SIGMA_INIT}->{SIGMA_INIT * SIGMA_DECAY**N_GENERATIONS:.3f}"
    )

    rng = jax.random.PRNGKey(SEED)
    nca = NCAContinuous(grid_size=GRID, d_state=D_STATE, p_drop=P_DROP, dt=DT)

    rng, pop_rng = split(rng)
    pop_init_rngs = split(pop_rng, POP_SIZE)
    pop_params = jax.vmap(lambda r: nca.default_params(r)["net_params"])(pop_init_rngs)

    fitness_jit = jax.jit(jax.vmap(lambda p, r: fitness_one(nca, p, r)))

    sigma = SIGMA_INIT
    history = []
    best_ever = None
    best_ever_score = -jnp.inf
    t0 = time.time()

    for gen in range(N_GENERATIONS):
        rng, eval_rng = split(rng)
        eval_rngs = split(eval_rng, POP_SIZE)
        preq_lens, id_bases, mse_floors = fitness_jit(pop_params, eval_rngs)
        preq_lens.block_until_ready()

        # selection
        order = jnp.argsort(preq_lens)
        elite_idx = order[-N_ELITE:]
        elite_params = jax.tree.map(lambda x: x[elite_idx], pop_params)

        best_idx = elite_idx[-1]
        best_preq = float(preq_lens[best_idx])
        best_id = float(id_bases[best_idx])
        mean_preq = float(preq_lens.mean())
        history.append((gen, best_preq, mean_preq, best_id, sigma))

        if best_preq > best_ever_score:
            best_ever_score = best_preq
            best_ever = jax.tree.map(lambda x: x[best_idx], pop_params)

        if gen % 5 == 0 or gen == N_GENERATIONS - 1:
            print(
                f"gen {gen:3d}  best_preq={best_preq:.3f}  mean_preq={mean_preq:.3f}  "
                f"best_id_base={best_id:.4f}  sigma={sigma:.3f}  ({time.time()-t0:.1f}s)"
            )

        # reproduce: elite survive unmutated; rest are mutated copies of elite (round-robin)
        n_offspring = POP_SIZE - N_ELITE
        src_idx = jnp.tile(jnp.arange(N_ELITE), n_offspring // N_ELITE + 1)[:n_offspring]
        offspring_params = jax.tree.map(lambda x: x[src_idx], elite_params)
        rng, mut_rng = split(rng)
        offspring_params = mutate_population(offspring_params, mut_rng, sigma)

        pop_params = jax.tree.map(
            lambda e, o: jnp.concatenate([e, o], axis=0),
            elite_params, offspring_params,
        )
        sigma *= SIGMA_DECAY

    # snapshot best-ever NCA
    rng, snap_rng = split(rng)
    sims_best = rollout_for_params(nca, best_ever, snap_rng, N_IC, ROLLOUT_STEPS)
    t_idx = jnp.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).astype(jnp.int32)
    render_rollout_grid(
        sims_best, nca, t_idx,
        f"best-ever NCA after {N_GENERATIONS} generations (preq_length={best_ever_score:.3f})",
        os.path.join(OUT_DIR, "evolve_nca_preq_best.png"),
    )

    # training curves
    hist = jnp.array(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist[:, 0], hist[:, 1], label="best in gen", color="tab:blue")
    ax[0].plot(hist[:, 0], hist[:, 2], label="mean", color="tab:gray", alpha=0.6)
    ax[0].set_xlabel("generation"); ax[0].set_ylabel("preq_length")
    ax[0].set_title("preq_length over generations")
    ax[0].legend(fontsize=8)
    ax[1].plot(hist[:, 0], hist[:, 3], label="best id_baseline", color="tab:orange")
    ax[1].set_xlabel("generation"); ax[1].set_ylabel("identity baseline")
    ax[1].set_title("dynamics activity (best)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "evolve_nca_preq_curve.png"), dpi=130)
    plt.close(fig)

    log_path = os.path.join(OUT_DIR, "evolve_nca_preq_log.tsv")
    with open(log_path, "w") as f:
        f.write("gen\tbest_preq\tmean_preq\tbest_id_baseline\tsigma\n")
        for row in history:
            f.write("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                              for x in row) + "\n")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
