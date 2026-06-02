"""
Evolution strategy maximizing preq_length * gzip_ratio on NCAContinuous.

Builds on scripts/evolve_nca_preq_continuous.py, which converged to a single
'slow drift collapse' attractor at preq_length≈0.294. Adding gzip complexity
as a multiplicative gate should kill that attractor: a collapsed/static rollout
compresses to ~0, multiplying preq_length down to 0. Combined signal selects
for both "Conv3x3-reducible structure" (preq) and "uncompressible / nontrivial
spatiotemporal pattern" (gzip).

This is the class-IV filter sketched in utils/nca.py:519-525, ported to the
continuous substrate.

Run from repo root:
    .venv/Scripts/python.exe scripts/evolve_nca_preq_gzip_continuous.py
"""
import gzip
import io
import os
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from einops import rearrange
from jax.random import split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca import NCAContinuous, _ProbeConv, rollout_simulation

GRID = 16
D_STATE = 3
P_DROP = 0.5
DT = 0.05
ROLLOUT_STEPS = 32
N_IC = 2
PROBE_STEPS = 50
PROBE_LR = 1e-2

POP_SIZE = 128
N_ELITE = 16
N_GENERATIONS = 600
SIGMA_INIT = 0.1
SIGMA_DECAY = 0.995
SEED = 0
RENDER_STEPS = 8
RUN_NAME = "evolve_largepop"

# class-IV band: penalize both pure-order (gzip→0) and pure-chaos (gzip→1).
# Gaussian centered on the target with width 0.2 → contributes meaningfully in [~0.3, ~0.7].
GZIP_TARGET = 0.5   # back to default
GZIP_WIDTH = 0.2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out", RUN_NAME)
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


def jax_fitness_one(nca, net_params, rng):
    """JAX-side: rollout + probe. Returns sims so host can gzip them."""
    sims = rollout_for_params(nca, net_params, rng, N_IC, ROLLOUT_STEPS)
    preq_length, mse_floor, id_baseline = probe_preq_identity_init(
        sims, D_STATE, PROBE_STEPS, PROBE_LR,
    )
    return preq_length, id_baseline, mse_floor, sims


def gzip_ratio(sims_one: np.ndarray) -> float:
    """sims_one: numpy float array in [0,1]. Returns compressed/original byte ratio.
    Higher = more complex / less compressible. Matches utils/nca.py:gzip_complexity.
    """
    quantized = np.clip(sims_one * 255.0, 0, 255).astype(np.uint8)
    byte_data = quantized.tobytes()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    return len(buf.getvalue()) / len(byte_data)


def mutate_population(pop_params, rng, sigma):
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
        f"sigma {SIGMA_INIT}->{SIGMA_INIT * SIGMA_DECAY**N_GENERATIONS:.3f}  "
        f"fitness = preq_length * exp(-(gzip - {GZIP_TARGET})^2 / (2*{GZIP_WIDTH}^2))"
    )

    rng = jax.random.PRNGKey(SEED)
    nca = NCAContinuous(grid_size=GRID, d_state=D_STATE, p_drop=P_DROP, dt=DT)

    rng, pop_rng = split(rng)
    pop_init_rngs = split(pop_rng, POP_SIZE)
    pop_params = jax.vmap(lambda r: nca.default_params(r)["net_params"])(pop_init_rngs)

    fitness_jit = jax.jit(jax.vmap(lambda p, r: jax_fitness_one(nca, p, r)))

    sigma = SIGMA_INIT
    history = []
    best_ever = None
    best_ever_combined = -np.inf
    t0 = time.time()

    for gen in range(N_GENERATIONS):
        rng, eval_rng = split(rng)
        eval_rngs = split(eval_rng, POP_SIZE)
        preq_lens, id_bases, mse_floors, sims_all = fitness_jit(pop_params, eval_rngs)
        sims_np = np.asarray(sims_all)  # (POP, n_ic, T, H, W, D)
        preq_np = np.asarray(preq_lens)
        id_np = np.asarray(id_bases)

        gzip_scores = np.array([gzip_ratio(sims_np[i]) for i in range(POP_SIZE)])
        gzip_band = np.exp(-((gzip_scores - GZIP_TARGET) ** 2) / (2 * GZIP_WIDTH ** 2))
        combined = preq_np * gzip_band

        # selection
        order = np.argsort(combined)
        elite_idx = jnp.asarray(order[-N_ELITE:])
        elite_params = jax.tree.map(lambda x: x[elite_idx], pop_params)

        best_i = int(order[-1])
        best_combined = float(combined[best_i])
        best_preq = float(preq_np[best_i])
        best_gzip = float(gzip_scores[best_i])
        best_id = float(id_np[best_i])

        history.append((
            gen, best_combined, float(combined.mean()),
            best_preq, best_gzip, best_id, sigma,
        ))

        if best_combined > best_ever_combined:
            best_ever_combined = best_combined
            best_ever = jax.tree.map(lambda x: x[best_i], pop_params)

        if gen % 5 == 0 or gen == N_GENERATIONS - 1:
            print(
                f"gen {gen:3d}  best_comb={best_combined:.4f}  "
                f"(preq={best_preq:.3f} * gzip={best_gzip:.3f})  "
                f"id_base={best_id:.4f}  sigma={sigma:.3f}  ({time.time()-t0:.1f}s)"
            )

        # reproduce
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

    # snapshot best-ever
    rng, snap_rng = split(rng)
    sims_best = rollout_for_params(nca, best_ever, snap_rng, N_IC, ROLLOUT_STEPS)
    t_idx = jnp.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).astype(jnp.int32)
    render_rollout_grid(
        sims_best, nca, t_idx,
        f"best-ever NCA after {N_GENERATIONS} gens (combined={best_ever_combined:.4f})",
        os.path.join(OUT_DIR, "evolve_nca_preq_gzip_best.png"),
    )

    hist = np.array(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(hist[:, 0], hist[:, 1], label="best combined", color="tab:purple")
    ax[0].plot(hist[:, 0], hist[:, 2], label="mean combined", color="tab:gray", alpha=0.6)
    ax[0].set_xlabel("generation"); ax[0].set_ylabel("preq_length × gzip_ratio")
    ax[0].set_title("combined fitness"); ax[0].legend(fontsize=8)
    ax[1].plot(hist[:, 0], hist[:, 3], label="preq_length", color="tab:blue")
    ax[1].plot(hist[:, 0], hist[:, 4], label="gzip_ratio", color="tab:green")
    ax[1].set_xlabel("generation"); ax[1].set_ylabel("component")
    ax[1].set_title("components of best individual"); ax[1].legend(fontsize=8)
    ax[2].plot(hist[:, 0], hist[:, 5], color="tab:orange")
    ax[2].set_xlabel("generation"); ax[2].set_ylabel("id_baseline")
    ax[2].set_title("dynamics activity (best)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "evolve_nca_preq_gzip_curve.png"), dpi=130)
    plt.close(fig)

    log_path = os.path.join(OUT_DIR, "evolve_nca_preq_gzip_log.tsv")
    with open(log_path, "w") as f:
        f.write("gen\tbest_combined\tmean_combined\tbest_preq\tbest_gzip\tbest_id_base\tsigma\n")
        for row in history:
            f.write("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                              for x in row) + "\n")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
