"""
Train an NCAContinuous to maximize the prequential probe length.

Outer loop: Adam on nca.net_params.
Inner loop: existing differentiable preq probe (50-100 SGD steps via lax.scan).
Outer loss: -preq_length  (we want the probe to take many steps to fit).

Small grid (16) so the whole bilevel scan fits comfortably on CPU.
This is a sketch to check whether the gradient signal actually grows
preq_length over outer steps.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/legacy/train_bilevel_jax.py
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

from utils.nca import (
    NCAContinuous,
    _ProbeConv,
    rollout_simulation,
)

GRID = 16
D_STATE = 3
P_DROP = 0.5
DT = 0.05            # bump from 0.01 so dynamics move within a short rollout
ROLLOUT_STEPS = 32   # full BPTT through the rollout (no truncation; saddle issue)
N_IC = 2
PROBE_STEPS = 50
PROBE_LR = 1e-2
OUTER_STEPS = 150
OUTER_LR = 3e-3
N_CANDIDATES = 64    # search this many random NCAs, pick top preq_length to start
SEED = 0
RENDER_STEPS = 8

OUT_DIR = str(DEMO_OUT)
os.makedirs(OUT_DIR, exist_ok=True)


def rollout_for_params(nca, net_params, rng, n_ic, rollout_steps):
    """Full rollout, differentiable in net_params. Used for visualization."""
    params = {"net_params": net_params}

    def one_rollout(_rng):
        return rollout_simulation(
            _rng, params, substrate=nca,
            rollout_steps=rollout_steps, time_sampling='video',
            start_step=0, k_steps=1,
        )

    return jax.vmap(one_rollout)(split(rng, n_ic))


def rollout_truncated(nca, net_params, rng, n_ic, warmup_steps, bptt_steps):
    """Truncated-BPTT rollout. Warmup runs forward with stop_gradient on state,
    so the probe + optimizer only see/feel the steady-state segment.

    Returns sims: (n_ic, bptt_steps + 1, H, W, D), where frame 0 is the
    (detached) post-warmup state.
    """
    params = {"net_params": net_params}

    def one_rollout(ic_rng):
        init_rng, warm_rng, bptt_rng = split(ic_rng, 3)
        s0 = nca.init_state(init_rng, params)

        def warm_step(state, r):
            return nca.step_state(r, state, params), None
        state_warm, _ = lax.scan(warm_step, s0, split(warm_rng, warmup_steps))
        state_warm = lax.stop_gradient(state_warm)

        def bptt_step(state, r):
            ns = nca.step_state(r, state, params)
            return ns, ns
        _, frames = lax.scan(bptt_step, state_warm, split(bptt_rng, bptt_steps))
        return jnp.concatenate([state_warm[None], frames], axis=0)

    return jax.vmap(one_rollout)(split(rng, n_ic))


def identity_probe_params(d_state):
    """Conv3x3 params that implement identity: output[i,j] = input[i,j].

    With this init the probe's initial loss equals identity-baseline MSE,
    so preq_length only accumulates when the probe finds structure strictly
    better than copy-paste. Static, collapsed, and chaotic dynamics give
    preq_length is about 0; only learnable spatial-local rules contribute.
    """
    kernel = jnp.zeros((3, 3, d_state, d_state))
    kernel = kernel.at[1, 1].set(jnp.eye(d_state))
    bias = jnp.zeros(d_state)
    return {"params": {"Conv_0": {"kernel": kernel, "bias": bias}}}


def probe_preq_identity_init(sims, d_state, probe_steps, lr):
    """Same as _train_probe_one_rule_preq_continuous but probe starts as identity."""
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

    identity_baseline = loss_fn(params)  # initial loss = MSE(s_{t+1}, s_t)
    (_, _), losses = lax.scan(step, (params, opt_state), None, length=probe_steps)
    mse_floor = losses[-1]
    preq_length = jnp.sum(jnp.maximum(losses - mse_floor, 0.0))

    mean_y = y_flat.mean(axis=(0, 1, 2), keepdims=True)
    marg_baseline = jnp.mean((y_flat - mean_y) ** 2)
    return preq_length, mse_floor, identity_baseline, marg_baseline


def score_rule(nca, net_params, rng):
    """preq_length + id_baseline for a fixed NCA (no training). Used for candidate search."""
    sims = rollout_for_params(nca, net_params, rng, N_IC, ROLLOUT_STEPS)
    preq_length, _, identity_baseline, _ = probe_preq_identity_init(
        sims, D_STATE, PROBE_STEPS, PROBE_LR,
    )
    return preq_length, identity_baseline


def make_grad_fn(nca):
    def outer_loss(net_params, rollout_rng, _probe_rng_unused):
        sims = rollout_for_params(nca, net_params, rollout_rng, N_IC, ROLLOUT_STEPS)
        preq_length, mse_floor, identity_baseline, marg_baseline = probe_preq_identity_init(
            sims, D_STATE, PROBE_STEPS, PROBE_LR,
        )
        loss = -preq_length
        return loss, (preq_length, mse_floor, identity_baseline, marg_baseline, sims)

    return jax.jit(jax.value_and_grad(outer_loss, has_aux=True))


def render_rollout_grid(sims, nca, t_idx, title, path):
    """sims: (n_ic, T, H, W, D). Render IC=0 across t_idx."""
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
        f"grid={GRID} d_state={D_STATE} dt={DT} T={ROLLOUT_STEPS} n_ic={N_IC} "
        f"probe_steps={PROBE_STEPS} outer_steps={OUTER_STEPS} outer_lr={OUTER_LR} "
        f"n_candidates={N_CANDIDATES} (identity-init probe, full BPTT, candidate search)"
    )

    rng = jax.random.PRNGKey(SEED)
    nca = NCAContinuous(grid_size=GRID, d_state=D_STATE, p_drop=P_DROP, dt=DT)

    # === candidate search ===
    print(f"\nScoring {N_CANDIDATES} random NCAs to find a nontrivial starting point...")
    t_search = time.time()
    rng, cand_init_rng, cand_score_rng = split(rng, 3)
    cand_init_rngs = split(cand_init_rng, N_CANDIDATES)
    cand_score_rngs = split(cand_score_rng, N_CANDIDATES)
    cand_params = jax.vmap(lambda r: nca.default_params(r)["net_params"])(cand_init_rngs)

    score_jit = jax.jit(jax.vmap(lambda p, r: score_rule(nca, p, r)))
    preq_scores, id_bases = score_jit(cand_params, cand_score_rngs)
    preq_scores.block_until_ready()
    best_idx = int(jnp.argmax(preq_scores))
    print(
        f"  candidate preq_length: min={float(preq_scores.min()):.3f} "
        f"median={float(jnp.median(preq_scores)):.3f} max={float(preq_scores.max()):.3f}"
    )
    print(
        f"  candidate id_baseline: min={float(id_bases.min()):.4f} "
        f"median={float(jnp.median(id_bases)):.4f} max={float(id_bases.max()):.4f}"
    )
    print(
        f"  picked rule {best_idx}: preq_length={float(preq_scores[best_idx]):.3f}, "
        f"id_baseline={float(id_bases[best_idx]):.4f}  (search: {time.time()-t_search:.1f}s)\n"
    )
    net_params = jax.tree.map(lambda x: x[best_idx], cand_params)

    optimizer = optax.adam(OUTER_LR)
    opt_state = optimizer.init(net_params)

    grad_fn = make_grad_fn(nca)

    # initial rollout for "before" snapshot
    rng, snap_rng = split(rng)
    sims_init = rollout_for_params(nca, net_params, snap_rng, N_IC, ROLLOUT_STEPS)
    t_idx = jnp.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).astype(jnp.int32)
    render_rollout_grid(
        sims_init, nca, t_idx,
        f"before training (step 0)",
        os.path.join(OUT_DIR, "train_nca_preq_before.png"),
    )

    log = []
    t0 = time.time()
    for step in range(OUTER_STEPS):
        rng, r1, r2 = split(rng, 3)
        (loss_val, aux), grads = grad_fn(net_params, r1, r2)
        preq_length, mse_floor, identity_baseline, marg_baseline, _ = aux

        updates, opt_state = optimizer.update(grads, opt_state)
        net_params = optax.apply_updates(net_params, updates)

        log.append((step, float(preq_length), float(mse_floor),
                    float(identity_baseline), float(marg_baseline)))
        if step % 10 == 0 or step == OUTER_STEPS - 1:
            print(
                f"step {step:4d}  preq_len={float(preq_length):.3f}  "
                f"floor={float(mse_floor):.4f}  "
                f"id_base={float(identity_baseline):.4f}  "
                f"marg_base={float(marg_baseline):.4f}  "
                f"({time.time() - t0:.1f}s)"
            )

    # final rollout for "after" snapshot (fresh ICs)
    rng, snap_rng = split(rng)
    sims_final = rollout_for_params(nca, net_params, snap_rng, N_IC, ROLLOUT_STEPS)
    render_rollout_grid(
        sims_final, nca, t_idx,
        f"after {OUTER_STEPS} outer steps",
        os.path.join(OUT_DIR, "train_nca_preq_after.png"),
    )

    # training curves
    log_arr = jnp.array(log)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(log_arr[:, 0], log_arr[:, 1], label="preq_length")
    ax[0].set_xlabel("outer step"); ax[0].set_ylabel("preq_length")
    ax[0].set_title("epiplexity (objective)")
    ax[1].plot(log_arr[:, 0], log_arr[:, 2], label="mse_floor", color="tab:orange")
    ax[1].plot(log_arr[:, 0], log_arr[:, 3], label="identity_baseline", color="tab:blue")
    ax[1].plot(log_arr[:, 0], log_arr[:, 4], label="marg_baseline", color="tab:gray", ls="--")
    ax[1].set_xlabel("outer step"); ax[1].set_ylabel("MSE")
    ax[1].set_title("probe floor vs identity / marginal baselines")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, "train_nca_preq_curve.png")
    fig.savefig(curve_path, dpi=130)
    plt.close(fig)
    print(f"saved {curve_path}")

    log_path = os.path.join(OUT_DIR, "train_nca_preq_log.tsv")
    with open(log_path, "w") as f:
        f.write("step\tpreq_length\tmse_floor\tidentity_baseline\tmarg_baseline\n")
        for row in log:
            f.write("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                              for x in row) + "\n")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
