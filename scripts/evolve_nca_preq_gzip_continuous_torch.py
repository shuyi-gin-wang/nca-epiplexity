"""
PyTorch port of scripts/evolve_nca_preq_gzip_continuous.py.

Same (mu+lambda) ES loop with fitness = preq_length * gzip_band, where
gzip_band = exp(-((gzip_ratio - GZIP_TARGET)^2) / (2 * GZIP_WIDTH^2)) gates
out both collapsed (gzip→0) and chaotic (gzip→1) rollouts. Mirrors the JAX
version's defaults exactly so results are directly comparable.

Run from repo root:
    .venv/Scripts/python.exe scripts/evolve_nca_preq_gzip_continuous_torch.py
"""
import argparse
import gzip
import io
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca_torch import NCANetworkTorch
from scripts.evolve_nca_preq_continuous_torch import (
    init_pop_params,
    mutate_population,
    rollout_one_indiv,
    fitness_one,
    sample_eval_state,
    render_rollout_grid,
)

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
RUN_NAME = "evolve_largepop_torch"

GZIP_TARGET = 0.5
GZIP_WIDTH = 0.2
GZIP_THRESHOLD = 0.3
GZIP_MODE = "threshold"  # "threshold" gates static-only; "band" Gaussian-bands both tails

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)


def gzip_ratio(sims_one: np.ndarray) -> float:
    """sims_one: numpy float array in [0,1]. Returns compressed/original byte ratio.
    Layout (NCHW vs NHWC) doesn't affect the ratio since bytes are the same set."""
    quantized = np.clip(sims_one * 255.0, 0, 255).astype(np.uint8)
    byte_data = quantized.tobytes()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    return len(buf.getvalue()) / len(byte_data)


def fitness_one_with_sims(net_template, params, x0, masks_T, d_state, dt, probe_steps, lr):
    """Same as fitness_one but also returns the rollout so the host can gzip it."""
    sims = rollout_one_indiv(net_template, params, x0, masks_T, dt)
    N_IC_, T, D, H, W = sims.shape
    x = sims[:, : T - 1].reshape(N_IC_ * (T - 1), D, H, W)
    y = sims[:, 1:].reshape(N_IC_ * (T - 1), D, H, W)

    import torch.nn.functional as F
    from torch.func import grad_and_value
    from scripts.evolve_nca_preq_continuous_torch import (
        identity_probe_params, probe_loss, adam_step,
    )

    probe_w, probe_b = identity_probe_params(d_state, x.device)
    m_w = torch.zeros_like(probe_w); v_w = torch.zeros_like(probe_w)
    m_b = torch.zeros_like(probe_b); v_b = torch.zeros_like(probe_b)
    grad_fn = grad_and_value(probe_loss, argnums=(0, 1))
    losses = []
    for step in range(probe_steps):
        (gw, gb), loss = grad_fn(probe_w, probe_b, x, y)
        probe_w, m_w, v_w = adam_step(probe_w, m_w, v_w, gw, step + 1, lr)
        probe_b, m_b, v_b = adam_step(probe_b, m_b, v_b, gb, step + 1, lr)
        losses.append(loss)
    losses_t = torch.stack(losses)
    mse_floor = losses_t[-1]
    initial_loss = losses_t[0]
    preq_length = torch.clamp(losses_t - mse_floor, min=0.0).sum()
    return preq_length, initial_loss, mse_floor, sims, losses_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gzip-mode", type=str, default=GZIP_MODE,
                        choices=["threshold", "band"],
                        help="threshold: fitness = preq * 1[gzip > gzip_threshold] (gate static only). "
                             "band: fitness = preq * Gaussian(gzip; gzip_target, gzip_width) (penalize both tails).")
    parser.add_argument("--gzip-threshold", type=float, default=GZIP_THRESHOLD,
                        help="threshold mode only: rollouts with gzip <= threshold get fitness 0.")
    parser.add_argument("--gzip-target", type=float, default=GZIP_TARGET,
                        help="band mode only: Gaussian center.")
    parser.add_argument("--gzip-width", type=float, default=GZIP_WIDTH,
                        help="band mode only: Gaussian std.")
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--p-drop", type=float, default=P_DROP,
                        help="per-cell Bernoulli update-mask drop probability (default 0.5)")
    parser.add_argument("--sigma-decay", type=float, default=SIGMA_DECAY,
                        help="per-generation multiplicative decay of mutation sigma (default 0.995)")
    args = parser.parse_args()
    gzip_mode = args.gzip_mode
    gzip_threshold = args.gzip_threshold
    gzip_target = args.gzip_target
    gzip_width = args.gzip_width
    pop_size = args.pop_size
    n_generations = args.n_generations
    n_elite = max(1, pop_size * N_ELITE // POP_SIZE)
    seed = args.seed
    p_drop = args.p_drop
    sigma_decay = args.sigma_decay
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}  run_name={args.run_name}")
    if gzip_mode == "threshold":
        fitness_desc = f"fitness = preq_length * 1[gzip > {gzip_threshold}]"
    else:
        fitness_desc = f"fitness = preq_length * exp(-(gzip - {gzip_target})^2 / (2*{gzip_width}^2))"
    print(
        f"pop={pop_size} elite={n_elite} gens={n_generations} seed={seed} p_drop={p_drop} "
        f"sigma {SIGMA_INIT}->{SIGMA_INIT * sigma_decay**n_generations:.3f} (decay={sigma_decay})  "
        f"{fitness_desc}"
    )

    g = torch.Generator(device=device).manual_seed(seed)

    net_template = NCANetworkTorch(d_state=D_STATE).to(device)
    pop_params = init_pop_params(net_template, pop_size, D_STATE, device, g)

    fitness_vmapped = vmap(
        lambda p, x, m: fitness_one_with_sims(
            net_template, p, x, m, D_STATE, DT, PROBE_STEPS, PROBE_LR,
        ),
        in_dims=(0, 0, 0),
    )

    sigma = SIGMA_INIT
    history = []
    best_probe_curves = []  # (n_generations, probe_steps) best individual's probe MSE per step
    mean_probe_curves = []  # (n_generations, probe_steps) population mean probe MSE per step
    best_ever_params = None
    best_ever_combined = -float("inf")
    t0 = time.time()

    for gen in range(n_generations):
        x0, masks = sample_eval_state(pop_size, N_IC, D_STATE, GRID, ROLLOUT_STEPS, p_drop, device, g)
        with torch.no_grad():
            preq_lens, init_losses, mse_floors, sims_all, probe_curves = fitness_vmapped(pop_params, x0, masks)
        if device.type == "cuda":
            torch.cuda.synchronize()

        sims_np = sims_all.detach().cpu().numpy()
        preq_np = preq_lens.detach().cpu().numpy()
        id_np = init_losses.detach().cpu().numpy()
        probe_curves_np = probe_curves.detach().cpu().numpy()  # (pop, probe_steps)

        gzip_scores = np.array([gzip_ratio(sims_np[i]) for i in range(pop_size)])
        if gzip_mode == "threshold":
            gzip_band = (gzip_scores > gzip_threshold).astype(np.float32)
        else:
            gzip_band = np.exp(-((gzip_scores - gzip_target) ** 2) / (2 * gzip_width ** 2))
        combined = preq_np * gzip_band

        order = np.argsort(combined)
        elite_idx = torch.from_numpy(order[-n_elite:].copy()).to(device)
        elite_params = {k: v[elite_idx] for k, v in pop_params.items()}

        best_i = int(order[-1])
        best_combined = float(combined[best_i])
        best_preq = float(preq_np[best_i])
        best_gzip = float(gzip_scores[best_i])
        best_id = float(id_np[best_i])

        best_probe_curves.append(probe_curves_np[best_i].copy())
        mean_probe_curves.append(probe_curves_np.mean(axis=0))

        history.append((
            gen, best_combined, float(combined.mean()),
            best_preq, best_gzip, best_id, sigma,
        ))

        if best_combined > best_ever_combined:
            best_ever_combined = best_combined
            best_ever_params = {k: v[best_i].detach().clone() for k, v in pop_params.items()}

        if gen % 5 == 0 or gen == n_generations - 1:
            print(
                f"gen {gen:3d}  best_comb={best_combined:.4f}  "
                f"(preq={best_preq:.3f} * gzip={best_gzip:.3f})  "
                f"id_base={best_id:.4f}  sigma={sigma:.3f}  ({time.time()-t0:.1f}s)",
                flush=True,
            )

        n_offspring = pop_size - n_elite
        src_idx = torch.arange(n_elite, device=device).repeat(n_offspring // n_elite + 1)[:n_offspring]
        offspring_params = {k: v[src_idx] for k, v in elite_params.items()}
        offspring_params = mutate_population(offspring_params, sigma, g)
        pop_params = {
            k: torch.cat([elite_params[k], offspring_params[k]], dim=0)
            for k in pop_params
        }
        sigma *= sigma_decay

    # snapshot best-ever
    x0_snap, masks_snap = sample_eval_state(1, N_IC, D_STATE, GRID, ROLLOUT_STEPS, P_DROP, device, g)
    best_stack = {k: v.unsqueeze(0) for k, v in best_ever_params.items()}
    with torch.no_grad():
        sims_best_stack = vmap(
            lambda p, x, m: rollout_one_indiv(net_template, p, x, m, DT),
            in_dims=(0, 0, 0),
        )(best_stack, x0_snap, masks_snap)
    sims_best = sims_best_stack[0]
    t_idx = torch.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).long().tolist()
    render_rollout_grid(
        sims_best[0], t_idx,
        f"best-ever NCA after {n_generations} gens (combined={best_ever_combined:.4f})",
        os.path.join(out_dir, "evolve_nca_preq_gzip_best_torch.png"),
    )

    ckpt_path = os.path.join(out_dir, "best_ever_params.pt")
    torch.save(
        {
            "params": {k: v.detach().cpu() for k, v in best_ever_params.items()},
            "d_state": D_STATE,
            "grid": GRID,
            "rollout_steps": ROLLOUT_STEPS,
            "dt": DT,
            "p_drop": p_drop,
            "gzip_mode": gzip_mode,
            "gzip_threshold": gzip_threshold,
            "gzip_target": gzip_target,
            "gzip_width": gzip_width,
            "best_combined": best_ever_combined,
            "seed": seed,
            "n_generations": n_generations,
            "pop_size": pop_size,
        },
        ckpt_path,
    )
    print(f"saved {ckpt_path}")

    hist = np.array(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(hist[:, 0], hist[:, 1], label="best combined", color="tab:purple")
    ax[0].plot(hist[:, 0], hist[:, 2], label="mean combined", color="tab:gray", alpha=0.6)
    if gzip_mode == "threshold":
        ax[0].set_ylabel(f"preq_length × 1[gzip > {gzip_threshold}]")
    else:
        ax[0].set_ylabel("preq_length × gzip_band")
    ax[0].set_xlabel("generation")
    ax[0].set_title(f"combined fitness (torch, gzip_mode={gzip_mode})"); ax[0].legend(fontsize=8)
    ax[1].plot(hist[:, 0], hist[:, 3], label="preq_length", color="tab:blue")
    ax[1].plot(hist[:, 0], hist[:, 4], label="gzip_ratio", color="tab:green")
    ax[1].set_xlabel("generation"); ax[1].set_ylabel("component")
    ax[1].set_title("components of best individual"); ax[1].legend(fontsize=8)
    ax[2].plot(hist[:, 0], hist[:, 5], color="tab:orange")
    ax[2].set_xlabel("generation"); ax[2].set_ylabel("id_baseline")
    ax[2].set_title("dynamics activity (best)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "evolve_nca_preq_gzip_curve_torch.png"), dpi=130)
    plt.close(fig)

    log_path = os.path.join(out_dir, "evolve_nca_preq_gzip_log_torch.tsv")
    with open(log_path, "w") as f:
        f.write("gen\tbest_combined\tmean_combined\tbest_preq\tbest_gzip\tbest_id_base\tsigma\n")
        for row in history:
            f.write("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                              for x in row) + "\n")
    print(f"saved {log_path}")

    # probe learning curves: (n_generations, probe_steps)
    best_curves = np.stack(best_probe_curves, axis=0)
    mean_curves = np.stack(mean_probe_curves, axis=0)
    np.save(os.path.join(out_dir, "probe_curves_best.npy"), best_curves)
    np.save(os.path.join(out_dir, "probe_curves_mean.npy"), mean_curves)

    # sampled snapshots: pick ~6 evenly spaced generations, overlay best + mean curves.
    n_snap = min(6, n_generations)
    snap_gens = np.linspace(0, n_generations - 1, n_snap).astype(int)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, g_idx in enumerate(snap_gens):
        color = cmap(j / max(1, n_snap - 1))
        ax[0].plot(best_curves[g_idx], color=color, label=f"gen {g_idx}")
        ax[1].plot(mean_curves[g_idx], color=color, label=f"gen {g_idx}")
    for a, title in zip(ax, ("best-of-gen probe curve", "population-mean probe curve")):
        a.set_xlabel("probe SGD step"); a.set_ylabel("probe MSE")
        a.set_title(title); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_curves_snapshots.png"), dpi=130)
    plt.close(fig)
    print(f"saved probe-curve arrays + snapshots in {out_dir}")


if __name__ == "__main__":
    main()
