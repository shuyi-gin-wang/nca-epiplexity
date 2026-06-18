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
import torch.nn.functional as F
from torch.func import grad_and_value, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca_torch import NCANetworkTorch
from utils.probes_torch import PROBE_REGISTRY, make_probe, probe_param_count
from scripts.evolve_nca_preq_continuous_torch import (
    init_pop_params,
    mutate_population,
    rollout_one_indiv,
    adam_step,
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

PROBE_ARCH = "linear"
PROBE_HIDDEN = 0  # 0 = use arch default
PROBE_HORIZON = 1
HORIZON_MODE = "autoregressive"  # autoregressive | direct | multi
MULTI_KS = (1, 2, 4, 8, 16)

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


def _build_pairs(sims, mode, K, multi_ks):
    """Returns (x, y_or_ys, ks_used).

    - autoregressive / direct: x=(B,D,H,W) starts, y=(B,D,H,W) targets at +K.
    - multi: x starts, ys is list of (B,D,H,W) targets at +k for each k in ks_used.
      ks_used is the multi_ks values that fit in the rollout window.
    """
    N_IC_, T, D, H, W = sims.shape
    if mode in ("autoregressive", "direct"):
        Kc = max(1, min(K, T - 1))
        x = sims[:, : T - Kc].reshape(N_IC_ * (T - Kc), D, H, W)
        y = sims[:, Kc:].reshape(N_IC_ * (T - Kc), D, H, W)
        return x, y, (Kc,)
    # multi
    ks_used = tuple(k for k in multi_ks if k <= T - 1)
    Kmax = max(ks_used)
    x = sims[:, : T - Kmax].reshape(N_IC_ * (T - Kmax), D, H, W)
    ys = [sims[:, k : T - Kmax + k].reshape(N_IC_ * (T - Kmax), D, H, W) for k in ks_used]
    return x, ys, ks_used


def _make_loss_fn(probe_fwd, mode, K, ks_used):
    """Returns loss_fn(probe_params_tuple, x, y_or_ys) -> scalar."""
    if mode == "direct":
        def loss_fn(params, x, y):
            pred = probe_fwd(params, x)
            return ((pred - y) ** 2).mean()
        return loss_fn
    if mode == "autoregressive":
        def loss_fn(params, x, y):
            cur = x
            for _ in range(K):
                cur = probe_fwd(params, cur)
            return ((cur - y) ** 2).mean()
        return loss_fn
    # multi: ys is a tuple/list of targets, aligned with ks_used.
    ks_set = set(ks_used)
    kmax = max(ks_used)
    ks_list = list(ks_used)
    def loss_fn(params, x, ys):
        cur = x
        total = x.new_zeros(())
        for k in range(1, kmax + 1):
            cur = probe_fwd(params, cur)
            if k in ks_set:
                idx = ks_list.index(k)
                total = total + ((cur - ys[idx]) ** 2).mean()
        return total / len(ks_list)
    return loss_fn


def fitness_one_with_sims(net_template, params, x0, masks_T,
                          probe_init_params, probe_fwd,
                          mode, K, multi_ks,
                          dt, probe_steps, lr):
    """Roll out the NCA, train the probe to predict next-state (mode-dependent
    horizon), and return (preq_length, initial_loss, mse_floor, sims, loss_curve).

    probe_init_params: tuple of probe param tensors, shared across the population
        (broadcast via vmap in_dims=None) so the prequential score is measured
        from the same probe baseline for every individual within a generation.
    probe_fwd: callable(params_tuple, x_nchw) -> next-state prediction.
    mode: 'autoregressive' applies probe K times; 'direct' is one forward pass
        targeted at state[t+K]; 'multi' sums losses across multi_ks horizons.
    """
    sims = rollout_one_indiv(net_template, params, x0, masks_T, dt)
    x, y_or_ys, ks_used = _build_pairs(sims, mode, K, multi_ks)

    loss_fn = _make_loss_fn(probe_fwd, mode, ks_used[0] if mode != "multi" else K, ks_used)
    grad_fn = grad_and_value(loss_fn, argnums=0)

    probe_params = tuple(probe_init_params)
    m_state = tuple(torch.zeros_like(p) for p in probe_params)
    v_state = tuple(torch.zeros_like(p) for p in probe_params)

    losses = []
    for step in range(probe_steps):
        grads, loss = grad_fn(probe_params, x, y_or_ys)
        new_params, new_m, new_v = [], [], []
        for p, m, v, g in zip(probe_params, m_state, v_state, grads):
            p_new, m_new, v_new = adam_step(p, m, v, g, step + 1, lr)
            new_params.append(p_new); new_m.append(m_new); new_v.append(v_new)
        probe_params = tuple(new_params)
        m_state = tuple(new_m); v_state = tuple(new_v)
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
    parser.add_argument("--probe-arch", type=str, default=PROBE_ARCH,
                        choices=list(PROBE_REGISTRY.keys()),
                        help="student probe architecture (complexity ladder).")
    parser.add_argument("--probe-hidden", type=int, default=PROBE_HIDDEN,
                        help="hidden width for non-linear probes; 0 uses arch default.")
    parser.add_argument("--probe-horizon", type=int, default=PROBE_HORIZON,
                        help="horizon K for autoregressive/direct modes.")
    parser.add_argument("--horizon-mode", type=str, default=HORIZON_MODE,
                        choices=["autoregressive", "direct", "multi"],
                        help="autoregressive: apply probe K times. "
                             "direct: probe predicts state[t+K] in one pass. "
                             "multi: sum losses across multi_ks horizons.")
    parser.add_argument("--multi-ks", type=str, default=",".join(str(k) for k in MULTI_KS),
                        help="comma-separated K values used by --horizon-mode multi.")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="If >0, save population's best-of-gen params to checkpoints/gen_<N>.pt "
                             "every N generations. 0 disables.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a best_ever_params.pt. If set, seeds the population by "
                             "replicating those params across pop_size (one preserved as elite, "
                             "the rest perturbed by sigma_init noise). Resets sigma to SIGMA_INIT "
                             "for fresh exploration. best_ever_combined is also seeded so we don't "
                             "regress.")
    parser.add_argument("--sigma-init", type=float, default=SIGMA_INIT,
                        help="Starting mutation sigma. Default 0.1 matches the schedule used by "
                             "fresh runs; lower this when resuming if you want to fine-tune around "
                             "the loaded best rather than re-explore.")
    parser.add_argument("--probe-lr", type=float, default=PROBE_LR,
                        help="Adam learning rate for the inner probe-training loop. Default 1e-2 "
                             "suits linear/MLP probes; the transformer probe is more stable at "
                             "3e-3 (its probe-MSE curve descends monotonically rather than "
                             "overshooting in the first few steps).")
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
    probe_arch = args.probe_arch
    probe_hidden = args.probe_hidden
    probe_horizon = max(1, args.probe_horizon)
    horizon_mode = args.horizon_mode
    multi_ks = tuple(int(x) for x in args.multi_ks.split(",") if x.strip())
    checkpoint_every = max(0, args.checkpoint_every)
    sigma_init = args.sigma_init
    probe_lr = args.probe_lr
    resume_from = args.resume_from
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints") if checkpoint_every > 0 else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}  run_name={args.run_name}")
    if gzip_mode == "threshold":
        fitness_desc = f"fitness = preq_length * 1[gzip > {gzip_threshold}]"
    else:
        fitness_desc = f"fitness = preq_length * exp(-(gzip - {gzip_target})^2 / (2*{gzip_width}^2))"
    pcount = probe_param_count(probe_arch, D_STATE, probe_hidden)
    if horizon_mode == "multi":
        horizon_desc = f"multi(ks={list(multi_ks)})"
    else:
        horizon_desc = f"{horizon_mode}(K={probe_horizon})"
    print(
        f"pop={pop_size} elite={n_elite} gens={n_generations} seed={seed} p_drop={p_drop} "
        f"sigma {sigma_init}->{sigma_init * sigma_decay**n_generations:.3f} (decay={sigma_decay})  "
        f"{fitness_desc}  "
        f"probe={probe_arch}(h={probe_hidden if probe_hidden else 'auto'},~{pcount}p,lr={probe_lr:g}) {horizon_desc}"
    )

    g = torch.Generator(device=device).manual_seed(seed)

    net_template = NCANetworkTorch(d_state=D_STATE).to(device)
    if resume_from is not None:
        seed_ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        seed_params = {k: v.to(device) for k, v in seed_ckpt["params"].items()}
        print(
            f"resume: seeded population from {resume_from} "
            f"(prev best_combined={seed_ckpt.get('best_combined', float('nan')):.4f}, "
            f"prev gens={seed_ckpt.get('n_generations', '?')})"
        )
        # Replicate the loaded params across the population, then add
        # sigma_init noise to all but the first slot (which we keep as the
        # untouched elite seed). The standard ES selection in gen 0 will then
        # re-elect this exact copy if it's still best.
        pop_params = {
            k: v.unsqueeze(0).expand(pop_size, *v.shape).clone()
            for k, v in seed_params.items()
        }
        with torch.no_grad():
            for k, v in pop_params.items():
                noise = torch.randn(v.shape, generator=g, device=device) * sigma_init
                noise[0].zero_()  # preserve slot 0 as the exact elite seed
                pop_params[k] = v + noise
    else:
        pop_params = init_pop_params(net_template, pop_size, D_STATE, device, g)

    probe_fwd = PROBE_REGISTRY[probe_arch]["forward"]

    # vmap over (pop_params, x0, masks); probe init is shared (in_dims=None).
    fitness_vmapped = vmap(
        lambda p, x, m, probe_p: fitness_one_with_sims(
            net_template, p, x, m,
            probe_p, probe_fwd,
            horizon_mode, probe_horizon, multi_ks,
            DT, PROBE_STEPS, probe_lr,
        ),
        in_dims=(0, 0, 0, None),
    )

    sigma = sigma_init
    history = []
    best_probe_curves = []  # (n_generations, probe_steps) best individual's probe MSE per step
    mean_probe_curves = []  # (n_generations, probe_steps) population mean probe MSE per step
    if resume_from is not None:
        # Seed best_ever with the loaded checkpoint so a regression in early
        # gens doesn't overwrite the saved-from-prior-run achievement.
        best_ever_params = {k: v[0].detach().clone() for k, v in pop_params.items()}
        best_ever_combined = float(seed_ckpt.get("best_combined", -float("inf")))
    else:
        best_ever_params = None
        best_ever_combined = -float("inf")
    t0 = time.time()

    for gen in range(n_generations):
        x0, masks = sample_eval_state(pop_size, N_IC, D_STATE, GRID, ROLLOUT_STEPS, p_drop, device, g)
        probe_init_params, _ = make_probe(probe_arch, D_STATE, probe_hidden, device, g)
        with torch.no_grad():
            preq_lens, init_losses, mse_floors, sims_all, probe_curves = fitness_vmapped(
                pop_params, x0, masks, probe_init_params,
            )
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

        if ckpt_dir is not None and ((gen + 1) % checkpoint_every == 0 or gen == n_generations - 1):
            ckpt_gen_path = os.path.join(ckpt_dir, f"gen_{gen+1:05d}.pt")
            torch.save(
                {
                    "gen": gen + 1,
                    "best_in_gen_params": {k: v[best_i].detach().cpu() for k, v in pop_params.items()},
                    "best_ever_params": {k: v.detach().cpu() for k, v in best_ever_params.items()},
                    "best_in_gen_combined": best_combined,
                    "best_ever_combined": best_ever_combined,
                    "best_in_gen_preq": best_preq,
                    "best_in_gen_gzip": best_gzip,
                    "sigma": sigma,
                },
                ckpt_gen_path,
            )

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
            "probe_arch": probe_arch,
            "probe_hidden": probe_hidden,
            "probe_lr": probe_lr,
            "probe_horizon": probe_horizon,
            "horizon_mode": horizon_mode,
            "multi_ks": list(multi_ks),
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
