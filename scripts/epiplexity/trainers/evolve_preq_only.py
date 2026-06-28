"""
PyTorch port of scripts/epiplexity/legacy/evolve_preq_only_jax.py.

Same (mu+lambda) ES loop, same fitness (identity-init preq_length on a
Conv3x3 probe), same defaults - just torch.func.vmap over the population
instead of jax.vmap, so it runs on CUDA on Windows without WSL.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/trainers/evolve_preq_only.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT, add_repo_root_to_path

add_repo_root_to_path()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad_and_value, vmap

from utils.nca_torch import NCANetworkTorch, _wrap_pad, describe_torch_device, resolve_torch_device

GRID = 16
D_STATE = 3
P_DROP = 0.5
DT = 0.05
ROLLOUT_STEPS = 32
N_IC = 2
PROBE_STEPS = 50
PROBE_LR = 1e-2

POP_SIZE = 64
N_ELITE = 8
N_GENERATIONS = 100
SIGMA_INIT = 0.1
SIGMA_DECAY = 0.99
SEED = 0
RENDER_STEPS = 8
PROBE_HIDDEN = 16


def init_pop_params(net_template, pop_size, d_state, device, generator):
    """Sample POP independent NCANetworks and stack their params with a leading
    POP axis. Returns dict[name -> (POP, *param_shape)]."""
    param_dicts = []
    for _ in range(pop_size):
        net = NCANetworkTorch(d_state=d_state).to(device)
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = (1.0 / fan_in) ** 0.5
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=generator, device=device) * std)
                    if m.bias is not None:
                        m.bias.zero_()
        param_dicts.append({k: v.detach().clone() for k, v in net.named_parameters()})
    return {k: torch.stack([d[k] for d in param_dicts], dim=0) for k in param_dicts[0]}


def identity_probe_params(d_state, device):
    weight = torch.zeros(d_state, d_state, 3, 3, device=device)
    weight[:, :, 1, 1] = torch.eye(d_state, device=device)
    bias = torch.zeros(d_state, device=device)
    return weight, bias


def random_probe_params(d_state, device, generator):
    """Match nn.Conv2d default: Kaiming-uniform with a=sqrt(5)."""
    fan_in = d_state * 9
    w_bound = (1.0 / fan_in) ** 0.5
    weight = (torch.rand((d_state, d_state, 3, 3), generator=generator, device=device) * 2 - 1) * w_bound
    b_bound = 1.0 / (fan_in ** 0.5)
    bias = (torch.rand((d_state,), generator=generator, device=device) * 2 - 1) * b_bound
    return weight, bias


def _kaiming_uniform(shape, fan_in, generator, device):
    bound = (1.0 / fan_in) ** 0.5
    return (torch.rand(shape, generator=generator, device=device) * 2 - 1) * bound


def identity_probe_params_mlp(d_state, hidden, device, generator):
    """Residual MLP probe: pred = x + Conv1x1_H->D(ReLU(Conv3x3_D->H(x))).

    Near-identity init: w1 Kaiming, b1=0, w2 Kaiming*0.1, b2=0.
    Output at step 0 is x + tiny_block(x), so initial_loss ~ identity baseline
    while gradient still flows through w2 (nonzero, just small)."""
    w1 = _kaiming_uniform((hidden, d_state, 3, 3), d_state * 9, generator, device)
    b1 = torch.zeros(hidden, device=device)
    w2 = _kaiming_uniform((d_state, hidden, 1, 1), hidden, generator, device) * 0.1
    b2 = torch.zeros(d_state, device=device)
    return w1, b1, w2, b2


def random_probe_params_mlp(d_state, hidden, device, generator):
    """Residual MLP probe init at full Kaiming scale on both layers + biases.
    Output at step 0 is x + Kaiming_block(x); no special identity baseline."""
    w1 = _kaiming_uniform((hidden, d_state, 3, 3), d_state * 9, generator, device)
    b1 = _kaiming_uniform((hidden,), d_state * 9, generator, device)
    w2 = _kaiming_uniform((d_state, hidden, 1, 1), hidden, generator, device)
    b2 = _kaiming_uniform((d_state,), hidden, generator, device)
    return w1, b1, w2, b2


def nca_forward(net_template, params, state_nchw):
    """state_nchw: (N_IC, D, H, W). Returns dstate of same shape."""
    return functional_call(net_template, params, (state_nchw,))


def rollout_one_indiv(net_template, params, x0, masks_T, dt):
    """params: dict per-individual.
    x0: (N_IC, D, H, W). masks_T: (T, N_IC, 1, H, W).
    Returns (N_IC, T, D, H, W)."""
    state = x0
    states = []
    T = masks_T.shape[0]
    for t in range(T):
        dstate = nca_forward(net_template, params, state)
        state = torch.clamp(state + dstate * dt * masks_T[t], 0.0, 1.0)
        states.append(state)
    return torch.stack(states, dim=1)


def probe_loss(probe_w, probe_b, x_nchw, y_nchw):
    pred = F.conv2d(_wrap_pad(x_nchw, 1), probe_w, probe_b)
    return ((pred - y_nchw) ** 2).mean()


def probe_loss_mlp(probe_w1, probe_b1, probe_w2, probe_b2, x_nchw, y_nchw):
    """Residual MLP probe forward: pred = x + Conv1x1(ReLU(Conv3x3(x))).

    Wrap-pad on the 3x3, no padding on the 1x1. Residual skip means a
    near-zero block init gives pred ~ x (identity baseline), matching the
    semantics of the linear probe's identity init."""
    hidden = F.conv2d(_wrap_pad(x_nchw, 1), probe_w1, probe_b1)
    hidden = F.relu(hidden)
    delta = F.conv2d(hidden, probe_w2, probe_b2)
    pred = x_nchw + delta
    return ((pred - y_nchw) ** 2).mean()


def adam_step(p, m, v, g, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m_new = b1 * m + (1.0 - b1) * g
    v_new = b2 * v + (1.0 - b2) * g * g
    m_hat = m_new / (1.0 - b1 ** t)
    v_hat = v_new / (1.0 - b2 ** t)
    p_new = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return p_new, m_new, v_new


def _preq_components(losses_t):
    mse_floor = losses_t[-1]
    initial_loss = losses_t[0]
    preq_length = torch.clamp(losses_t - mse_floor, min=0.0).sum()
    probe_steps_t = torch.tensor(float(losses_t.shape[0]), device=losses_t.device)
    denom = torch.clamp((initial_loss - mse_floor) * probe_steps_t, min=1e-10)
    preq_gain = torch.clamp(preq_length / denom, min=0.0, max=1.0)
    return preq_length, preq_gain, initial_loss, mse_floor


def fitness_one(net_template, params, x0, masks_T, probe_w_init, probe_b_init, dt, probe_steps, lr):
    """Per-individual fitness for the linear (single Conv3x3) probe.

    probe_w_init, probe_b_init: starting probe weights. Shared across the population
    (broadcast via vmap in_dims=None on these args), so every individual is scored
    against the same probe baseline within a generation.
    """
    sims = rollout_one_indiv(net_template, params, x0, masks_T, dt)
    N_IC_, T, D, H, W = sims.shape
    x = sims[:, : T - 1].reshape(N_IC_ * (T - 1), D, H, W)
    y = sims[:, 1:].reshape(N_IC_ * (T - 1), D, H, W)

    probe_w = probe_w_init
    probe_b = probe_b_init
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
    preq_length, preq_gain, initial_loss, mse_floor = _preq_components(losses_t)
    return preq_length, preq_gain, initial_loss, mse_floor, losses_t


def fitness_one_mlp(net_template, params, x0, masks_T,
                    pw1_init, pb1_init, pw2_init, pb2_init,
                    dt, probe_steps, lr):
    """Per-individual fitness for the residual-MLP probe.

    Same rollout + Adam-on-probe protocol as fitness_one, but the probe is
    Conv3x3 -> ReLU -> Conv1x1 with a residual skip (see probe_loss_mlp)."""
    sims = rollout_one_indiv(net_template, params, x0, masks_T, dt)
    N_IC_, T, D, H, W = sims.shape
    x = sims[:, : T - 1].reshape(N_IC_ * (T - 1), D, H, W)
    y = sims[:, 1:].reshape(N_IC_ * (T - 1), D, H, W)

    pw1 = pw1_init; pb1 = pb1_init; pw2 = pw2_init; pb2 = pb2_init
    m_w1 = torch.zeros_like(pw1); v_w1 = torch.zeros_like(pw1)
    m_b1 = torch.zeros_like(pb1); v_b1 = torch.zeros_like(pb1)
    m_w2 = torch.zeros_like(pw2); v_w2 = torch.zeros_like(pw2)
    m_b2 = torch.zeros_like(pb2); v_b2 = torch.zeros_like(pb2)

    grad_fn = grad_and_value(probe_loss_mlp, argnums=(0, 1, 2, 3))
    losses = []
    for step in range(probe_steps):
        (gw1, gb1, gw2, gb2), loss = grad_fn(pw1, pb1, pw2, pb2, x, y)
        pw1, m_w1, v_w1 = adam_step(pw1, m_w1, v_w1, gw1, step + 1, lr)
        pb1, m_b1, v_b1 = adam_step(pb1, m_b1, v_b1, gb1, step + 1, lr)
        pw2, m_w2, v_w2 = adam_step(pw2, m_w2, v_w2, gw2, step + 1, lr)
        pb2, m_b2, v_b2 = adam_step(pb2, m_b2, v_b2, gb2, step + 1, lr)
        losses.append(loss)
    losses_t = torch.stack(losses)
    preq_length, preq_gain, initial_loss, mse_floor = _preq_components(losses_t)
    return preq_length, preq_gain, initial_loss, mse_floor, losses_t


def mutate_population(pop_params, sigma, generator):
    return {
        k: v + sigma * torch.randn(v.shape, generator=generator, device=v.device)
        for k, v in pop_params.items()
    }


def sample_eval_state(pop_size, n_ic, d_state, grid, rollout_steps, p_drop, device, generator):
    """ICs uniform in [0,1]; masks ~ Bernoulli(1 - p_drop), shape (T, POP, N_IC, 1, H, W).
    Vmapped over POP so we lay the POP axis as second to match in_dims=(0,..)."""
    x0 = torch.rand(
        (pop_size, n_ic, d_state, grid, grid),
        generator=generator, device=device,
    )
    rand_masks = torch.rand(
        (pop_size, rollout_steps, n_ic, 1, grid, grid),
        generator=generator, device=device,
    )
    masks = (rand_masks < (1.0 - p_drop)).to(torch.float32)
    return x0, masks


def render_rollout_grid(sims_one_ic, t_idx, title, path):
    """sims_one_ic: (T, D, H, W). For D=3 we map directly to RGB."""
    fig, axes = plt.subplots(1, len(t_idx), figsize=(1.6 * len(t_idx), 1.8))
    for col, t in enumerate(t_idx):
        img = sims_one_ic[t].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        axes[col].imshow(img)
        axes[col].axis("off")
        axes[col].set_title(f"t={t}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None,
                        help="If set, outputs go to demo_out/<run-name>/. Otherwise flat in demo_out/ "
                             "(legacy behavior).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device for training. Defaults to CUDA; use --device cpu --allow-cpu only for short checks.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU execution. Intended only for short checks/debugging.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--p-drop", type=float, default=P_DROP,
                        help="per-cell Bernoulli update-mask drop probability (default 0.5)")
    parser.add_argument("--sigma-init", type=float, default=SIGMA_INIT)
    parser.add_argument("--sigma-decay", type=float, default=SIGMA_DECAY)
    parser.add_argument("--probe-init", type=str, default="identity",
                        choices=["identity", "random"],
                        help="identity: probe init'd to copy-paste (preq_length tracks learnable structure beyond identity). "
                             "random: probe init'd Kaiming-uniform (preq_length tracks the probe's whole learning curve, "
                             "including learning identity).")
    parser.add_argument("--probe-arch", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="linear: single wrap-padded Conv3x3 (original). "
                             "mlp: residual block Conv3x3->ReLU->Conv1x1 with skip; "
                             "identity init keeps a near-identity start with nonzero gradient.")
    parser.add_argument("--probe-hidden", type=int, default=PROBE_HIDDEN,
                        help="Hidden width H of the MLP probe (Conv3x3 D->H, Conv1x1 H->D). "
                             "Ignored for --probe-arch linear.")
    parser.add_argument("--score", type=str, default="preq_length",
                        choices=["preq_length", "preq_gain"],
                        help="selection signal. preq_length: raw integrated area above floor. "
                             "preq_gain: same area normalized by (initial_loss - floor)*probe_steps, "
                             "bounded in [0,1]; cancels per-gen probe-init magnitude noise.")
    args = parser.parse_args()

    pop_size = args.pop_size
    n_generations = args.n_generations
    n_elite = max(1, pop_size * N_ELITE // POP_SIZE)
    seed = args.seed
    p_drop = args.p_drop
    sigma_init = args.sigma_init
    sigma_decay = args.sigma_decay
    probe_init = args.probe_init
    probe_arch = args.probe_arch
    probe_hidden = args.probe_hidden
    score_name = args.score

    base_out = os.path.join(str(DEMO_OUT), "runs")
    if args.run_name:
        out_dir = os.path.join(base_out, args.run_name)
    else:
        out_dir = str(DEMO_OUT)  # legacy flat
    best_png = "evolve_nca_preq_best_torch.png"
    curve_png = "evolve_nca_preq_curve_torch.png"
    log_tsv = "evolve_nca_preq_log_torch.tsv"
    os.makedirs(out_dir, exist_ok=True)

    device = resolve_torch_device(args.device, allow_cpu=args.allow_cpu)
    print(f"torch device: {describe_torch_device(device)}  run_name={args.run_name}")
    arch_tag = f"probe_arch={probe_arch}"
    if probe_arch == "mlp":
        arch_tag += f"(H={probe_hidden})"
    print(
        f"pop={pop_size} elite={n_elite} gens={n_generations} seed={seed} p_drop={p_drop} "
        f"probe_init={probe_init} {arch_tag} score={score_name} "
        f"sigma {sigma_init}->{sigma_init * sigma_decay**n_generations:.4f}"
    )

    g = torch.Generator(device=device).manual_seed(seed)

    net_template = NCANetworkTorch(d_state=D_STATE).to(device)
    pop_params = init_pop_params(net_template, pop_size, D_STATE, device, g)

    if probe_arch == "linear":
        fitness_vmapped = vmap(
            lambda p, x, m, pw, pb: fitness_one(
                net_template, p, x, m, pw, pb, DT, PROBE_STEPS, PROBE_LR,
            ),
            in_dims=(0, 0, 0, None, None),  # last two are shared probe init across population
        )
    else:
        fitness_vmapped = vmap(
            lambda p, x, m, pw1, pb1, pw2, pb2: fitness_one_mlp(
                net_template, p, x, m, pw1, pb1, pw2, pb2, DT, PROBE_STEPS, PROBE_LR,
            ),
            in_dims=(0, 0, 0, None, None, None, None),
        )

    sigma = sigma_init
    history = []
    best_probe_curves = []  # (n_generations, probe_steps) best individual's probe MSE per step
    mean_probe_curves = []  # (n_generations, probe_steps) population mean probe MSE per step
    best_ever_params = None
    best_ever_score = -float("inf")
    t0 = time.time()

    for gen in range(n_generations):
        x0, masks = sample_eval_state(pop_size, N_IC, D_STATE, GRID, ROLLOUT_STEPS, p_drop, device, g)
        if probe_arch == "linear":
            if probe_init == "identity":
                probe_w0, probe_b0 = identity_probe_params(D_STATE, device)
            else:
                probe_w0, probe_b0 = random_probe_params(D_STATE, device, g)
            with torch.no_grad():
                preq_lens, preq_gains, init_losses, mse_floors, probe_curves = fitness_vmapped(
                    pop_params, x0, masks, probe_w0, probe_b0,
                )
        else:
            if probe_init == "identity":
                pw1_0, pb1_0, pw2_0, pb2_0 = identity_probe_params_mlp(D_STATE, probe_hidden, device, g)
            else:
                pw1_0, pb1_0, pw2_0, pb2_0 = random_probe_params_mlp(D_STATE, probe_hidden, device, g)
            with torch.no_grad():
                preq_lens, preq_gains, init_losses, mse_floors, probe_curves = fitness_vmapped(
                    pop_params, x0, masks, pw1_0, pb1_0, pw2_0, pb2_0,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()

        scores = preq_lens if score_name == "preq_length" else preq_gains
        order = torch.argsort(scores)
        elite_idx = order[-n_elite:]
        elite_params = {k: v[elite_idx] for k, v in pop_params.items()}

        best_idx = elite_idx[-1]
        best_score = float(scores[best_idx])
        best_preq = float(preq_lens[best_idx])
        best_gain = float(preq_gains[best_idx])
        best_id = float(init_losses[best_idx])
        mean_score = float(scores.mean())
        history.append((gen, best_score, mean_score, best_preq, best_gain, best_id, sigma))

        probe_curves_np = probe_curves.detach().cpu().numpy()  # (pop, probe_steps)
        best_probe_curves.append(probe_curves_np[int(best_idx.item())].copy())
        mean_probe_curves.append(probe_curves_np.mean(axis=0))

        if best_score > best_ever_score:
            best_ever_score = best_score
            best_ever_params = {k: v[best_idx].detach().clone() for k, v in pop_params.items()}

        if gen % 5 == 0 or gen == n_generations - 1:
            print(
                f"gen {gen:3d}  best_{score_name}={best_score:.4f}  mean={mean_score:.4f}  "
                f"(preq_len={best_preq:.3f} preq_gain={best_gain:.3f})  "
                f"id_base={best_id:.4f}  sigma={sigma:.4f}  ({time.time()-t0:.1f}s)",
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

    # snapshot best-ever NCA on a fresh IC
    x0_snap, masks_snap = sample_eval_state(1, N_IC, D_STATE, GRID, ROLLOUT_STEPS, p_drop, device, g)
    best_stack = {k: v.unsqueeze(0) for k, v in best_ever_params.items()}
    with torch.no_grad():
        sims_best_stack = vmap(
            lambda p, x, m: rollout_one_indiv(net_template, p, x, m, DT),
            in_dims=(0, 0, 0),
        )(best_stack, x0_snap, masks_snap)
    sims_best = sims_best_stack[0]  # (N_IC, T, D, H, W)
    t_idx = torch.linspace(0, ROLLOUT_STEPS - 1, RENDER_STEPS).long().tolist()
    render_rollout_grid(
        sims_best[0], t_idx,
        f"best-ever NCA after {n_generations} generations ({score_name}={best_ever_score:.3f})",
        os.path.join(out_dir, best_png),
    )

    if args.run_name:
        ckpt_path = os.path.join(out_dir, "best_ever_params.pt")
        torch.save(
            {
                "params": {k: v.detach().cpu() for k, v in best_ever_params.items()},
                "d_state": D_STATE,
                "grid": GRID,
                "rollout_steps": ROLLOUT_STEPS,
                "dt": DT,
                "p_drop": p_drop,
                "probe_init": probe_init,
                "probe_arch": probe_arch,
                "probe_hidden": probe_hidden,
                "score_name": score_name,
                "best_score": best_ever_score,
                "seed": seed,
                "n_generations": n_generations,
                "pop_size": pop_size,
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path}")

    hist = torch.tensor(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(hist[:, 0], hist[:, 1], label="best in gen", color="tab:blue")
    ax[0].plot(hist[:, 0], hist[:, 2], label="mean", color="tab:gray", alpha=0.6)
    ax[0].set_xlabel("generation"); ax[0].set_ylabel(score_name)
    ax[0].set_title(f"{score_name} (selection signal) over generations")
    ax[0].legend(fontsize=8)
    ax[1].plot(hist[:, 0], hist[:, 3], label="preq_length", color="tab:blue")
    ax[1].plot(hist[:, 0], hist[:, 4], label="preq_gain", color="tab:green")
    ax[1].set_xlabel("generation"); ax[1].set_ylabel("component")
    ax[1].set_title("both components of best individual")
    ax[1].legend(fontsize=8)
    ax[2].plot(hist[:, 0], hist[:, 5], label="id_baseline", color="tab:orange")
    ax[2].set_xlabel("generation"); ax[2].set_ylabel("identity baseline")
    ax[2].set_title("dynamics activity (best)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, curve_png), dpi=130)
    plt.close(fig)

    log_path = os.path.join(out_dir, log_tsv)
    with open(log_path, "w") as f:
        f.write(f"gen\tbest_{score_name}\tmean_{score_name}\tbest_preq_length\tbest_preq_gain\tbest_id_baseline\tsigma\n")
        for row in history:
            f.write("\t".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                              for x in row) + "\n")
    print(f"saved {log_path}")

    # probe learning curves: (n_generations, probe_steps)
    best_curves = np.stack(best_probe_curves, axis=0)
    mean_curves = np.stack(mean_probe_curves, axis=0)
    np.save(os.path.join(out_dir, "probe_curves_best.npy"), best_curves)
    np.save(os.path.join(out_dir, "probe_curves_mean.npy"), mean_curves)

    # heatmap: gen (y) x probe_step (x) -> MSE. Show best and population-mean side by side.
    vmin = float(min(best_curves.min(), mean_curves.min()))
    vmax = float(max(best_curves.max(), mean_curves.max()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = ax[0].imshow(
        best_curves, aspect="auto", origin="lower",
        extent=[0, PROBE_STEPS, 0, n_generations],
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    ax[0].set_xlabel("probe SGD step"); ax[0].set_ylabel("generation")
    ax[0].set_title("best-of-gen probe learning curve  MSE")
    fig.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].imshow(
        mean_curves, aspect="auto", origin="lower",
        extent=[0, PROBE_STEPS, 0, n_generations],
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    ax[1].set_xlabel("probe SGD step"); ax[1].set_ylabel("generation")
    ax[1].set_title("population-mean probe learning curve  MSE")
    fig.colorbar(im1, ax=ax[1], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_curves_heatmap.png"), dpi=130)
    plt.close(fig)

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
    print(f"saved probe-curve arrays + heatmap + snapshots in {out_dir}")


if __name__ == "__main__":
    main()
