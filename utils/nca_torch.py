"""
PyTorch port of the continuous NCA + prequential probe pipeline from utils/nca.py.

Mirrors:
  - NCAContinuous          -> NCAContinuousTorch
  - _generate_rule_rollouts_continuous -> generate_rule_rollouts_continuous
  - _train_probe_one_rule_preq_continuous (MSE, variance baseline)
       -> train_probe_preq_continuous

Differences:
  - CUDA is the default execution target for training scripts. CPU runs are
    explicit quick-check/debug opt-ins.
  - Prequential gain normalization fixed: divides by (initial_loss - floor) * probe_steps,
    so the score is bounded in [0,1] and means "fraction of the early-vs-floor gap
    actually integrated" rather than "fraction of variance" (which had no [0,1] bound
    when loss could exceed variance early in training).
  - All rules trained as a single batched conv (groups=B) instead of vmap.

Tensor layout: NCHW for conv ops, converted to/from NHWC at substrate boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_torch_device(device: str = "cuda", allow_cpu: bool = False) -> torch.device:
    """Resolve a torch device with CUDA required by default.

    Parameters
    ----------
    device:
        Device string. Use "cuda" or "cuda:N" for real training. "auto" means
        CUDA when available, otherwise CPU only if allow_cpu=True.
    allow_cpu:
        Permit CPU execution for short checks and local debugging.
    """
    requested = str(device)
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is required for epiplexity training, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or rerun with --device cpu --allow-cpu for a short check."
        )

    resolved = torch.device(requested)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {requested}, but CUDA is not available in this PyTorch runtime. "
                "Install a CUDA-enabled PyTorch build or use --device cpu --allow-cpu only for short checks."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return resolved

    if resolved.type == "cpu" and not allow_cpu:
        raise RuntimeError("CPU execution is disabled by default. Pass --allow-cpu for short/debug runs.")

    return resolved


def describe_torch_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)


def _wrap_pad(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Circular pad an NCHW tensor by `pad` on H and W."""
    return F.pad(x, (pad, pad, pad, pad), mode="circular")


class NCANetworkTorch(nn.Module):
    """Conv3x3(4) -> Conv1x1(16) -> ReLU -> Conv1x1(d_state).

    Matches Flax NCANetwork from utils/nca.py: 'VALID' padding after wrap-pad,
    and the second Conv1x1 has no activation after it.
    """

    def __init__(self, d_state: int = 3):
        super().__init__()
        self.d_state = d_state
        self.conv3 = nn.Conv2d(d_state, 4, kernel_size=3, padding=0)
        self.conv1a = nn.Conv2d(4, 16, kernel_size=1)
        self.conv1b = nn.Conv2d(16, d_state, kernel_size=1)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        x = _wrap_pad(x_nchw, pad=1)
        x = self.conv3(x)
        x = self.conv1a(x)
        x = F.relu(x)
        x = self.conv1b(x)
        return x


def _init_nca_params(rng: torch.Generator, d_state: int, device: torch.device) -> NCANetworkTorch:
    """Create a fresh NCANetwork with deterministic init from `rng`.

    Uses LeCun-normal-ish init (default for nn.Conv2d is Kaiming uniform), which
    is close enough to Flax's default for our purposes — the substrate is meant
    to be sampled fresh per rule, not trained.
    """
    net = NCANetworkTorch(d_state=d_state).to(device)
    # Re-init weights deterministically from the given generator.
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            with torch.no_grad():
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = (1.0 / fan_in) ** 0.5
                m.weight.copy_(torch.randn(m.weight.shape, generator=rng, device=device) * std)
                if m.bias is not None:
                    m.bias.zero_()
    return net


@dataclass
class NCAContinuousTorch:
    grid_size: int = 128
    d_state: int = 3
    p_drop: float = 0.5
    dt: float = 0.01
    device: torch.device = torch.device("cuda")

    def sample_net(self, rule_seed: int) -> NCANetworkTorch:
        rng = torch.Generator(device=self.device).manual_seed(int(rule_seed))
        return _init_nca_params(rng, self.d_state, self.device)

    def init_state(self, ic_seed: int) -> torch.Tensor:
        """Returns (1, D, H, W) in [0,1]."""
        rng = torch.Generator(device=self.device).manual_seed(int(ic_seed))
        return torch.rand(
            (1, self.d_state, self.grid_size, self.grid_size),
            generator=rng, device=self.device,
        )

    @torch.no_grad()
    def step(self, state_nchw: torch.Tensor, net: NCANetworkTorch, step_rng: torch.Generator) -> torch.Tensor:
        dstate = net(state_nchw)
        mask = torch.rand(
            state_nchw.shape[0], 1, state_nchw.shape[2], state_nchw.shape[3],
            generator=step_rng, device=self.device,
        )
        mask = 1.0 - torch.floor(mask + self.p_drop)
        dstate = dstate * mask
        return torch.clamp(state_nchw + dstate * self.dt, 0.0, 1.0)


@torch.no_grad()
def rollout_one_rule(
    substrate: NCAContinuousTorch,
    rule_seed: int,
    rule_index: int,
    n_ic: int,
    rollout_steps: int,
    ic_rng_seed: int = 1,
) -> torch.Tensor:
    """Roll out a single rule. Returns (n_ic, T, H, W, D) on substrate.device."""
    H = W = substrate.grid_size
    D = substrate.d_state
    out = torch.empty(
        (n_ic, rollout_steps, H, W, D),
        device=substrate.device, dtype=torch.float32,
    )
    net = substrate.sample_net(int(rule_seed))
    net.eval()
    ic_seeds = [ic_rng_seed + rule_index * n_ic + i for i in range(n_ic)]
    state = torch.cat([substrate.init_state(s) for s in ic_seeds], dim=0)
    step_rng = torch.Generator(device=substrate.device).manual_seed(
        int(ic_rng_seed) * 7919 + rule_index
    )
    for t in range(rollout_steps):
        state = substrate.step(state, net, step_rng)
        out[:, t] = state.permute(0, 2, 3, 1)
    return out


@torch.no_grad()
def generate_rule_rollouts_continuous(
    substrate: NCAContinuousTorch,
    rule_seeds: torch.Tensor,
    n_ic: int,
    rollout_steps: int,
    ic_rng_seed: int = 1,
) -> torch.Tensor:
    """
    Aggregate version: returns rollouts of shape (B, n_ic, T, H, W, D).
    Only use when the result fits in device memory — for large sweeps prefer
    streaming via rollout_one_rule.
    """
    B = rule_seeds.shape[0]
    H = W = substrate.grid_size
    D = substrate.d_state
    out = torch.empty(
        (B, n_ic, rollout_steps, H, W, D),
        device=substrate.device, dtype=torch.float32,
    )
    for b in range(B):
        out[b] = rollout_one_rule(
            substrate, int(rule_seeds[b].item()), b, n_ic, rollout_steps, ic_rng_seed,
        )
    return out


class ProbeConv(nn.Module):
    """Single wrap-padded Conv3x3 -> d_state. Strictly weaker than NCANetwork."""

    def __init__(self, d_state: int):
        super().__init__()
        self.conv = nn.Conv2d(d_state, d_state, kernel_size=3, padding=0)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return self.conv(_wrap_pad(x_nchw, pad=1))


def train_probe_preq_one_rule(
    rollouts_one: torch.Tensor,
    d_state: int,
    probe_steps: int,
    lr: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train a probe on a single rule's rollouts.
    rollouts_one: (n_ic, T, H, W, D) on device.
    Returns 0-d tensors: preq_gain, preq_length, mse_floor, initial_loss, baseline.
    """
    device = rollouts_one.device
    n_ic, T, H, W, D = rollouts_one.shape
    x = rollouts_one[:, : T - 1].reshape(n_ic * (T - 1), H, W, D).permute(0, 3, 1, 2).contiguous()
    y = rollouts_one[:, 1:].reshape(n_ic * (T - 1), H, W, D).permute(0, 3, 1, 2).contiguous()

    probe = ProbeConv(d_state=d_state).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    losses = []
    for s in range(probe_steps):
        pred = probe(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    losses_t = torch.stack(losses)
    mse_floor = losses_t[-1]
    initial_loss = losses_t[0]

    preq_length = torch.clamp(losses_t - mse_floor, min=0.0).sum()
    denom = torch.clamp((initial_loss - mse_floor) * probe_steps, min=1e-10)
    preq_gain = torch.clamp(preq_length / denom, min=0.0, max=1.0)

    with torch.no_grad():
        mean_y = y.mean(dim=(0, 2, 3), keepdim=True)
        baseline = ((y - mean_y) ** 2).mean()

    return preq_gain, preq_length, mse_floor, initial_loss, baseline


def sweep_rules_streaming(
    substrate: NCAContinuousTorch,
    rule_seeds: torch.Tensor,
    n_ic: int,
    rollout_steps: int,
    probe_steps: int,
    lr: float,
    ic_rng_seed: int = 1,
    probe_rng_seed: int = 0,
    progress_every: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each rule: roll out on device, train probe on device, discard rollouts.
    Avoids materializing the full (B, n_ic, T, H, W, D) tensor.

    Returns tensors of shape (B,) on CPU.
    """
    import time as _time
    B = rule_seeds.shape[0]
    gains = torch.empty(B)
    lengths = torch.empty(B)
    floors = torch.empty(B)
    inits = torch.empty(B)
    baselines = torch.empty(B)

    torch.manual_seed(int(probe_rng_seed))
    t0 = _time.time()
    for b in range(B):
        sims_one = rollout_one_rule(
            substrate, int(rule_seeds[b].item()), b, n_ic, rollout_steps, ic_rng_seed,
        )
        g, ln, fl, il, bl = train_probe_preq_one_rule(
            sims_one, substrate.d_state, probe_steps, lr,
        )
        gains[b] = g.cpu(); lengths[b] = ln.cpu(); floors[b] = fl.cpu()
        inits[b] = il.cpu(); baselines[b] = bl.cpu()
        del sims_one
        if substrate.device.type == "cuda":
            torch.cuda.empty_cache()
        if progress_every and (b + 1) % progress_every == 0:
            elapsed = _time.time() - t0
            eta = elapsed / (b + 1) * (B - b - 1)
            print(f"  rule {b+1}/{B}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    return gains, lengths, floors, inits, baselines


def train_probe_preq_continuous(
    rollouts: torch.Tensor,
    d_state: int,
    probe_steps: int,
    lr: float,
    rng_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train one probe per rule (no vmap; just a Python loop over rules).

    rollouts: (B, n_ic, T, H, W, D)

    Fixed-normalization preq score:
        preq_gain = preq_length / ((initial_loss - mse_floor) * probe_steps)
    with preq_length = sum_t max(loss_t - mse_floor, 0). This is bounded in [0,1]
    when the loss curve is monotone decreasing, and reads as "fraction of the
    early-loss vs final-loss gap I actually integrated under". Higher = the
    probe spent more of training near the initial loss before snapping to floor,
    i.e. structure took more bits to describe -> richer dynamics.

    Returns tensors of shape (B,): preq_gain, preq_length, mse_floor,
        initial_loss, baseline_variance.
    """
    device = rollouts.device
    B, n_ic, T, H, W, D = rollouts.shape

    # Build (s_t, s_{t+1}) pairs across all ICs and timesteps.
    x = rollouts[:, :, : T - 1]  # (B, n_ic, T-1, H, W, D)
    y = rollouts[:, :, 1:]
    # Flatten time/IC into a single batch axis per rule, then to NCHW for conv.
    x = x.reshape(B, n_ic * (T - 1), H, W, D).permute(0, 1, 4, 2, 3).contiguous()
    y = y.reshape(B, n_ic * (T - 1), H, W, D).permute(0, 1, 4, 2, 3).contiguous()

    gains = torch.empty(B, device=device)
    lengths = torch.empty(B, device=device)
    floors = torch.empty(B, device=device)
    inits = torch.empty(B, device=device)
    baselines = torch.empty(B, device=device)

    torch.manual_seed(int(rng_seed))
    for b in range(B):
        probe = ProbeConv(d_state=d_state).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        losses = []
        for s in range(probe_steps):
            pred = probe(x[b])
            loss = F.mse_loss(pred, y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        losses_t = torch.stack(losses)
        mse_floor = losses_t[-1]
        initial_loss = losses_t[0]

        preq_length = torch.clamp(losses_t - mse_floor, min=0.0).sum()
        denom = torch.clamp((initial_loss - mse_floor) * probe_steps, min=1e-10)
        preq_gain = torch.clamp(preq_length / denom, min=0.0, max=1.0)

        # variance baseline kept for backward-comp diagnostics
        with torch.no_grad():
            mean_y = y[b].mean(dim=(0, 2, 3), keepdim=True)
            baseline = ((y[b] - mean_y) ** 2).mean()

        gains[b] = preq_gain
        lengths[b] = preq_length
        floors[b] = mse_floor
        inits[b] = initial_loss
        baselines[b] = baseline

    return gains, lengths, floors, inits, baselines
