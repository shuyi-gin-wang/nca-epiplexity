"""
Student probe registry: one-step next-state predictors of increasing complexity.

Each entry exposes:
    init(d_state, hidden, device, generator) -> tuple[Tensor, ...]
    forward(params, x_nchw) -> Tensor with same shape as x_nchw

All probes are designed with near-identity init: at step 0 they predict ~x,
so initial_loss reflects the NCA's own one-step delta rather than the
probe's random weights, keeping preq_length comparable across archs.

Param packing is a flat tuple of tensors so torch.func.grad(argnums=0) over
the tuple gives gradients in the same order, enabling a single generic Adam
loop in the fitness function.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.nca_torch import _wrap_pad


def _kaiming_uniform(shape, fan_in, generator, device, scale: float = 1.0):
    bound = scale * (1.0 / fan_in) ** 0.5
    return (torch.rand(shape, generator=generator, device=device) * 2 - 1) * bound


# ----------------------------- linear (Conv3x3) -----------------------------

def linear_init(d_state, hidden, device, generator):
    w = torch.zeros(d_state, d_state, 3, 3, device=device)
    w[:, :, 1, 1] = torch.eye(d_state, device=device)
    b = torch.zeros(d_state, device=device)
    return (w, b)


def linear_fwd(params, x):
    w, b = params
    return F.conv2d(_wrap_pad(x, 1), w, b)


# ----------------------- residual MLP (Conv3x3 -> 1x1) ----------------------

def mlp_init(d_state, hidden, device, generator):
    w1 = _kaiming_uniform((hidden, d_state, 3, 3), d_state * 9, generator, device)
    b1 = torch.zeros(hidden, device=device)
    w2 = _kaiming_uniform((d_state, hidden, 1, 1), hidden, generator, device, scale=0.1)
    b2 = torch.zeros(d_state, device=device)
    return (w1, b1, w2, b2)


def mlp_fwd(params, x):
    w1, b1, w2, b2 = params
    h = F.conv2d(_wrap_pad(x, 1), w1, b1)
    h = F.relu(h)
    d = F.conv2d(h, w2, b2)
    return x + d


# --------------------- deep residual MLP (two Conv3x3) ----------------------

def deep_mlp_init(d_state, hidden, device, generator):
    w1 = _kaiming_uniform((hidden, d_state, 3, 3), d_state * 9, generator, device)
    b1 = torch.zeros(hidden, device=device)
    w2 = _kaiming_uniform((hidden, hidden, 3, 3), hidden * 9, generator, device)
    b2 = torch.zeros(hidden, device=device)
    w3 = _kaiming_uniform((d_state, hidden, 1, 1), hidden, generator, device, scale=0.1)
    b3 = torch.zeros(d_state, device=device)
    return (w1, b1, w2, b2, w3, b3)


def deep_mlp_fwd(params, x):
    w1, b1, w2, b2, w3, b3 = params
    h = F.conv2d(_wrap_pad(x, 1), w1, b1)
    h = F.relu(h)
    h = F.conv2d(_wrap_pad(h, 1), w2, b2)
    h = F.relu(h)
    d = F.conv2d(h, w3, b3)
    return x + d


# --------------------------- tiny ViT-style block ---------------------------

_TRANSFORMER_N_HEADS = 2


def _layernorm_chw(x, gain, bias, eps=1e-5):
    """LayerNorm over the channel dim of an (N, C, H, W) tensor.

    Implemented functionally (mean/var rather than F.layer_norm) so it composes
    cleanly under torch.func.vmap/grad. gain/bias are (C,) tensors.
    """
    mu = x.mean(dim=1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=1, keepdim=True)
    xn = (x - mu) / torch.sqrt(var + eps)
    return xn * gain.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


def transformer_init(d_state, hidden, device, generator):
    H = hidden
    # token embed (Conv1x1 D->H)
    we = _kaiming_uniform((H, d_state, 1, 1), d_state, generator, device)
    be = torch.zeros(H, device=device)
    # learned positional embedding starting at zero (so init has no per-position bias)
    pos = torch.zeros(1, H, 1, 1, device=device)  # broadcast — global only
    pos_grid = torch.zeros(1, H, 16, 16, device=device)
    # QKV projection (1x1 conv H -> 3H)
    wqkv = _kaiming_uniform((3 * H, H, 1, 1), H, generator, device)
    bqkv = torch.zeros(3 * H, device=device)
    # attention output projection
    wproj = _kaiming_uniform((H, H, 1, 1), H, generator, device, scale=0.1)
    bproj = torch.zeros(H, device=device)
    # FFN
    wf1 = _kaiming_uniform((4 * H, H, 1, 1), H, generator, device)
    bf1 = torch.zeros(4 * H, device=device)
    wf2 = _kaiming_uniform((H, 4 * H, 1, 1), 4 * H, generator, device, scale=0.1)
    bf2 = torch.zeros(H, device=device)
    # decode (Conv1x1 H -> D), small scale so initial pred ~= x via outer residual
    wo = _kaiming_uniform((d_state, H, 1, 1), H, generator, device, scale=0.1)
    bo = torch.zeros(d_state, device=device)
    # pre-norm LayerNorm gains/biases (gain=1, bias=0 -> identity at init): one
    # before attention (ln1), one before the FFN (ln2), one before decode (lnf).
    # Stabilizes Adam early-step dynamics so the probe MSE descends monotonically
    # instead of overshooting (the unnormalized block spiked around step ~5).
    ln1_g = torch.ones(H, device=device); ln1_b = torch.zeros(H, device=device)
    ln2_g = torch.ones(H, device=device); ln2_b = torch.zeros(H, device=device)
    lnf_g = torch.ones(H, device=device); lnf_b = torch.zeros(H, device=device)
    return (we, be, pos_grid, wqkv, bqkv, wproj, bproj, wf1, bf1, wf2, bf2, wo, bo,
            ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b)


def transformer_fwd(params, x):
    (we, be, pos_grid, wqkv, bqkv, wproj, bproj, wf1, bf1, wf2, bf2, wo, bo,
     ln1_g, ln1_b, ln2_g, ln2_b, lnf_g, lnf_b) = params
    N, D, Hg, Wg = x.shape
    H = we.shape[0]
    n_heads = _TRANSFORMER_N_HEADS
    head_dim = H // n_heads
    L = Hg * Wg

    if pos_grid.shape[-2:] != (Hg, Wg):
        pos = F.interpolate(pos_grid, size=(Hg, Wg), mode="bilinear", align_corners=False)
    else:
        pos = pos_grid
    h = F.conv2d(x, we, be) + pos  # (N, H, Hg, Wg)

    # self-attention (pre-norm)
    hn = _layernorm_chw(h, ln1_g, ln1_b)
    qkv = F.conv2d(hn, wqkv, bqkv)  # (N, 3H, Hg, Wg)
    q, k, v = qkv.chunk(3, dim=1)
    q = q.reshape(N, n_heads, head_dim, L)
    k = k.reshape(N, n_heads, head_dim, L)
    v = v.reshape(N, n_heads, head_dim, L)
    scale = head_dim ** -0.5
    attn = torch.einsum("nhdl,nhdm->nhlm", q, k) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.einsum("nhlm,nhdm->nhdl", attn, v).reshape(N, H, Hg, Wg)
    out = F.conv2d(out, wproj, bproj)
    h = h + out

    # FFN (pre-norm)
    hn2 = _layernorm_chw(h, ln2_g, ln2_b)
    h2 = F.conv2d(hn2, wf1, bf1)
    h2 = F.gelu(h2)
    h2 = F.conv2d(h2, wf2, bf2)
    h = h + h2

    # final norm before decode
    hf = _layernorm_chw(h, lnf_g, lnf_b)
    d = F.conv2d(hf, wo, bo)
    return x + d


# ------------------------------- registry -----------------------------------

PROBE_REGISTRY = {
    "linear":      dict(init=linear_init,      forward=linear_fwd,      default_hidden=0),
    "mlp_small":   dict(init=mlp_init,         forward=mlp_fwd,         default_hidden=8),
    "mlp_wide":    dict(init=mlp_init,         forward=mlp_fwd,         default_hidden=32),
    "deep_mlp":    dict(init=deep_mlp_init,    forward=deep_mlp_fwd,    default_hidden=16),
    "transformer": dict(init=transformer_init, forward=transformer_fwd, default_hidden=32),
}


def make_probe(arch: str, d_state: int, hidden: int, device, generator):
    """Returns (init_params: tuple[Tensor], forward_fn: callable)."""
    cfg = PROBE_REGISTRY[arch]
    h = hidden if hidden > 0 else cfg["default_hidden"]
    init_params = cfg["init"](d_state, h, device, generator)
    return init_params, cfg["forward"]


def probe_param_count(arch: str, d_state: int, hidden: int) -> int:
    """Approximate trainable param count for logging."""
    if arch == "linear":
        return d_state * d_state * 9 + d_state
    h = hidden if hidden > 0 else PROBE_REGISTRY[arch]["default_hidden"]
    if arch in ("mlp_small", "mlp_wide"):
        return h * d_state * 9 + h + d_state * h + d_state
    if arch == "deep_mlp":
        return h * d_state * 9 + h + h * h * 9 + h + d_state * h + d_state
    if arch == "transformer":
        return (
            h * d_state + h            # embed
            + h * 16 * 16              # learned pos grid, resized at runtime when needed
            + 3 * h * h + 3 * h        # qkv
            + h * h + h                # proj
            + 4 * h * h + 4 * h        # ffn1
            + h * 4 * h + h            # ffn2
            + d_state * h + d_state    # decode
            + 6 * h                    # 3x LayerNorm (gain+bias)
        )
    return -1
