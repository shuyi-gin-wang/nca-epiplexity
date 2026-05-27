import flax.linen as nn
import jax
from flax.core import freeze, unfreeze

import jax.lax as lax
import jax.numpy as jnp
import optax
from einops import rearrange, reduce, repeat
from jax.random import split

import io
import gzip
from functools import partial
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from .tokenizers import Tokenizer

#############################################################
#  NCA Simulation
#############################################################

def rollout_simulation(rng, params, s0=None,
                       substrate=None, fm=None, rollout_steps=256, time_sampling='final', img_size=224, start_step=0,return_state=False, k_steps=1):
    """
    Rollout a simulation described by the specified substrate and parameters.

    Parameters
    ----------
    rng : jax rng seed for the rollout
    params : parameters to configure the simulation within the substrate
    s0 : (optional) initial state of the simulation. If None, then substrate.init_state(rng, params) is used.
    substrate : the substrate object
    fm : the foundation model object. If None, then no image embedding is calculated.
    rollout_steps : number of timesteps to run simulation for
    time_sampling : one of either
        - 'final': return only the final state data (default)
        - 'video': return the entire rollout
        - (K, chunk_ends): return the rollout at K sampled intervals, if chunk_ends is True then end of intervals is sampled
    img_size : image size to render at. Leave at 224 to avoid resizing again for CLIP.
    return_state : return the state data, leave as False, unless you really need it.

    Returns
    ----------
    A dictionary containing
    'state' : the state data of the simulation, None if return_state is False
        shape: (...)
    'rgb' : the image data of the simulation,
        shape (H, W, C)
    'z' : the image embedding of the simulation using the foundation model,
        shape (D)

    If time_sampling is 'video' then the returned shapes become (rollout_steps, ...).
    If time_sampling is an int then the returned shapes become (time_sampling, ...).

    ----------
    This function should be used like this:
    ```
    fm = create_foundation_model('clip')
    substrate = create_substrate('lenia')
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=256, time_sampling=8, img_size=224, return_state=False)
    rollout_fn = jax.jit(rollout_fn) # jit for speed
    # now you can use rollout_fn as you need...
    rng = jax.random.PRNGKey(0)
    params = substrate.default_params(rng)
    rollout_data = rollout_fn(rng, params)
    ```

    Note: 
    - when time_sampling is 'final', the function returns data for the T timestep.
    - when time_sampling is 'video', the function returns data for the [0, ..., T-1] timesteps.
    - when time_sampling is (K, False), the function returns data for the [0, T//K, T//K * 2, ..., T - T//K] timesteps.
    - when time_sampling is (K, True), the function returns data for the [T//K, T//K * 2, ..., T] timesteps.
    """

    if s0 is None:
        s0 = substrate.init_state(rng, params)
    
    if time_sampling == 'final': # return only the final state
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, None
        state_final, _ = jax.lax.scan(step_fn, s0, split(rng, rollout_steps))
        
        if return_state:
            return jnp.stack([s0, state_final])
        else:
            return state_final
    elif time_sampling == 'video': # return the entire rollout
        def step_fn(state, _rng):
            next_state = substrate.step_state(_rng, state, params)
            return next_state, state
        _, state_vid = jax.lax.scan(step_fn, s0, split(rng, start_step+rollout_steps))
        idx = jnp.arange(start_step, start_step+rollout_steps, k_steps)
        return state_vid[idx]
    else:
        raise ValueError(f"time_sampling {time_sampling} not recognized")


#############################################################
#  NCA Network adapted from https://github.com/SakanaAI/asal/
#############################################################

class NCANetwork(nn.Module):
    d_state: int = 16
    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, pad_width=1, mode='wrap')
        x = nn.Conv(features=4, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.Conv(features=16, kernel_size=(1, 1))(x)

        x = nn.relu(x)
        x = nn.Conv(features=self.d_state, kernel_size=(1, 1))(x)
        return x

class RandomMLP(nn.Module):
    n_layers: int
    d_hidden: int
    d_out: int
    activation: Callable
    @nn.compact
    def __call__(self, x, train=False):
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.d_hidden, kernel_init=nn.initializers.normal(1), bias_init=nn.initializers.normal(1))(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
            x = self.activation(x)
        x = nn.Dense(features=self.d_out)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0., use_bias=False, use_scale=False)(x)
        return x
        
def create_random_net(net, rng, x):
    rng, _rng = split(rng)
    params = net.init(_rng, jnp.zeros_like(x[0]))
    y, updates = net.apply(params, x, train=True, mutable=['batch_stats'])
    params = unfreeze(params)
    params['batch_stats'] = updates['batch_stats']
    params = freeze(params)
    y = net.apply(params, x)
    assert jnp.allclose(y.mean(axis=0), jnp.zeros_like(y.mean(axis=0)), atol=1e-2)
    assert jnp.allclose(y.std(axis=0), jnp.ones_like(y.std(axis=0)), atol=1e-2)


#############################################################
#  NCA Parameterization
#############################################################

"""
The Discrete NCA substrate.
This substrate models a grid of integer values and updates them using a stochastic neural cellular automata.
"""
class NCA():
    def __init__(self, grid_size=64, d_state=8, n_groups=1, identity_bias=0., temperature=1., color_map='fixed'):
        self.grid_size = grid_size
        self.d_state, self.n_groups = d_state, n_groups
        self.nca = NCANetwork(d_state=d_state*n_groups)

        self.identity_bias, self.temperature = identity_bias, temperature

        self.color_map = color_map
        self.color_palette = 'ff0000-00ff00-0000ff-ffff00-ff00ff-00ffff-ffffff-8f5d00'
        self.color_palette = jnp.array([mcolors.to_rgb(f"#{a}") for a in self.color_palette.split('-')]) # 8 3

    def default_params(self, rng):
        params = dict()
        if self.color_map != 'fixed':
            rng, _rng = split(rng)
            params['color_map'] = jax.random.normal(_rng, (self.n_groups, self.d_state, 3)) # unconstrainted

        rng, _rng = split(rng)
        params['net_params'] = self.nca.init(_rng, jnp.zeros((self.grid_size, self.grid_size, self.d_state*self.n_groups))) # unconstrainted

        rng, _rng = split(rng)
        params['init'] = jax.random.normal(_rng, (self.n_groups, self.d_state)) # unconstrainted
        return params
    
    def init_state(self, rng, params):
        init = repeat(jax.random.normal(rng, (self.n_groups, self.d_state)), "G D -> H W G D", H=self.grid_size, W=self.grid_size)
        # init = repeat(params['init'], "G D -> H W G D", H=self.grid_size, W=self.grid_size)
        # changing the way we initialize starting states
        state = jax.random.categorical(rng, init, axis=-1)
        return state
    
    def step_state(self, rng, state, params):
        state_oh = jax.nn.one_hot(state, self.d_state)
        state_oh_f = rearrange(state_oh, "H W G D -> H W (G D)")
        logits = self.nca.apply(params['net_params'], state_oh_f)
        logits = rearrange(logits, "H W (G D) -> H W G D", G=self.n_groups)
        
        # identity_bias = jax.nn.sigmoid(params['identity_bias'])*10
        next_state = jax.random.categorical(rng, (logits + state_oh*self.identity_bias)/self.temperature, axis=-1)
        return next_state
    
    def render_state(self, state, params, img_size=None):
        if self.color_map=='fixed':
            img = self.color_palette[state[:, :, 0]]
        else:
            def get_color(color_map, state):
                return color_map[state]
            # color_map: G D 3 # state: H W G
            get_color = jax.vmap(get_color, in_axes=(0, 2))
            img = get_color(jax.nn.sigmoid(params['color_map']), state)
            img = img.mean(axis=0) # average over groups

        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='nearest')
        return img

#############################################################
#  NCA Dataset Generation
#############################################################

def generate_nca_dataset(
    seed: jax.random.PRNGKey,
    num_sims:int,
    grid: int = 12,
    d_state: int = 10,
    n_groups: int = 1,
    identity_bias: float = 0.,
    temperature: float = 0.,
    num_examples: int = 10,
    num_rules: int = 10,
    dT: int = 2,
    start_step: int = 0,
    rule_seeds = None
):
    generator = NCA(
        grid_size=grid,
        d_state=d_state,
        n_groups=n_groups,
        identity_bias=identity_bias,
        temperature=temperature
    )

    if(rule_seeds is None):
        rule_seeds = jax.random.split(seed, num_rules)
    else:
        rule_seeds = jnp.tile(rule_seeds, (num_sims // num_rules, 1))
        rule_seeds = jnp.concat([rule_seeds, rule_seeds[:(num_sims - rule_seeds.shape[0])]], axis=0)

    # generate simulations
    def rollout_fn(rng, task_seed):
        params = generator.default_params(task_seed)
        return rollout_simulation(rng, params, substrate=generator, rollout_steps=dT*num_examples, k_steps=dT, time_sampling='video', start_step=start_step)
        
    sim_data = jax.vmap(rollout_fn, in_axes=(0,0))(split(seed, num_sims), rule_seeds)

    return sim_data

#############################################################
#  Rule Generation
#############################################################

def generate_rules_batch(
    seed: jax.random.PRNGKey,
    num_rules: int,
    tokenizer: Tokenizer = None,
    threshold: float = 40,
    upper_bound: float = None,
    dT: int = 1,
    n_steps:int = 10,
    mode: str = 'shannon', # 'diff' or 'shannon'
    start_step: int = 0,
    grid: int = 12,
    d_state: int = 10,
    identity_bias: float = 0.,
    temperature: float = 0.
):
    current_seed = seed
    current_rules = 0
    rules = []

    if(upper_bound is None):
        upper_bound = float('inf')

    while(current_rules < num_rules):
        seeds = jax.random.split(current_seed, num_rules) # sample more rules to ensure we get enough
        score = compute_rule_gzip_batch(seeds, tokenizer, grid, d_state, identity_bias, temperature, n_steps, dT, start_step, mode)
        idx = jnp.logical_and(score > threshold, score < upper_bound)

        selected_rules = seeds[idx]
        score = score[idx]

        if (len(selected_rules) > 0):
            rules.append(selected_rules[:num_rules - current_rules])
            current_rules += len(selected_rules)
        current_seed = jax.random.split(current_seed, 1)[0]

    return jnp.concat(rules, axis=0)

def compute_rule_gzip_batch(
    seeds,
    tokenizer: Tokenizer,
    grid: int = 12,
    d_state: int = 10,
    identity_bias: float = 0.,
    temperature: float = 0.,
    n_steps: int = 10,
    dT: int = 1,
    start_step: int = 0,
    mode: str = 'gzip'
):
    sims = generate_nca_dataset(
            seeds[0],
            num_sims=seeds.shape[0],
            grid=grid,
            d_state=d_state,
            n_groups=1,
            identity_bias=identity_bias,
            temperature=temperature,
            num_examples=n_steps, # number of examples to rollout
            dT=dT,
            rule_seeds=seeds,
            num_rules=seeds.shape[0],
            start_step = start_step
        )

    B, T, H, W, C = sims.shape

    if(mode == 'diff'):
        diff = (sims[:,2:,...] != sims[:,1:-1,...]).astype(int)
        average_diff = diff.mean(axis=(1)) # average across timesteps
        score = average_diff.mean(axis=tuple(range(1, sims.ndim)))

    elif(mode == 'gzip'):
        seq, _ = tokenizer.encode_task(sims)
        grid_len = (H*W)//(tokenizer.patch**2)+2
        seq = seq.reshape(B, -1, grid_len)
        seq = seq[:, :, 1:-1]
        seq = jnp.array(seq).reshape(B, -1)
        score = []
        for b in range(B):
            byte_data = seq[b].tobytes()
            score.append(gzip_complexity(byte_data))
        score = jnp.array(score)
    return score

def gzip_complexity(byte_data: bytes):
    """
    Compute the GZIP compression complexity of a sequence of tokens.
    """
    buf = io.BytesIO()

    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    compressed_size = len(buf.getvalue())
    original_size = len(byte_data)
    return compressed_size / original_size


#############################################################
#  Probe-based "reducible structure" score
#
#  Filter B for epiplexity: train a tiny linear probe to predict
#  s_{t+k} from s_t on each rule's rollouts (with multiple ICs).
#  The score = 1 - probe_loss / marginal_baseline_loss is the
#  fraction of marginal entropy the probe can explain.
#
#  - High gzip + high probe-gain  -> class IV (reducible structure)
#  - High gzip + low  probe-gain  -> class III (chaos / noise)
#  - Low  gzip + high probe-gain  -> class II  (ordered / periodic)
#  - Low  gzip + low  probe-gain  -> class I   (fixed point)
#############################################################


class _ProbeConv(nn.Module):
    """Single Conv3x3 -> logits. Strictly weaker than NCANetwork."""
    d_total: int
    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, pad_width=((1, 1), (1, 1), (0, 0)), mode='wrap')
        x = nn.Conv(features=self.d_total, kernel_size=(3, 3), padding='VALID')(x)
        return x


def _train_probe_one_rule(rng, rollouts, d_state, n_groups, k, probe_steps, lr):
    """
    rollouts: (n_ic, T, H, W, G) integer states
    Returns (probe_gain, final_loss, baseline_loss).
    """
    d_total = d_state * n_groups
    T = rollouts.shape[1]

    x = rollouts[:, :T - k]   # (n_ic, T-k, H, W, G)
    y = rollouts[:, k:]       # (n_ic, T-k, H, W, G)

    x_oh = jax.nn.one_hot(x, d_state)
    x_oh = rearrange(x_oh, "I T H W G D -> (I T) H W (G D)")
    y_flat = rearrange(y, "I T H W G -> (I T) H W G")

    probe = _ProbeConv(d_total=d_total)
    rng_init, _ = split(rng)
    params = probe.init(rng_init, x_oh[0])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def forward(p, batch):
        logits = jax.vmap(probe.apply, in_axes=(None, 0))(p, batch)
        return rearrange(logits, "N H W (G D) -> N H W G D", G=n_groups)

    def loss_fn(p):
        logits = forward(p, x_oh)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y_flat).mean()

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = lax.scan(step, (params, opt_state), None, length=probe_steps)
    final_loss = losses[-1]

    # marginal baseline: cross-entropy of predicting the per-rule color histogram of y
    y_oh = jax.nn.one_hot(y_flat.reshape(-1), d_state)
    counts = y_oh.sum(axis=0)
    p_marg = counts / jnp.maximum(counts.sum(), 1.0)
    baseline = -jnp.sum(p_marg * jnp.log(p_marg + 1e-10))

    gain = jnp.maximum(0.0, 1.0 - final_loss / jnp.maximum(baseline, 1e-10))
    return gain, final_loss, baseline


def _generate_rule_rollouts(
    seeds, grid, d_state, n_groups, identity_bias, temperature,
    rollout_steps, n_ic, start_step, ic_rng_seed,
):
    """
    Roll out n_ic trajectories per rule from different ICs.
    Returns sims of shape (B, n_ic, T, H, W, G).
    """
    B = seeds.shape[0]
    rule_seeds_tiled = jnp.tile(seeds, (n_ic, 1))
    ic_rng = jax.random.PRNGKey(ic_rng_seed)
    ic_seeds = split(ic_rng, n_ic * B)

    generator = NCA(
        grid_size=grid, d_state=d_state, n_groups=n_groups,
        identity_bias=identity_bias, temperature=temperature,
    )

    def rollout_fn(ic_rng_one, rule_seed_one):
        params = generator.default_params(rule_seed_one)
        return rollout_simulation(
            ic_rng_one, params, substrate=generator,
            rollout_steps=rollout_steps, k_steps=1, time_sampling='video',
            start_step=start_step,
        )

    sims = jax.vmap(rollout_fn, in_axes=(0, 0))(ic_seeds, rule_seeds_tiled)
    return rearrange(sims, "(I B) T H W G -> B I T H W G", B=B, I=n_ic)


def _train_probes_from_rollouts(sims, d_state, n_groups, k, probe_steps, lr, probe_rng_seed):
    """sims: (B, n_ic, T, H, W, G). Trains one probe per rule, vmapped."""
    B = sims.shape[0]
    probe_rngs = split(jax.random.PRNGKey(probe_rng_seed), B)
    train_one = partial(
        _train_probe_one_rule,
        d_state=d_state, n_groups=n_groups, k=k,
        probe_steps=probe_steps, lr=lr,
    )
    return jax.jit(jax.vmap(train_one))(probe_rngs, sims)


def compute_rule_probe_batch(
    seeds: jnp.ndarray,
    grid: int = 12,
    d_state: int = 10,
    n_groups: int = 1,
    identity_bias: float = 0.,
    temperature: float = 1e-4,
    k: int = 8,
    n_ic: int = 4,
    rollout_steps: int = 24,
    start_step: int = 0,
    probe_steps: int = 200,
    lr: float = 1e-2,
    probe_rng_seed: int = 0,
):
    """
    For each candidate rule:
      1. Roll out n_ic trajectories of length `rollout_steps` from different ICs.
      2. Train a tiny Conv3x3 probe to predict s_{t+k} from s_t.
      3. Return probe_gain = 1 - probe_loss / marginal_baseline_loss.

    Returns
    -------
    gain : (B,)        probe gain in [0, 1]
    final_loss : (B,)  raw probe cross-entropy at last step
    baseline   : (B,)  marginal cross-entropy baseline
    """
    assert rollout_steps > k, f"need rollout_steps ({rollout_steps}) > k ({k})"
    sims = _generate_rule_rollouts(
        seeds, grid, d_state, n_groups, identity_bias, temperature,
        rollout_steps, n_ic, start_step, ic_rng_seed=probe_rng_seed + 1,
    )
    return _train_probes_from_rollouts(
        sims, d_state, n_groups, k, probe_steps, lr, probe_rng_seed,
    )


def compute_rule_probe_sweep_batch(
    seeds: jnp.ndarray,
    ks=(1, 2, 4, 8, 16),
    grid: int = 12,
    d_state: int = 10,
    n_groups: int = 1,
    identity_bias: float = 0.,
    temperature: float = 1e-4,
    n_ic: int = 4,
    rollout_steps: int = None,
    start_step: int = 0,
    probe_steps: int = 200,
    lr: float = 1e-2,
    probe_rng_seed: int = 0,
):
    """
    Probe-gain at multiple horizons k. Generates rollouts ONCE and trains
    a fresh probe per (rule, k). The shape of gain-vs-k is the long-horizon
    learnability signal:
      - class IV: gain stays relatively high across k
      - class III: gain near 0 at all k
      - class II : gain high at small k, falls off
      - class I  : gain near 0 (baseline ~ 0)

    Returns
    -------
    gain       : (B, K)  probe gain at each k
    final_loss : (B, K)
    baseline   : (B, K)
    ks         : tuple of ints (same as input)
    """
    ks = tuple(ks)
    max_k = max(ks)
    if rollout_steps is None:
        rollout_steps = max_k + 8
    assert rollout_steps > max_k, f"need rollout_steps ({rollout_steps}) > max k ({max_k})"

    sims = _generate_rule_rollouts(
        seeds, grid, d_state, n_groups, identity_bias, temperature,
        rollout_steps, n_ic, start_step, ic_rng_seed=probe_rng_seed + 1,
    )

    gains, final_losses, baselines = [], [], []
    for k in ks:
        g, fl, bl = _train_probes_from_rollouts(
            sims, d_state, n_groups, k, probe_steps, lr, probe_rng_seed,
        )
        gains.append(g); final_losses.append(fl); baselines.append(bl)

    return (jnp.stack(gains, axis=-1),
            jnp.stack(final_losses, axis=-1),
            jnp.stack(baselines, axis=-1),
            ks)


def _train_probe_one_rule_preq(rng, rollouts, d_state, n_groups, probe_steps, lr):
    """
    Autoregressive (k=1) probe; scores by area under loss curve above the
    final loss (prequential MDL, Finzi et al. arXiv:2601.03220).
    rollouts: (n_ic, T, H, W, G).
    Returns (preq_gain, preq_length, entropy_floor, baseline).
    """
    d_total = d_state * n_groups
    T = rollouts.shape[1]

    x = rollouts[:, :T - 1]
    y = rollouts[:, 1:]

    x_oh = jax.nn.one_hot(x, d_state)
    x_oh = rearrange(x_oh, "I T H W G D -> (I T) H W (G D)")
    y_flat = rearrange(y, "I T H W G -> (I T) H W G")

    probe = _ProbeConv(d_total=d_total)
    rng_init, _ = split(rng)
    params = probe.init(rng_init, x_oh[0])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def forward(p, batch):
        logits = jax.vmap(probe.apply, in_axes=(None, 0))(p, batch)
        return rearrange(logits, "N H W (G D) -> N H W G D", G=n_groups)

    def loss_fn(p):
        logits = forward(p, x_oh)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y_flat).mean()

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = lax.scan(step, (params, opt_state), None, length=probe_steps)
    entropy_floor = losses[-1]

    y_oh = jax.nn.one_hot(y_flat.reshape(-1), d_state)
    counts = y_oh.sum(axis=0)
    p_marg = counts / jnp.maximum(counts.sum(), 1.0)
    baseline = -jnp.sum(p_marg * jnp.log(p_marg + 1e-10))

    # area under loss curve above the floor; clip guards against SGD-noise dips
    preq_length = jnp.sum(jnp.maximum(losses - entropy_floor, 0.0))
    preq_gain = jnp.maximum(
        0.0, preq_length / jnp.maximum(baseline * probe_steps, 1e-10)
    )
    return preq_gain, preq_length, entropy_floor, baseline


def _train_probes_from_rollouts_preq(sims, d_state, n_groups, probe_steps, lr, probe_rng_seed):
    """sims: (B, n_ic, T, H, W, G). One autoregressive probe per rule, vmapped."""
    B = sims.shape[0]
    probe_rngs = split(jax.random.PRNGKey(probe_rng_seed), B)
    train_one = partial(
        _train_probe_one_rule_preq,
        d_state=d_state, n_groups=n_groups,
        probe_steps=probe_steps, lr=lr,
    )
    return jax.jit(jax.vmap(train_one))(probe_rngs, sims)


def compute_rule_probe_preq_batch(
    seeds: jnp.ndarray,
    grid: int = 12,
    d_state: int = 10,
    n_groups: int = 1,
    identity_bias: float = 0.,
    temperature: float = 1e-4,
    n_ic: int = 4,
    rollout_steps: int = 24,
    start_step: int = 0,
    probe_steps: int = 200,
    lr: float = 1e-2,
    probe_rng_seed: int = 0,
):
    """
    Prequential autoregressive variant of compute_rule_probe_batch.

    For each candidate rule:
      1. Roll out n_ic trajectories.
      2. Train a Conv3x3 probe autoregressively (k=1) on all (s_t, s_{t+1}) pairs.
      3. Score by area under the loss curve above the final loss, an analog of
         the prequential description length from Finzi et al. (arXiv:2601.03220).

    Differences vs compute_rule_probe_batch:
      - no fixed-k horizon: k=1 autoregressive over the whole rollout
      - integrates the full loss trajectory, so late-emerging / non-stationary
        structure contributes instead of being washed out by a single
        final-loss readout

    Returns
    -------
    preq_gain     : (B,)  preq_length / (baseline * probe_steps), in [0, 1]
    preq_length   : (B,)  raw area-above-floor, in nats
    entropy_floor : (B,)  final (computationally bounded) cross-entropy
    baseline      : (B,)  marginal cross-entropy baseline
    """
    sims = _generate_rule_rollouts(
        seeds, grid, d_state, n_groups, identity_bias, temperature,
        rollout_steps, n_ic, start_step, ic_rng_seed=probe_rng_seed + 1,
    )
    return _train_probes_from_rollouts_preq(
        sims, d_state, n_groups, probe_steps, lr, probe_rng_seed,
    )