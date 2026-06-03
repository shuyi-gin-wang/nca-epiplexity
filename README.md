# Evolving NCAs with Epiplexity

[Project page (gh-pages)](https://shuyi-gin-wang.github.io/nca-epiplexity/)

We evolve Neural Cellular Automata toward increased *learnable complexity*
through an epiplexity heuristic: a fixed linear probe plays the role of a
**bounded student**, and the evolution strategy maximizes the probe's
prequential loss on rollout transitions while a gzip band on the rollout
filters out collapse to static attractors.

Epiplexity (Finzi et al. 2026, [arXiv:2601.03220](https://arxiv.org/abs/2601.03220))
captures the structural information present to a computationally bounded
observer, excluding noise that is otherwise signal to an unbounded observer
under Kolmogorov complexity.

## Setup

- **Teacher NCA** (the evolved rule, ~243 params): 16&times;16 grid, 3 state
  channels, 32-step rollout with circular padding and per-cell Bernoulli(0.5)
  update masking. Architecture `Conv3x3(3->4) -> Conv1x1(4->16) -> ReLU ->
  Conv1x1(16->3)`, applied as `s_{t+1} = clip(s_t + 0.05 * mask * f(s_t), 0, 1)`.
  Evolution mutates these weights.
- **Linear probe** (the fixed student, ~84 params): a single wrap-padded
  `Conv3x3(3->3)` with identity init at the center tap, retrained from scratch
  each evaluation with Adam (lr=1e-2, 50 steps) to predict the next state from
  the current state.
- **Fitness** = `preq_length * exp(-(gzip - target)^2 / (2 * 0.2^2))`, where
  `preq_length` sums the probe's per-step MSE above its final loss floor over
  rollout transitions (high = dynamics not trivially predictable by the probe).
- **(&mu;+&lambda;) ES** with pop=128, elite=16, &sigma;: 0.1 &rarr; ~0.005 over
  600 generations.

## What we found

- The gzip band (byte-level compression) acts as a coarse complexity filter
  that prevents collapse to static attractors. It works well as a floor
  complexity filter (evolutionary search around 0.3 gzip band yields
  low-complexity oscillations) but becomes a constraint for high-epiplexity
  search.
- Low targets give stable, blocky patterns; mid targets give travelling
  structures; high targets give shimmering, noise-like fields.

## Reproducing

Evolution loop:

```bash
.venv/Scripts/python.exe scripts/evolve_nca_preq_gzip_continuous_torch.py \
    --gzip-mode band --gzip-target 0.85 --gzip-width 0.2 \
    --p-drop 0 --run-name torch_pdrop0/evolve_torch_pdrop0_target085
```

Render a long trajectory of the best-ever individual:

```bash
.venv/Scripts/python.exe scripts/render_long_trajectory_torch.py \
    --run-dir scripts/demo_out/runs/torch_pdrop0/evolve_torch_pdrop0_target085 \
    --n-steps 256 --cols 64
```

## Forked from

[danihyunlee/nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining)
&mdash; *Training Language Models via Neural Cellular Automata* (Lee, Han,
Kumar, Agrawal). The original README documenting that codebase is preserved
in git history at commit
[`bdd1c71`](https://github.com/shuyi-gin-wang/nca-epiplexity/blob/bdd1c71/README.md).
