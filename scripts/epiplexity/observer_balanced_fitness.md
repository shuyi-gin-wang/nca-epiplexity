# Observer-Balanced Fitness

This note describes the fitness function used by the current observer-balanced
NCA run:

```text
epx_observer_balanced_hmean_gain015_directK16_floor03_g1500_h100
```

The conceptual goal is to evolve an NCA "universe" whose dynamics contain
observable structure for multiple observers with different capacities. A
successful rollout should not only look non-static or noisy; it should expose
learnable patterns to a linear observer, medium-capacity MLP observers, and a
larger transformer observer at the same time.

## Top-Level Formula

For candidate NCA `i`, the current fitness is:

```text
fitness_i = hmean(preq_i,linear,
                  preq_i,mlp_wide,
                  preq_i,deep_mlp,
                  preq_i,transformer)
            * learnability_gate_i
            * gzip_gate_i
```

With the current Slurm config:

```text
probes              = linear, mlp_wide, deep_mlp, transformer
probe combination   = harmonic mean
learnability floor  = 0.15
gzip threshold      = 0.3
horizon mode        = direct
probe horizon       = K=16
population size     = 128
elites per gen      = 16
generations         = 1500
```

## NCA Rollout

Each candidate NCA is rolled out from sampled random initial conditions. The
probe observers do not see the NCA parameters. They only receive states from the
rollout.

The current run uses:

```text
grid          = 16 x 16
state dims    = 3 channels
rollout steps = 32
p_drop        = 0.0
dt            = 0.05
```

For each candidate, the rollout produces a sequence:

```text
x_0, x_1, x_2, ..., x_31
```

## Probe Prediction Task

Each probe is trained on the same rollout. In the current run the horizon mode
is direct with `K=16`, so the probe learns:

```text
state at time t  ->  state at time t + 16
```

That is different from autoregressive prediction. The probe is not applied 16
times in a loop. It makes one direct prediction 16 steps into the future.

For a probe `p`, the supervised pairs are:

```text
input  = x_t
target = x_{t+16}
```

The probe is freshly initialized for scoring and trained for 50 Adam steps.
During those 50 steps we record its MSE loss curve:

```text
loss_p,0, loss_p,1, ..., loss_p,49
```

## Per-Probe Epiplexity / Prequential Score

For each probe, the code computes:

```text
final_loss_p = loss_p,49

preq_p = sum over training step s:
           max(loss_p,s - final_loss_p, 0)
```

This is the "epiplexity" or prequential-length score used by the experiment.
It is the area above the final probe loss during the probe's learning process.

Interpretation:

```text
high preq_p:
  The probe initially makes errors, then learns useful predictive structure.

low preq_p because losses are flat and low:
  The rollout is too trivial for that probe.

low preq_p because losses are flat and high:
  The rollout is not learnable by that probe.
```

So the score is not simply "final prediction error". It rewards dynamics that
create a learnable learning curve for the observer.

## Combining Observers

The current run combines the four probe scores with a harmonic mean:

```text
observer_score_i = 4 / (
    1 / max(preq_i,linear, eps)
  + 1 / max(preq_i,mlp_wide, eps)
  + 1 / max(preq_i,deep_mlp, eps)
  + 1 / max(preq_i,transformer, eps)
)
```

The harmonic mean is intentionally stricter than a normal average. It punishes
the candidate when any observer sees little learnable structure.

Example:

```text
linear      = 10
mlp_wide    = 10
deep_mlp    = 10
transformer = 1

mean  = 7.75
hmean = 3.08
```

This makes the objective prefer worlds that are observable across the whole
capacity ladder, not worlds that only one powerful probe can exploit.

## Learnability Gate

The learnability gate checks whether the probes actually improved during their
inner training loop.

For each probe:

```text
gain_p = (initial_loss_p - final_loss_p) / max(initial_loss_p, eps)
gain_p = clamp(gain_p, 0, 1)
```

Then:

```text
learnability_gate_i = mean over probes:
    clamp(gain_p / 0.15, 0, 1)
```

If every probe improves by at least 15 percent, the gate is 1.0. If a probe
does not improve, it pulls the gate down.

In the current run logs, this has usually been:

```text
learn=1.000
```

That means the probes are clearing the 15 percent improvement floor. The gate is
not currently the bottleneck; the harmonic mean of the per-probe preq scores is.

## Gzip Floor Gate

The gzip score is computed by quantizing the rollout to bytes and measuring:

```text
gzip_ratio = compressed_bytes / raw_bytes
```

The current run uses threshold mode:

```text
gzip_gate_i = 1 if gzip_ratio_i > 0.3 else 0
gzip_gate_i = 0 otherwise
```

This is only a floor gate. It does not reward higher gzip once the candidate is
above 0.3. It exists to remove collapsed or overly static rollouts from
selection.

Important log-reading detail:

```text
gen 330 best_comb=6.8823 (preq_hmean=6.882 * learn=1.000 * gzip=0.307)
```

The displayed `gzip=0.307` is the raw gzip ratio. Since it is greater than
0.3, the multiplier used in fitness is `1`, not `0.307`.

So this line means:

```text
fitness = 6.882 * 1.000 * 1
```

not:

```text
fitness = 6.882 * 1.000 * 0.307
```

## Selection

After all candidates in the population are scored:

```text
combined_i = observer_score_i * learnability_gate_i * gzip_gate_i
```

The candidates are sorted by `combined_i`. The top 16 of 128 become elites.
The next generation is made by keeping those elites and filling the rest of the
population with mutated copies of them.

Mutation scale starts at:

```text
sigma = 0.1
```

and decays by:

```text
sigma = sigma * 0.998
```

each generation.

## Why This Matches The Observer Goal

The NCA is treated as the universe. The probes are observers with different
capacities:

```text
linear      = low-capacity observer
mlp_wide    = medium-capacity observer
deep_mlp    = deeper nonlinear observer
transformer = high-capacity observer
```

The objective asks for a rollout where:

```text
1. Each observer can learn something predictive.
2. No observer is allowed to be ignored.
3. The rollout is not collapsed/static under gzip.
4. Complexity is selected through observer-relative learnability, not raw noise.
```

That is why the current objective is better aligned with "observable complexity
at multiple levels" than either a single transformer probe or a simple average
over probes.

