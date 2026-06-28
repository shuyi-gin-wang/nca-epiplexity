# NCA Epiplexity

[Project page (gh-pages)](https://shuyi-gin-wang.github.io/nca-epiplexity/)

This repo is now centered on **epiplexity training** for neural cellular
automata (NCA): evolve NCA rules whose dynamics are rich, structured, and hard
for a weak predictive probe to compress quickly.

The original paper pipeline, **Training Language Models via Neural Cellular
Automata**, is still present under `src/` and the legacy launch scripts. The
newer work in this checkout focuses on finding better NCA training signals by
optimizing an epiplexity objective directly.

Epiplexity (Finzi et al. 2026, [arXiv:2601.03220](https://arxiv.org/abs/2601.03220))
captures the structural information present to a computationally bounded
observer, excluding noise that is otherwise signal to an unbounded observer
under Kolmogorov complexity.

## What Epiplexity Means Here

Most older files still use the implementation name `preq_length`. In this repo:

- `preq_length` is the epiplexity score.
- A small student probe is trained to predict NCA state transitions.
- The score is the area of the probe's loss curve above its final loss.
- Higher score means the rule takes more probe-training effort to describe.
- Gzip complexity is used as a gate or band so the search avoids trivial static
  collapse and pure noise.

The current main trainer is:

```bash
python scripts/epiplexity/train.py
```

That launcher delegates to the lower-level torch evolution script:

```bash
scripts/epiplexity/trainers/evolve_gzip.py
```

The lower-level filename is kept because the committed artifacts and TSV logs
were generated with the older `preq` terminology.

## Quick Start

Epiplexity training is CUDA-first. The default commands require a CUDA-enabled
PyTorch runtime and will fail fast if `torch.cuda.is_available()` is false.

Create the environment:

```bash
mamba env create -f environment.yml
mamba activate ai2
```

Run a tiny GPU check:

```bash
python scripts/epiplexity/train.py --run-name epx_quick_check --generations 3 --pop-size 16
```

On a CPU-only machine, short checks must opt in explicitly:

```bash
python scripts/epiplexity/train.py --run-name epx_quick_check_cpu --generations 3 --pop-size 16 --device cpu --allow-cpu
```

Run the default deterministic epiplexity recipe:

```bash
python scripts/epiplexity/train.py --run-name epx_pdrop0_target085
```

The default recipe uses:

- deterministic NCA updates: `--p-drop 0`
- gzip band mode: `--gzip-mode band`
- gzip target: `--target 0.85`
- population size: `128`
- generations: `600`
- student probe: `linear`

That default is chosen because the committed `torch_pdrop0` artifacts show the
strongest 600-generation result near gzip target `0.85`.

Useful variants:

```bash
# Higher-complexity target
python scripts/epiplexity/train.py --run-name epx_pdrop0_target10 --target 1.0 --generations 3000 --sigma-decay 0.999

# Gate out static dynamics but do not penalize high gzip
python scripts/epiplexity/train.py --run-name epx_gate03 --gzip-mode threshold --threshold 0.3

# Stronger student probe
python scripts/epiplexity/train.py --run-name epx_mlp_direct16 --probe-arch mlp_wide --horizon-mode direct --probe-horizon 16
```

Bare run names land in `scripts/demo_out/runs/epiplexity/<run-name>/`.
Pass a nested `--run-name`, such as `runs/my_group/my_run`, to choose another
group under `scripts/demo_out/`.

Outputs usually include:

- `evolve_nca_preq_gzip_log_torch.tsv` - per-generation metrics
- `best_ever_params.pt` - best evolved NCA parameters
- `evolve_nca_preq_gzip_curve_torch.png` - fitness and component curves
- `probe_curves_best.npy` / `probe_curves_mean.npy` - probe-learning traces
- rollout images or GIFs when rendering scripts are used

## Training Workflows

### 1. Single Epiplexity Run

Use `scripts/epiplexity/train.py` for normal work. It exposes the main knobs
without requiring you to remember the older trainer name.

The delegated trainer supports:

- `--gzip-mode band` for a Gaussian complexity target
- `--gzip-mode threshold` for a minimum-complexity gate
- `--p-drop` for stochastic vs deterministic NCA updates
- `--probe-arch linear|mlp_small|mlp_wide|deep_mlp|transformer`
- `--horizon-mode autoregressive|direct|multi`
- `--resume-from path/to/best_ever_params.pt`
- `--checkpoint-every N`
- `--device cuda` by default; use `--device cuda:N` to choose a GPU
- `--allow-cpu` only for short checks/debugging

### 2. Student-Probe Sweep

To test whether evolved rules remain hard for stronger probes and longer
horizons:

```bash
python scripts/epiplexity/sweeps/student_probe.py
```

The sweep covers linear, MLP, deep MLP, and small transformer probes across
autoregressive, direct, and multi-horizon modes. New outputs are grouped under
`scripts/demo_out/runs/sweeps/`.

### 3. Deterministic Target Sweep

To repeat the long p-drop 0 gzip-target sweep:

```bash
python scripts/epiplexity/sweeps/pdrop0_targets.py
```

This queues target values `0.30`, `0.50`, `0.70`, `0.85`, `0.95`, and `1.00`
with longer schedules and periodic checkpoints. New outputs are grouped under
`scripts/demo_out/runs/pdrop0_long/`.

## Repo Map

- `scripts/epiplexity/train.py` - recommended epiplexity launcher
- `scripts/epiplexity/trainers/evolve_gzip.py` - main torch ES trainer with gzip gating
- `scripts/epiplexity/trainers/evolve_preq_only.py` - epiplexity-only torch ES trainer
- `scripts/epiplexity/sweeps/student_probe.py` - student-probe architecture/horizon sweep
- `scripts/epiplexity/sweeps/pdrop0_targets.py` - deterministic gzip-target sweep
- `scripts/epiplexity/render/` - trajectory and channel rendering helpers
- `scripts/epiplexity/demos/` - probe/epiplexity scoring sanity checks
- `scripts/epiplexity/analysis/` - plotting helpers for existing logs
- `scripts/epiplexity/legacy/` - older JAX epiplexity experiments
- `utils/probes_torch.py` - student probe registry
- `utils/nca_torch.py` - torch continuous NCA and probe utilities
- `utils/nca.py` - original JAX/Flax NCA, gzip, and probe-score utilities
- `scripts/demo_out/` - committed reference plots, logs, and checkpoints

## Reading The Existing Results

The committed artifacts are reference runs, not a tidy benchmark suite. The
most useful folders are:

- `scripts/demo_out/runs/epiplexity/` - ignored outputs from new launcher runs
- `scripts/demo_out/runs/sweeps/` - ignored outputs from student-probe sweeps
- `scripts/demo_out/runs/pdrop0_long/` - ignored outputs from long target sweeps
- `scripts/demo_out/runs/torch_pdrop0/` - deterministic gzip-gated ES runs
- `scripts/demo_out/runs/torch_target/` - p-drop 0.5 target runs
- `scripts/demo_out/runs/numpy/` - early torch/JAX comparison and preq-only runs

Older flat files directly under `scripts/demo_out/` are kept as historical
demo/reference artifacts. New training output should go under
`scripts/demo_out/runs/`.

For gzip-gated logs, columns are:

```text
gen best_combined mean_combined best_preq best_gzip best_id_base sigma
```

Interpret `best_preq` as epiplexity. `best_combined` is epiplexity multiplied
by the gzip gate or band score.

## Legacy Language-Model Pipeline

The original NCA pre-pretraining pipeline is still available:

- NCA pre-pretraining: `scripts/prepretraining/nca_prepretraining.sh`
- OpenWebText continuation: `scripts/pretraining/owt_ft.sh`
- CodeParrot continuation: `scripts/pretraining/ft_codeparrot.sh`
- GSM8K fine-tuning: `scripts/instruction-ft/ft_instruction_gsm8k.sh`
- BigBench-Lite fine-tuning: `scripts/instruction-ft/ft_instruction_bbl.sh`
- Evaluation scripts: `scripts/eval/`

Those launchers are mostly templates with placeholder paths. They call into
`src/nca_ppt.py`, `src/openwebtext_pt.py`, and `src/language_train.py`.

## Notes For Future Cleanup

- Keep old `preq_*` filenames until the committed artifacts are migrated or
  regenerated; the historical names make provenance easier.
- New generated training and sweep outputs are ignored by default so git
  history does not fill with large GIFs and checkpoints.
- Existing tracked plots and TSVs under `scripts/demo_out/` are retained as
  reference evidence for the current direction.

## Forked From

[danihyunlee/nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining)
- *Training Language Models via Neural Cellular Automata* (Lee, Han, Kumar,
Agrawal). The original README documenting that codebase is preserved in git
history at commit
[`bdd1c71`](https://github.com/shuyi-gin-wang/nca-epiplexity/blob/bdd1c71/README.md).

## Citation

```bibtex
@misc{lee2026traininglanguagemodelsneural,
      title={Training Language Models via Neural Cellular Automata},
      author={Dan Lee and Seungwook Han and Akarsh Kumar and Pulkit Agrawal},
      year={2026},
      eprint={2603.10055},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.10055},
}
```
