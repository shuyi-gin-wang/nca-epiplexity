# Epiplexity Scripts

Use `train.py` for normal runs:

```bash
python scripts/epiplexity/train.py --run-name epx_pdrop0_target085
```

Code is grouped by workflow:

- `trainers/` - lower-level torch trainers.
- `sweeps/` - multi-run drivers.
- `render/` - rollout image and GIF helpers.
- `demos/` - scoring and probe sanity checks.
- `analysis/` - plots from existing logs.
- `legacy/` - older JAX experiments kept for provenance.
- `slurm/` - CoreHPC batch launchers.

Outputs should continue to go under `scripts/demo_out/runs/`.
