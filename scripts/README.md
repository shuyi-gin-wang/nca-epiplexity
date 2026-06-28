# Scripts

Epiplexity run code now lives under `scripts/epiplexity/`.

## Epiplexity Layout

- `epiplexity/train.py` - recommended launcher for normal runs.
- `epiplexity/trainers/evolve_gzip.py` - main torch ES trainer with gzip gating.
- `epiplexity/trainers/evolve_preq_only.py` - epiplexity-only torch ES trainer.
- `epiplexity/sweeps/student_probe.py` - probe architecture and horizon sweep.
- `epiplexity/sweeps/pdrop0_targets.py` - deterministic gzip-target sweep.
- `epiplexity/render/` - trajectory, GIF, and channel rendering helpers.
- `epiplexity/demos/` - probe/epiplexity scoring sanity checks.
- `epiplexity/analysis/` - plotting helpers for existing logs.
- `epiplexity/legacy/` - older JAX experiments kept for provenance.
- `epiplexity/slurm/` - CoreHPC Slurm launchers.

## Output Layout

New output should live under `scripts/demo_out/runs/`:

- `runs/epiplexity/<run-name>/` for `epiplexity/train.py`.
- `runs/sweeps/<cell>/` for `epiplexity/sweeps/student_probe.py`.
- `runs/pdrop0_long/<cell>/` for `epiplexity/sweeps/pdrop0_targets.py`.

Flat files directly under `scripts/demo_out/` are legacy tracked artifacts from
earlier demos and reference runs. Keep them in place unless deliberately
migrating the committed artifact set.
