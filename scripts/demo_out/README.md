# Demo Output

This directory mixes committed reference artifacts with ignored generated runs.

## Current Layout

- `runs/epiplexity/` - new ignored outputs from `scripts/epiplexity/train.py`.
- `runs/sweeps/` - new ignored outputs from `scripts/epiplexity/sweeps/student_probe.py`.
- `runs/pdrop0_long/` - new ignored outputs from `scripts/epiplexity/sweeps/pdrop0_targets.py`.
- `runs/torch_pdrop0/`, `runs/torch_target/`, and `runs/numpy/` - committed reference runs.
- Top-level `probe_*`, `evolve_*`, `train_*`, and `rule_*` files - legacy tracked demo figures and logs.

## Rule Of Thumb

Do not add new flat artifacts here. Give new training runs a bare `--run-name`
and let the launcher place them under `runs/epiplexity/`, or pass an explicit
nested run name such as `runs/my_group/my_run`.
