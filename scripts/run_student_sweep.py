"""
Driver: queue ES runs over (probe_arch, horizon_mode, K) cells back-to-back.

Each config is a separate subprocess invocation of evolve_nca_preq_gzip_continuous_torch.py
so an OOM or other failure in one cell does not poison the rest of the sweep.

Run from repo root:
    .venv/Scripts/python.exe scripts/run_student_sweep.py
Logs land in scripts/demo_out/runs/sweep_<run_name>/launch.log per cell, plus
scripts/demo_out/runs/sweep_summary.tsv listing duration/exit code per cell.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(REPO_ROOT, "scripts", "evolve_nca_preq_gzip_continuous_torch.py")
# evolve script writes to scripts/demo_out/<run_name>/ (legacy flat layout),
# we keep our launch.log + summary in scripts/demo_out/runs/.
OUTPUT_ROOT = os.path.join(REPO_ROOT, "scripts", "demo_out")
RUNS_ROOT = os.path.join(REPO_ROOT, "scripts", "demo_out", "runs")
SUMMARY_TSV = os.path.join(RUNS_ROOT, "sweep_summary.tsv")

# Per-cell budget. Transformer probe forwards are ~200x more expensive than linear
# so we cap them low: still enough gens to see whether the gate moves at all,
# without burning days on a single cell.
N_GENERATIONS_DEFAULT = 200
N_GENERATIONS_PER_ARCH = {
    "transformer": 50,
}


def gens_for(arch: str) -> int:
    return N_GENERATIONS_PER_ARCH.get(arch, N_GENERATIONS_DEFAULT)


def cell_done(name: str, expected_gens: int) -> bool:
    """A cell is considered complete iff its log_tsv has reached the requested
    generation count. Cheap check that lets us safely restart the driver."""
    log = os.path.join(OUTPUT_ROOT, name, "evolve_nca_preq_gzip_log_torch.tsv")
    if not os.path.exists(log):
        return False
    try:
        with open(log) as f:
            n = sum(1 for _ in f) - 1  # minus header
        return n >= expected_gens
    except OSError:
        return False


def cell_name(arch: str, mode: str, K: int, suffix: str = "") -> str:
    if mode == "multi":
        base = f"sweep_{arch}_multi"
    else:
        base = f"sweep_{arch}_{mode}_K{K}"
    return base + suffix


def build_sweep():
    """List of dicts: each is one subprocess call's args.

    Order: long-horizon first (the user's primary interest), short-horizon mid,
    K=1 baseline last (already done for MLPs; transformer K=1 deferred since it's
    the slowest and not what we're focused on).

    pop_size is shrunk for transformer because vmap-stacked autograd graphs through
    iterated probe forwards OOM at pop=128 on an 8 GB GPU.
    """
    sweep = []

    # ---- Phase A: long iterated horizon (autoregressive K=16) ----
    # Probe is rolled out 16 steps and must match the NCA's 16-step trajectory.
    for arch in ("linear", "mlp_small", "mlp_wide"):
        sweep.append(dict(arch=arch, mode="autoregressive", K=16, pop=128, gens=gens_for(arch)))
    sweep.append(dict(arch="deep_mlp", mode="autoregressive", K=16, pop=64, gens=gens_for("deep_mlp")))

    # ---- Phase B: long skip-step horizon (direct K=16) ----
    # Probe predicts state[t+16] in a single forward pass — transformer fits here
    # because there's no iterated-forward autograd chain.
    for arch in ("linear", "mlp_small", "mlp_wide", "deep_mlp", "transformer"):
        sweep.append(dict(arch=arch, mode="direct", K=16, pop=128, gens=gens_for(arch)))

    # ---- Phase C: multi-horizon (autoregressive internally up to Kmax) ----
    for arch in ("linear", "mlp_small", "mlp_wide", "deep_mlp"):
        sweep.append(dict(arch=arch, mode="multi", K=1, pop=128, gens=gens_for(arch),
                          multi_ks="1,2,4,8,16"))
    sweep.append(dict(arch="transformer", mode="multi", K=1, pop=32, gens=gens_for("transformer"),
                      multi_ks="1,2,4"))

    # ---- Phase D: mid iterated horizon (autoregressive K=4) ----
    for arch in ("linear", "mlp_small", "mlp_wide", "deep_mlp"):
        sweep.append(dict(arch=arch, mode="autoregressive", K=4, pop=128, gens=gens_for(arch)))
    sweep.append(dict(arch="transformer", mode="autoregressive", K=4, pop=32, gens=gens_for("transformer")))

    # ---- Phase E: mid skip-step horizon (direct K=8) ----
    for arch in ("linear", "mlp_small", "mlp_wide", "deep_mlp", "transformer"):
        sweep.append(dict(arch=arch, mode="direct", K=8, pop=128, gens=gens_for(arch)))

    # ---- Phase F: K=1 transformer baseline (deferred — slowest, lowest priority) ----
    # MLP K=1 cells are already complete from the earlier sweep.
    sweep.append(dict(arch="transformer", mode="autoregressive", K=1, pop=64, gens=gens_for("transformer")))

    # ---- Phase G: K=1 MLP baselines ----
    # Only run when this sweep config hasn't seen them yet (e.g. fresh p_drop=0
    # rerun). cell_done() will skip them if a prior sweep already produced the
    # log_tsv, so adding them here is safe to leave on.
    for arch in ("linear", "mlp_small", "mlp_wide", "deep_mlp"):
        sweep.append(dict(arch=arch, mode="autoregressive", K=1, pop=128, gens=gens_for(arch)))

    return sweep


def run_one(cfg, p_drop: float, suffix: str) -> tuple[int, float]:
    name = cell_name(cfg["arch"], cfg["mode"], cfg["K"], suffix)
    out_dir = os.path.join(RUNS_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "launch.log")

    cmd = [
        PYTHON, SCRIPT,
        "--run-name", name,
        "--probe-arch", cfg["arch"],
        "--horizon-mode", cfg["mode"],
        "--probe-horizon", str(cfg["K"]),
        "--pop-size", str(cfg["pop"]),
        "--n-generations", str(cfg["gens"]),
        "--p-drop", str(p_drop),
    ]
    if "multi_ks" in cfg:
        cmd += ["--multi-ks", cfg["multi_ks"]]

    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write("# cmd: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return proc.returncode, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-drop", type=float, default=0.5,
                        help="Per-cell Bernoulli update-mask drop probability passed to every cell. "
                             "Default 0.5 matches the original sweep.")
    parser.add_argument("--name-suffix", type=str, default="",
                        help="Appended to every cell's run name (e.g. '_pdrop0'). Lets us run "
                             "alternate-config sweeps without clobbering existing checkpoints.")
    args = parser.parse_args()
    p_drop = args.p_drop
    suffix = args.name_suffix

    os.makedirs(RUNS_ROOT, exist_ok=True)
    sweep = build_sweep()
    # Keep the legacy sweep_summary.tsv schema for the default run; new schemas
    # land in suffixed files so we don't corrupt the original analyses.
    summary_path = (
        SUMMARY_TSV if not suffix
        else os.path.join(RUNS_ROOT, f"sweep_summary{suffix}.tsv")
    )
    print(f"queued {len(sweep)} cells in sweep (p_drop={p_drop}, suffix='{suffix}', summary={summary_path}):")
    for i, cfg in enumerate(sweep):
        print(f"  [{i:02d}] {cell_name(cfg['arch'], cfg['mode'], cfg['K'], suffix):44s}  pop={cfg['pop']} gens={cfg['gens']}")
    sys.stdout.flush()

    # Append-mode summary so a partial sweep + restart accumulates results.
    summary_exists = os.path.exists(summary_path)
    with open(summary_path, "a") as sf:
        if not summary_exists:
            sf.write("cell\tarch\tmode\tK\tpop\tgens\tp_drop\texit_code\tduration_sec\twall_clock\n")
        sf.flush()

        sweep_t0 = time.time()
        for i, cfg in enumerate(sweep):
            name = cell_name(cfg["arch"], cfg["mode"], cfg["K"], suffix)
            elapsed_total = time.time() - sweep_t0
            if cell_done(name, cfg["gens"]):
                print(f"[{i+1}/{len(sweep)}] SKIP {name} (already complete)  (sweep elapsed {elapsed_total/60:.1f} min)", flush=True)
                continue
            print(f"[{i+1}/{len(sweep)}] starting {name}  (sweep elapsed {elapsed_total/60:.1f} min)", flush=True)
            rc, dt = run_one(cfg, p_drop, suffix)
            wc = time.strftime("%Y-%m-%dT%H:%M:%S")
            sf.write(f"{name}\t{cfg['arch']}\t{cfg['mode']}\t{cfg['K']}\t{cfg['pop']}\t{cfg['gens']}\t{p_drop}\t{rc}\t{dt:.1f}\t{wc}\n")
            sf.flush()
            print(f"  -> exit={rc}  duration={dt:.1f}s", flush=True)

    print(f"sweep complete. summary: {summary_path}")


if __name__ == "__main__":
    main()
