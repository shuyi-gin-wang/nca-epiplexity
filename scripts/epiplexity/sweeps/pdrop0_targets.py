"""
Driver: sweep gzip_target at p_drop=0 (deterministic NCA), with longer gens
and per-100-gen checkpointing.

Mirrors the past scripts/demo_out/runs/torch_pdrop0/* runs (band mode,
p_drop=0, gzip_width=0.2, pop=128) but extends to 1500 generations and
saves a parameter snapshot every 100 gens in <out_dir>/checkpoints/.

Run from repo root:
    python scripts/epiplexity/sweeps/pdrop0_targets.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT, REPO_ROOT

REPO_ROOT = str(REPO_ROOT)
PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parents[1] / "trainers" / "evolve_gzip.py")
OUTPUT_ROOT = str(DEMO_OUT)
RUNS_ROOT = os.path.join(OUTPUT_ROOT, "runs")
PDROP0_ROOT = os.path.join(RUNS_ROOT, "pdrop0_long")
SUMMARY_TSV = os.path.join(PDROP0_ROOT, "summary.tsv")

# 1500 gens with sigma_decay=0.998 ends sigma at ~0.005, matching the
# end-of-run sigma of the previous 600-gen / 0.995-decay sweep.
N_GENERATIONS = 1500
SIGMA_DECAY = 0.998
P_DROP = 0.0
GZIP_WIDTH = 0.2
POP_SIZE = 128
CHECKPOINT_EVERY = 100

# Same target sweep as the original torch_pdrop0 run, named to match.
GZIP_TARGETS = [
    ("03",  0.30),
    ("05",  0.50),
    ("07",  0.70),
    ("085", 0.85),
    ("095", 0.95),
    ("10",  1.00),
]


def cell_name(tag: str) -> str:
    return f"pdrop0_long_target{tag}"


def run_dir(name: str) -> str:
    return os.path.join(PDROP0_ROOT, name)


def trainer_run_name(name: str) -> str:
    return f"runs/pdrop0_long/{name}"


def log_candidates(name: str) -> list[str]:
    log_name = "evolve_nca_preq_gzip_log_torch.tsv"
    return [
        os.path.join(run_dir(name), log_name),
        # Backward compatibility for pre-cleanup target sweep outputs.
        os.path.join(OUTPUT_ROOT, name, log_name),
    ]


def cell_done(name: str) -> bool:
    for log in log_candidates(name):
        if not os.path.exists(log):
            continue
        try:
            with open(log) as f:
                n = sum(1 for _ in f) - 1
            if n >= N_GENERATIONS:
                return True
        except OSError:
            continue
    return False


def run_one(tag: str, target: float, device: str, allow_cpu: bool) -> tuple[int, float]:
    name = cell_name(tag)
    out_dir = run_dir(name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "launch.log")
    cmd = [
        PYTHON, SCRIPT,
        "--run-name", trainer_run_name(name),
        "--gzip-mode", "band",
        "--gzip-target", str(target),
        "--gzip-width", str(GZIP_WIDTH),
        "--p-drop", str(P_DROP),
        "--pop-size", str(POP_SIZE),
        "--n-generations", str(N_GENERATIONS),
        "--sigma-decay", str(SIGMA_DECAY),
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--device", device,
    ]
    if allow_cpu:
        cmd += ["--allow-cpu"]
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write("# cmd: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - t0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device passed to every cell. Defaults to CUDA.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU execution for short/debug sweeps.")
    args = parser.parse_args()

    os.makedirs(PDROP0_ROOT, exist_ok=True)
    print(f"queued {len(GZIP_TARGETS)} cells (pdrop0_long): gens={N_GENERATIONS} ckpt_every={CHECKPOINT_EVERY}")
    for tag, target in GZIP_TARGETS:
        print(f"  {cell_name(tag)}  gzip_target={target}")
    sys.stdout.flush()

    summary_exists = os.path.exists(SUMMARY_TSV)
    with open(SUMMARY_TSV, "a") as sf:
        if not summary_exists:
            sf.write("cell\tgzip_target\tp_drop\tpop\tgens\texit_code\tduration_sec\twall_clock\n")
        sf.flush()
        t0 = time.time()
        for i, (tag, target) in enumerate(GZIP_TARGETS):
            name = cell_name(tag)
            elapsed = (time.time() - t0) / 60
            if cell_done(name):
                print(f"[{i+1}/{len(GZIP_TARGETS)}] SKIP {name} (complete)  (elapsed {elapsed:.1f} min)", flush=True)
                continue
            print(f"[{i+1}/{len(GZIP_TARGETS)}] starting {name}  (elapsed {elapsed:.1f} min)", flush=True)
            rc, dt = run_one(tag, target, args.device, args.allow_cpu)
            wc = time.strftime("%Y-%m-%dT%H:%M:%S")
            sf.write(f"{name}\t{target}\t{P_DROP}\t{POP_SIZE}\t{N_GENERATIONS}\t{rc}\t{dt:.1f}\t{wc}\n")
            sf.flush()
            print(f"  -> exit={rc}  duration={dt:.1f}s", flush=True)
    print(f"pdrop0_long sweep complete. summary: {SUMMARY_TSV}")


if __name__ == "__main__":
    main()
