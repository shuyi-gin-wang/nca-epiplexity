"""
Epiplexity-first training launcher.

This is a thin, stable wrapper around
scripts/epiplexity/trainers/evolve_gzip.py. In the older code and TSV
logs, the epiplexity score is named `preq_length`; this launcher keeps the
old trainer intact while making the intended workflow easier to discover.

Examples
--------
Quick GPU check:
    python scripts/epiplexity/train.py --run-name epx_quick_check --generations 3 --pop-size 16

CPU-only quick check:
    python scripts/epiplexity/train.py --run-name epx_quick_check_cpu --generations 3 --pop-size 16 --device cpu --allow-cpu

Default deterministic epiplexity run:
    python scripts/epiplexity/train.py --run-name epx_pdrop0_target085

Bare run names are stored under scripts/demo_out/runs/epiplexity/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT, add_repo_root_to_path

add_repo_root_to_path()


def _float_tag(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _clean_relative_path(value: str, label: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    parts = normalized.split("/")
    if ":" in normalized or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a relative path under scripts/demo_out")
    return "/".join(parts)


def _grouped_run_name(run_name: str, output_group: str) -> str:
    run_name = _clean_relative_path(run_name, "--run-name")
    output_group = _clean_relative_path(output_group, "--output-group")
    if output_group and "/" not in run_name:
        return f"{output_group}/{run_name}"
    return run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch epiplexity training for continuous NCAs. The trainer evolves "
            "NCA rules to maximize prequential probe length, optionally gated by "
            "gzip complexity."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the trainer.")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command without running it.")

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run directory name. Bare names are stored under "
            "scripts/demo_out/runs/epiplexity/."
        ),
    )
    parser.add_argument(
        "--output-group",
        default="runs/epiplexity",
        help="Grouping folder under scripts/demo_out/ for bare run names. Use '' for top-level output.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device. Defaults to CUDA.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution for short/debug runs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="If >0, evaluate population fitness in chunks to reduce peak GPU memory.")
    parser.add_argument("--generations", type=int, default=600)
    parser.add_argument("--grid", type=int, default=16, help="NCA grid side length used for rollout/evaluation.")
    parser.add_argument("--rollout-steps", type=int, default=32, help="NCA rollout length used for probe scoring.")
    parser.add_argument("--p-drop", type=float, default=0.0, help="Per-cell update-mask drop probability.")

    parser.add_argument("--gzip-mode", choices=["band", "threshold"], default="band")
    parser.add_argument("--target", type=float, default=0.85, help="Band-mode gzip target.")
    parser.add_argument("--width", type=float, default=0.2, help="Band-mode gzip Gaussian width.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold-mode gzip floor.")

    parser.add_argument("--sigma-init", type=float, default=0.1)
    parser.add_argument("--sigma-decay", type=float, default=0.995)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume-from", default=None, help="Path to a best_ever_params.pt checkpoint.")

    parser.add_argument(
        "--probe-arch",
        choices=["linear", "mlp_small", "mlp_wide", "deep_mlp", "transformer"],
        default="linear",
        help="Student probe architecture used inside the epiplexity score.",
    )
    parser.add_argument(
        "--probe-archs",
        default="",
        help=(
            "Comma-separated probe ensemble. If set, it overrides --probe-arch "
            "and the trainer combines epiplexity scores across these probes."
        ),
    )
    parser.add_argument(
        "--probe-combine",
        choices=["mean", "min", "max", "hmean"],
        default="mean",
        help="How to combine per-probe epiplexity scores when --probe-archs is set.",
    )
    parser.add_argument("--learnability-floor", type=float, default=0.0, help="If >0, gate fitness by per-probe learning gains.")
    parser.add_argument("--probe-hidden", type=int, default=0, help="0 means use the probe default.")
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--horizon-mode", choices=["autoregressive", "direct", "multi"], default="autoregressive")
    parser.add_argument("--probe-horizon", type=int, default=1)
    parser.add_argument("--multi-ks", default="1,2,4,8,16", help="Comma-separated horizons for multi mode.")
    return parser


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    trainer = Path(__file__).resolve().parent / "trainers" / "evolve_gzip.py"
    if args.run_name:
        base_run_name = args.run_name
    elif args.gzip_mode == "band":
        base_run_name = f"epx_pdrop{_float_tag(args.p_drop)}_target{_float_tag(args.target)}"
    else:
        base_run_name = f"epx_pdrop{_float_tag(args.p_drop)}_gate{_float_tag(args.threshold)}"
    run_name = _grouped_run_name(base_run_name, args.output_group)

    cmd = [
        args.python,
        str(trainer),
        "--run-name",
        run_name,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--pop-size",
        str(args.pop_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--n-generations",
        str(args.generations),
        "--grid",
        str(args.grid),
        "--rollout-steps",
        str(args.rollout_steps),
        "--p-drop",
        str(args.p_drop),
        "--gzip-mode",
        args.gzip_mode,
        "--sigma-init",
        str(args.sigma_init),
        "--sigma-decay",
        str(args.sigma_decay),
        "--probe-arch",
        args.probe_arch,
        "--probe-combine",
        args.probe_combine,
        "--learnability-floor",
        str(args.learnability_floor),
        "--probe-hidden",
        str(args.probe_hidden),
        "--probe-lr",
        str(args.probe_lr),
        "--horizon-mode",
        args.horizon_mode,
        "--probe-horizon",
        str(args.probe_horizon),
        "--multi-ks",
        args.multi_ks,
    ]

    if args.gzip_mode == "band":
        cmd += ["--gzip-target", str(args.target), "--gzip-width", str(args.width)]
    else:
        cmd += ["--gzip-threshold", str(args.threshold)]

    if args.probe_archs:
        cmd += ["--probe-archs", args.probe_archs]
    if args.checkpoint_every > 0:
        cmd += ["--checkpoint-every", str(args.checkpoint_every)]
    if args.resume_from:
        cmd += ["--resume-from", args.resume_from]
    if args.allow_cpu:
        cmd += ["--allow-cpu"]

    return cmd + passthrough


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    cmd = build_command(args, passthrough)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

    print("Epiplexity launcher delegates to:")
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
