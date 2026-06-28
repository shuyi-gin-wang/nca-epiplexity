"""Zoom into the preq_length phase transition for an evolve-torch run.

Loads probe_curves_best.npy + log TSV from a run dir, auto-detects the
generations where best_preq rises from its initial plateau to its final
plateau, and renders:

  - preq_length over generations with the 10%-90% transition window shaded
    and a twin axis for initial_loss (L_0) vs probe floor (L_49); makes the
    "init loss grows, floor lags" picture explicit.
  - best-of-gen probe learning curves at ~10 evenly spaced generations
    inside the transition window, color-coded by gen, log-y.

Run from repo root, defaults to the v2 run:
    .venv/Scripts/python.exe scripts/epiplexity/analysis/plot_probe_transition.py
or:
    .venv/Scripts/python.exe scripts/epiplexity/analysis/plot_probe_transition.py \\
        --run-dir scripts/demo_out/runs/torch_pdrop0/evolve_torch_pdrop0_target03
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT


def find_transition_window(preq, lo_q=0.1, hi_q=0.9, smooth=11):
    """Return (gen_lo, gen_hi): smallest interval where smoothed best_preq
    crosses from (min + lo_q * span) up to (min + hi_q * span)."""
    k = np.ones(smooth) / smooth
    smoothed = np.convolve(preq, k, mode="same")
    lo = smoothed.min() + lo_q * (smoothed.max() - smoothed.min())
    hi = smoothed.min() + hi_q * (smoothed.max() - smoothed.min())
    above_lo = np.where(smoothed >= lo)[0]
    above_hi = np.where(smoothed >= hi)[0]
    g_lo = int(above_lo[0]) if len(above_lo) else 0
    g_hi = int(above_hi[0]) if len(above_hi) else len(preq) - 1
    return g_lo, g_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default=os.path.join(
            str(DEMO_OUT), "runs", "torch_target", "evolve_torch_target095_v2",
        ),
    )
    ap.add_argument("--n-snapshots", type=int, default=10,
                    help="number of probe curves to overlay through the transition")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                    help="fraction of the transition width to pad the sampling window on each side")
    args = ap.parse_args()

    run_dir = args.run_dir
    log_path = os.path.join(run_dir, "evolve_nca_preq_gzip_log_torch.tsv")
    curves_path = os.path.join(run_dir, "probe_curves_best.npy")
    if not os.path.exists(curves_path):
        # Non-gzip runs use a different log name; try the plain preq log too.
        log_alt = os.path.join(run_dir, "evolve_nca_preq_log_torch.tsv")
        if os.path.exists(log_alt):
            log_path = log_alt
    if not (os.path.exists(log_path) and os.path.exists(curves_path)):
        raise SystemExit(
            f"missing log or probe_curves_best.npy in {run_dir}.\n"
            f"  expected: {log_path}  +  {curves_path}"
        )

    curves = np.load(curves_path)  # (n_gens, probe_steps)
    n_gens, probe_steps = curves.shape
    log = np.genfromtxt(log_path, delimiter="\t", names=True)
    preq = log["best_preq"].astype(float)
    gens = log["gen"].astype(int)

    initial_loss = curves[:, 0]
    floor = curves[:, -1]

    g_lo, g_hi = find_transition_window(preq)
    span = g_hi - g_lo
    pad = int(args.pad_frac * max(span, 10))
    snap_lo = max(0, g_lo - pad)
    snap_hi = min(n_gens - 1, g_hi + pad)
    snap_gens = np.linspace(snap_lo, snap_hi, args.n_snapshots).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5),
                           gridspec_kw={"width_ratios": [1.1, 1.0]})

    # left: preq trajectory + transition window + twin axis for L_0 / L_49
    ax0 = ax[0]
    ax0.plot(gens, preq, color="tab:purple", label="best preq_length")
    ax0.axvspan(g_lo, g_hi, color="tab:orange", alpha=0.15,
                label=f"transition (gens {g_lo}-{g_hi})")
    for g in snap_gens:
        ax0.axvline(g, color="tab:gray", lw=0.5, alpha=0.5)
    ax0.set_xlabel("generation")
    ax0.set_ylabel("preq_length", color="tab:purple")
    ax0.tick_params(axis="y", labelcolor="tab:purple")
    ax0.set_title("preq_length phase transition")
    ax0.legend(loc="upper left", fontsize=8)

    ax0b = ax0.twinx()
    ax0b.plot(gens, initial_loss, color="tab:red", lw=1.0, alpha=0.7,
              label="initial loss $L_0$")
    ax0b.plot(gens, floor, color="tab:blue", lw=1.0, alpha=0.7,
              label="probe floor $L_{49}$")
    ax0b.set_ylabel("probe MSE")
    ax0b.legend(loc="lower right", fontsize=8)

    # right: probe curves at sampled gens through transition window
    ax1 = ax[1]
    cmap = plt.get_cmap("viridis")
    for j, gi in enumerate(snap_gens):
        color = cmap(j / max(1, len(snap_gens) - 1))
        ax1.plot(curves[gi], color=color, lw=1.2)
    ax1.set_xlabel("probe SGD step")
    ax1.set_ylabel("probe MSE")
    ax1.set_title(
        f"best-of-gen probe curves across transition\n"
        f"(gens {snap_gens[0]} -> {snap_gens[-1]})"
    )
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=snap_gens[0], vmax=snap_gens[-1])
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, pad=0.02)
    cb.set_label("generation")

    fig.tight_layout()
    out_path = os.path.join(run_dir, "probe_curves_transition.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")
    print(
        f"  transition window: gens {g_lo}-{g_hi}  "
        f"(preq {preq[g_lo]:.3f} -> {preq[g_hi]:.3f}, span {g_hi-g_lo} gens)"
    )
    print(
        f"  snapshots at: {list(snap_gens)}"
    )


if __name__ == "__main__":
    main()
