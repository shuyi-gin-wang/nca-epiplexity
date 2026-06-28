"""
Regenerate the plots that previously used log-scale axes, on a linear axis,
from the TSV / npy outputs already on disk. Avoids re-running the heavy sweeps.

Regenerates:
  - probe_gain_vs_k.png
  - probe_gain_vs_k_by_quadrant.png
  - probe_preq_continuous_torch_sweep_scatter.png
  - probe_curves_snapshots.png in every evolve_torch* run dir that has the npy
"""
import csv
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT

OUT_DIR = str(DEMO_OUT)

QUADRANTS = {
    "IV_target":  ("tab:green",  "high gzip  &  high gain"),
    "III_chaos":  ("tab:red",    "high gzip  &  low gain"),
    "II_ordered": ("tab:orange", "low gzip   &  high gain"),
    "I_fixed":    ("tab:gray",   "low gzip   &  low gain"),
}


def pick_exemplar(idx_list, gain_curve, gzip_scores, name):
    if not idx_list:
        return None
    last = gain_curve[:, -1]
    if name == "IV_target":
        scores = [float(last[i] + gzip_scores[i]) for i in idx_list]
    elif name == "III_chaos":
        scores = [float(gzip_scores[i] - last[i]) for i in idx_list]
    elif name == "II_ordered":
        scores = [float(last[i] - gzip_scores[i]) for i in idx_list]
    else:
        scores = [-float(gzip_scores[i] + last[i]) for i in idx_list]
    return idx_list[int(np.argmax(np.array(scores)))]


def replot_probe_gain_vs_k():
    tsv_path = os.path.join(OUT_DIR, "probe_sweep_scores.tsv")
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    k_cols = [c for c in rows[0].keys() if c.startswith("gain_k")]
    KS = [int(c.replace("gain_k", "")) for c in k_cols]
    num_rules = len(rows)
    gain_curve = np.array(
        [[float(r[c]) for c in k_cols] for r in rows]
    )
    gzip_scores = np.array([float(r["gzip"]) for r in rows])
    labels = [r["quadrant"] for r in rows]

    # Figure 1: gain-vs-k curves colored by quadrant
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ks_arr = np.array(KS)
    for b in range(num_rules):
        color, _ = QUADRANTS[labels[b]]
        ax.plot(ks_arr, gain_curve[b], color=color, alpha=0.35, lw=1)

    by_quad = {name: [i for i, lbl in enumerate(labels) if lbl == name]
               for name in QUADRANTS}
    for name, idx_list in by_quad.items():
        ex = pick_exemplar(idx_list, gain_curve, gzip_scores, name)
        if ex is not None:
            color, _ = QUADRANTS[name]
            ax.plot(ks_arr, gain_curve[ex], color=color, lw=2.5, marker="o",
                    label=f"{name} (rule {ex}, gzip={gzip_scores[ex]:.2f})")

    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("horizon  k  (steps ahead)")
    ax.set_ylabel("probe gain  (1 - loss / marginal-baseline)")
    ax.set_title(f"Probe gain vs horizon for {num_rules} sampled NCA rules\n"
                 f"colored by Wolfram quadrant (gzip * gain @ k={KS[-1]})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "probe_gain_vs_k.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    # Figure 2: quadrant-averaged curves with shaded IQR
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (color, _) in QUADRANTS.items():
        idx = by_quad[name]
        if not idx: continue
        curves = gain_curve[np.array(idx)]
        med = np.median(curves, axis=0)
        q25 = np.quantile(curves, 0.25, axis=0)
        q75 = np.quantile(curves, 0.75, axis=0)
        ax.fill_between(ks_arr, q25, q75, color=color, alpha=0.18)
        ax.plot(ks_arr, med, color=color, lw=2.2, marker="o",
                label=f"{name}  (n={len(idx)})")

    ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("horizon  k")
    ax.set_ylabel("probe gain (median; IQR shaded)")
    ax.set_title("Long-horizon learnability by quadrant")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "probe_gain_vs_k_by_quadrant.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def replot_torch_sweep_scatter():
    tsv_path = os.path.join(OUT_DIR, "probe_preq_continuous_torch_sweep_scores.tsv")
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    mse_floor = np.array([float(r["mse_floor"]) for r in rows])
    preq_gain = np.array([float(r["preq_gain"]) for r in rows])
    initial_loss = np.array([float(r["initial_loss"]) for r in rows])
    num_rules = len(rows)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mse_floor, preq_gain, s=28, alpha=0.7)
    for i in range(num_rules):
        ax.annotate(str(i), (mse_floor[i], preq_gain[i]),
                    fontsize=6, alpha=0.5)
    ax.set_xlabel("MSE floor (final probe loss)")
    ax.set_ylabel("prequential probe gain  (in [0,1])")
    ax.set_title(
        f"Continuous NCA preq sweep  N={num_rules}\n"
        f"gain [{preq_gain.min():.3f}, {preq_gain.max():.3f}]; "
        f"floor [{mse_floor.min():.5f}, {mse_floor.max():.5f}]; "
        f"L0 [{initial_loss.min():.4f}, {initial_loss.max():.4f}]"
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "probe_preq_continuous_torch_sweep_scatter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def replot_probe_curves_snapshots(run_dir):
    best_path = os.path.join(run_dir, "probe_curves_best.npy")
    mean_path = os.path.join(run_dir, "probe_curves_mean.npy")
    if not (os.path.exists(best_path) and os.path.exists(mean_path)):
        return False
    best_curves = np.load(best_path)
    mean_curves = np.load(mean_path)
    n_generations = best_curves.shape[0]

    n_snap = min(6, n_generations)
    snap_gens = np.linspace(0, n_generations - 1, n_snap).astype(int)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, g_idx in enumerate(snap_gens):
        color = cmap(j / max(1, n_snap - 1))
        ax[0].plot(best_curves[g_idx], color=color, label=f"gen {g_idx}")
        ax[1].plot(mean_curves[g_idx], color=color, label=f"gen {g_idx}")
    for a, title in zip(ax, ("best-of-gen probe curve", "population-mean probe curve")):
        a.set_xlabel("probe SGD step"); a.set_ylabel("probe MSE")
        a.set_title(title); a.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(run_dir, "probe_curves_snapshots.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
    return True


def replot_all_snapshots():
    for npy in sorted(glob.glob(os.path.join(OUT_DIR, "*", "probe_curves_best.npy"))):
        replot_probe_curves_snapshots(os.path.dirname(npy))


if __name__ == "__main__":
    replot_probe_gain_vs_k()
    replot_torch_sweep_scatter()
    replot_all_snapshots()
