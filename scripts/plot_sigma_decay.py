"""Visualize the sigma annealing schedule used by the pdrop0 runs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "demo_out"
PDROP0  = OUT_DIR / "runs" / "torch_pdrop0"

RUNS = [
    ("target03",  PDROP0 / "evolve_torch_pdrop0_target03"  / "evolve_nca_preq_gzip_log_torch.tsv"),
    ("target05",  PDROP0 / "evolve_torch_pdrop0_target05"  / "evolve_nca_preq_gzip_log_torch.tsv"),
    ("target07",  PDROP0 / "evolve_torch_pdrop0_target07"  / "evolve_nca_preq_gzip_log_torch.tsv"),
    ("target085", PDROP0 / "evolve_torch_pdrop0_target085" / "evolve_nca_preq_gzip_log_torch.tsv"),
    ("target095", PDROP0 / "evolve_torch_pdrop0_target095" / "evolve_nca_preq_gzip_log_torch.tsv"),
    ("target10",  PDROP0 / "evolve_torch_pdrop0_target10"  / "evolve_nca_preq_gzip_log_torch.tsv"),
]

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    gens = np.array([int(r["gen"]) for r in rows])
    sigma = np.array([float(r["sigma"]) for r in rows])
    preq  = np.array([float(r["best_preq"]) for r in rows])
    return gens, sigma, preq

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ----- panel 1: theoretical sigma vs gen, log scale -----
ax = axes[0]
gens = np.arange(0, 3001)
sigma_init = 0.1
floor = 0.005

decays = {
    "decay=0.995 (600-gen recipe)": 0.995,
    "decay=0.9990 (3000-gen recipe)": (floor / sigma_init) ** (1 / 3000),
}
colors = ["#d62728", "#2ca02c"]
for (label, d), c in zip(decays.items(), colors):
    ax.plot(gens, sigma_init * d**gens, label=label, color=c, lw=2)
ax.axhline(floor, color="grey", ls="--", lw=1, label=f"intended floor sigma={floor}")
for g in (600, 1200, 3000):
    ax.axvline(g, color="black", ls=":", lw=1, alpha=0.4)
ax.set_yscale("log")
ax.set_xlabel("generation")
ax.set_ylabel("sigma (log scale)")
ax.set_title("Theoretical schedules\nsigma_t = 0.1 * decay^t")
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# ----- panel 2: actual sigma from every run -----
ax = axes[1]
cmap = plt.get_cmap("viridis")
for i, (name, path) in enumerate(RUNS):
    g, s, _ = load(path)
    ax.plot(g, s, label=f"{name} (n={len(g)})", color=cmap(i / max(1, len(RUNS) - 1)), lw=1.8)
ax.axhline(floor, color="grey", ls="--", lw=1, label=f"floor sigma={floor}")
ax.set_yscale("log")
ax.set_xlabel("generation")
ax.set_ylabel("sigma (log scale)")
ax.set_title("Actual sigma logged per run")
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# ----- panel 3: best_preq overlaid with sigma for target095 vs target10 -----
ax = axes[2]
ax_r = ax.twinx()
focus = [
    ("target095", PDROP0 / "evolve_torch_pdrop0_target095" / "evolve_nca_preq_gzip_log_torch.tsv", "#d62728"),
    ("target10",  PDROP0 / "evolve_torch_pdrop0_target10"  / "evolve_nca_preq_gzip_log_torch.tsv", "#2ca02c"),
]
for name, path, c in focus:
    g, s, p = load(path)
    ax.plot(g, p, color=c, lw=2, label=f"{name} best_preq")
    ax_r.plot(g, s, color=c, lw=1, ls=":", alpha=0.8, label=f"{name} sigma")
ax.axvline(870, color="#d62728", ls=":", alpha=0.4)
ax.set_xlabel("generation")
ax.set_ylabel("best_preq (solid)")
ax_r.set_ylabel("sigma (dotted, log)")
ax_r.set_yscale("log")
ax.set_title("preq stalls when sigma collapses")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax_r.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Mutation sigma annealing - geometric decay sigma_{t+1} = sigma_t * decay", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = OUT_DIR / "sigma_decay_pdrop0.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")