"""
Render a long consecutive trajectory for a best-ever NCA saved by
scripts/evolve_nca_preq_gzip_continuous_torch.py.

Loads best_ever_params.pt from --run-dir, rolls out N_STEPS with no dropout
(all-ones masks) starting from a fresh uniform IC, and saves a grid PNG
showing every consecutive frame.

Run from repo root:
    .venv/Scripts/python.exe scripts/render_long_trajectory_torch.py \
        --run-dir scripts/demo_out/runs/torch_target/evolve_torch_target095 \
        --n-steps 256 --cols 64
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nca_torch import NCANetworkTorch
from scripts.evolve_nca_preq_continuous_torch import rollout_one_indiv


def render_grid(sims_one_ic: torch.Tensor, cols: int, title: str, path: str) -> None:
    """sims_one_ic: (T, D=3, H, W). Lays frames out in a rows x cols grid."""
    T = sims_one_ic.shape[0]
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(0.45 * cols, 0.5 * rows + 0.4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < T:
            img = sims_one_ic[idx].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img)
            ax.set_title(f"{idx}", fontsize=4, pad=0.5)
        ax.axis("off")
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005,
                        wspace=0.05, hspace=0.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str,
                        help="Directory containing best_ever_params.pt.")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Seed for the fresh IC. Independent of the evolution seed.")
    parser.add_argument("--p-drop", type=float, default=0.0,
                        help="Per-cell Bernoulli drop probability per step (default 0.0 = no dropout).")
    parser.add_argument("--out-name", type=str, default=None,
                        help="Output PNG name (default: long_trajectory_{n_steps}_{regime}.png).")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run_dir, "best_ever_params.pt")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_state = int(ckpt["d_state"])
    grid = int(ckpt["grid"])
    dt = float(ckpt["dt"])
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    best_combined = float(ckpt.get("best_combined", float("nan")))
    gzip_target = float(ckpt.get("gzip_target", float("nan")))

    print(
        f"loaded {ckpt_path}\n"
        f"  d_state={d_state} grid={grid} dt={dt} gzip_target={gzip_target} "
        f"best_combined={best_combined:.4f}"
    )

    net_template = NCANetworkTorch(d_state=d_state).to(device)

    g = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.rand((1, d_state, grid, grid), generator=g, device=device)
    if args.p_drop <= 0.0:
        masks = torch.ones((args.n_steps, 1, 1, grid, grid), device=device)
        regime_str = "no dropout"
        regime_slug = "nodrop"
    else:
        rand_masks = torch.rand(
            (args.n_steps, 1, 1, grid, grid), generator=g, device=device,
        )
        masks = (rand_masks < (1.0 - args.p_drop)).to(torch.float32)
        regime_str = f"p_drop={args.p_drop:.2f}"
        regime_slug = f"pdrop{int(round(args.p_drop * 100)):02d}"

    with torch.no_grad():
        sims = rollout_one_indiv(net_template, params, x0, masks, dt)
    sims_one_ic = sims[0]
    print(f"rolled out {args.n_steps} steps ({regime_str}), sims shape={tuple(sims_one_ic.shape)}")

    out_name = args.out_name or f"long_trajectory_{args.n_steps}_{regime_slug}.png"
    out_path = os.path.join(args.run_dir, out_name)
    title = (
        f"best-ever NCA consecutive rollout ({args.n_steps} steps, {regime_str}) — "
        f"gzip_target={gzip_target}, combined={best_combined:.3f}"
    )
    render_grid(sims_one_ic, args.cols, title, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
