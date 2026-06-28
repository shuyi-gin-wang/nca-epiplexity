"""
Render a long consecutive rollout for a best-ever NCA saved by
scripts/epiplexity/trainers/evolve_gzip.py as an animated GIF.

Loads best_ever_params.pt from --run-dir, rolls out N_STEPS, and writes a GIF
with one frame per step (frames nearest-neighbor upscaled for visibility).

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/render/long_trajectory_gif.py \
        --run-dir scripts/demo_out/runs/torch_pdrop0/evolve_torch_pdrop0_target095 \
        --n-steps 1000
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import add_repo_root_to_path

add_repo_root_to_path()

from utils.nca_torch import NCANetworkTorch, describe_torch_device, resolve_torch_device
from scripts.epiplexity.trainers.evolve_preq_only import rollout_one_indiv


def frames_to_gif(sims_one_ic: torch.Tensor, scale: int, path: str,
                  duration_ms: int) -> None:
    """sims_one_ic: (T, D=3, H, W). Writes an animated GIF."""
    arr = sims_one_ic.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    frames = []
    for i in range(arr_u8.shape[0]):
        img = Image.fromarray(arr_u8[i], mode="RGB")
        if scale != 1:
            img = img.resize(
                (img.width * scale, img.height * scale),
                resample=Image.NEAREST,
            )
        frames.append(img)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_one(run_dir: str, n_steps: int, seed: int, p_drop: float,
               scale: int, duration_ms: int, out_name: str | None,
               device: torch.device) -> str:
    ckpt_path = os.path.join(run_dir, "best_ever_params.pt")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_state = int(ckpt["d_state"])
    grid = int(ckpt["grid"])
    dt = float(ckpt["dt"])
    params = {k: v.to(device) for k, v in ckpt["params"].items()}

    net_template = NCANetworkTorch(d_state=d_state).to(device)

    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.rand((1, d_state, grid, grid), generator=g, device=device)
    if p_drop <= 0.0:
        masks = torch.ones((n_steps, 1, 1, grid, grid), device=device)
        regime_slug = "nodrop"
    else:
        rand_masks = torch.rand(
            (n_steps, 1, 1, grid, grid), generator=g, device=device,
        )
        masks = (rand_masks < (1.0 - p_drop)).to(torch.float32)
        regime_slug = f"pdrop{int(round(p_drop * 100)):02d}"

    with torch.no_grad():
        sims = rollout_one_indiv(net_template, params, x0, masks, dt)
    sims_one_ic = sims[0]

    name = out_name or f"long_trajectory_{n_steps}_{regime_slug}.gif"
    out_path = os.path.join(run_dir, name)
    frames_to_gif(sims_one_ic, scale, out_path, duration_ms)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str, nargs="+",
                        help="One or more directories each containing best_ever_params.pt.")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--p-drop", type=float, default=0.0)
    parser.add_argument("--scale", type=int, default=16,
                        help="Integer upscale factor (nearest). 16x16 grid * 16 = 256px.")
    parser.add_argument("--duration-ms", type=int, default=80,
                        help="Per-frame duration in milliseconds.")
    parser.add_argument("--out-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_torch_device(args.device, allow_cpu=args.allow_cpu)
    print(f"torch device: {describe_torch_device(device)}")
    for d in args.run_dir:
        out = render_one(
            d, args.n_steps, args.seed, args.p_drop,
            args.scale, args.duration_ms, args.out_name, device,
        )
        print(f"saved {out}")


if __name__ == "__main__":
    main()
