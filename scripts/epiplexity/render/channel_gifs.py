"""
Render per-channel (R/G/B) decomposition GIFs for the gh-pages site.

For each run, loads best_ever_params.pt, reproduces the exact rollout used for
the displayed GIF (same seed / steps / scale / frame duration), and writes
three GIFs that decompose the RGB rollout: channel 0 tinted red (r,0,0),
channel 1 green (0,g,0), channel 2 blue (0,0,b). Frame-for-frame these three
sum back to the combined RGB GIF already on the page.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/render/channel_gifs.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DEMO_OUT, REPO_ROOT, add_repo_root_to_path

add_repo_root_to_path()

from utils.nca_torch import NCANetworkTorch, describe_torch_device, resolve_torch_device
from scripts.epiplexity.trainers.evolve_preq_only import rollout_one_indiv

REPO = str(REPO_ROOT)
DEMO = str(DEMO_OUT)
PAGES = os.path.normpath(os.path.join(REPO, "..", "nca-gh-pages"))

SEED = 12345
SCALE = 16
DURATION_MS = 80
CHANNELS = ("r", "g", "b")

# (checkpoint dir, n_steps, output gif dir, output stem)
JOBS = []


def run_dir_with_legacy(primary, legacy):
    if os.path.isfile(os.path.join(primary, "best_ever_params.pt")):
        return primary
    if os.path.isfile(os.path.join(legacy, "best_ever_params.pt")):
        return legacy
    return primary


_NEW = {
    "linear_ar": "sweep_linear_autoregressive_K16_pdrop0",
    "mlp_small_ar": "sweep_mlp_small_autoregressive_K16_pdrop0",
    "mlp_wide_ar": "sweep_mlp_wide_autoregressive_K16_pdrop0",
    "deep_mlp_ar": "sweep_deep_mlp_autoregressive_K16_pdrop0",
    "linear_direct": "sweep_linear_direct_K16_pdrop0",
    "mlp_small_direct": "sweep_mlp_small_direct_K16_pdrop0",
    "mlp_wide_direct": "sweep_mlp_wide_direct_K16_pdrop0",
    "deep_mlp_direct": "sweep_deep_mlp_direct_K16_pdrop0",
    "transformer_direct": "sweep_transformer_direct_K16_pdrop0",
    "linear_multi": "sweep_linear_multi_pdrop0",
    "mlp_small_multi": "sweep_mlp_small_multi_pdrop0",
    "mlp_wide_multi": "sweep_mlp_wide_multi_pdrop0",
    "deep_mlp_multi": "sweep_deep_mlp_multi_pdrop0",
}
for stem, run in _NEW.items():
    JOBS.append((
        run_dir_with_legacy(
            os.path.join(DEMO, "runs", "sweeps", run),
            os.path.join(DEMO, run),
        ), 1000,
        os.path.join(PAGES, "gifs", "sweep_pdrop0", "channels"), stem,
    ))
_ORIG = {"target03": "03", "target05": "05", "target07": "07",
         "target085": "085", "target095": "095", "target10": "10"}
for stem, t in _ORIG.items():
    JOBS.append((
        os.path.join(DEMO, "runs", "torch_pdrop0", f"evolve_torch_pdrop0_target{t}"), 256,
        os.path.join(PAGES, "gifs", "torch_pdrop0", "channels"), stem,
    ))


def channel_gif(sims_one_ic, ch, scale, path, duration_ms):
    """sims_one_ic: (T,3,H,W). Keep only channel `ch`, zero the rest, save GIF."""
    arr = sims_one_ic.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
    tinted = np.zeros_like(arr)
    tinted[..., ch] = arr[..., ch]
    arr_u8 = (tinted * 255.0 + 0.5).astype(np.uint8)
    frames = []
    for i in range(arr_u8.shape[0]):
        img = Image.fromarray(arr_u8[i], mode="RGB")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, disposal=2, optimize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_torch_device(args.device, allow_cpu=args.allow_cpu)
    print(f"torch device: {describe_torch_device(device)}")
    for run_dir, n_steps, out_dir, stem in JOBS:
        ckpt_path = os.path.join(run_dir, "best_ever_params.pt")
        if not os.path.isfile(ckpt_path):
            print(f"SKIP (no ckpt): {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        d_state = int(ckpt["d_state"])
        grid = int(ckpt["grid"])
        dt = float(ckpt["dt"])
        params = {k: v.to(device) for k, v in ckpt["params"].items()}
        net = NCANetworkTorch(d_state=d_state).to(device)

        g = torch.Generator(device=device).manual_seed(SEED)
        x0 = torch.rand((1, d_state, grid, grid), generator=g, device=device)
        masks = torch.ones((n_steps, 1, 1, grid, grid), device=device)
        with torch.no_grad():
            sims = rollout_one_indiv(net, params, x0, masks, dt)
        sims_one_ic = sims[0]

        os.makedirs(out_dir, exist_ok=True)
        for ch, name in enumerate(CHANNELS):
            out_path = os.path.join(out_dir, f"{stem}_{name}.gif")
            channel_gif(sims_one_ic, ch, SCALE, out_path, DURATION_MS)
        print(f"{stem}: wrote {len(CHANNELS)} channel gifs ({n_steps} steps) -> {out_dir}")


if __name__ == "__main__":
    main()
