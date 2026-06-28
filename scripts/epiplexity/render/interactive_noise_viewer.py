"""
Interactive local NCA perturbation viewer.

Loads epiplexity NCA checkpoints, serves a small browser UI, and lets you
inject local noise patches into the live rollout while comparing it against the
same unperturbed reference rollout.

Run from repo root:
    python scripts/epiplexity/render/interactive_noise_viewer.py \
        --run-dir downloaded_runs/epx_transformer_direct_K64_pdrop0_floor035_grid32_r128_pop16_g1500
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import add_repo_root_to_path

add_repo_root_to_path()

from utils.nca_torch import NCANetworkTorch


DEFAULT_RUN_DIR = (
    "downloaded_runs/"
    "epx_transformer_direct_K64_pdrop0_floor035_grid32_r128_pop16_g1500"
)
DEFAULT_MODEL_ROOTS = ("downloaded_runs", "scripts/demo_out")
MODEL_WEIGHTS_PATH = Path(__file__).with_name("model_weights.json")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evolving NCAs with Epiplexity</title>
  <style>
    :root {
      --bg: #f4f2ed;
      --panel: #ffffff;
      --panel-2: #ebe7dd;
      --ink: #191817;
      --muted: #6b665d;
      --line: #d7d0c3;
      --teal: #0b6b66;
      --rust: #bd4f32;
      --gold: #a77512;
      --black: #111111;
      --shadow: 0 18px 42px rgba(31, 27, 20, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, input, select {
      font: inherit;
    }

    .app {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(12px);
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 720;
      letter-spacing: 0;
      line-height: 1.1;
    }


    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      height: 34px;
      padding: 0 12px;
      border-radius: 7px;
      cursor: pointer;
      box-shadow: 0 1px 0 rgba(0, 0, 0, 0.03);
    }

    button:hover {
      border-color: #aaa195;
    }

    button.primary {
      border-color: var(--teal);
      background: var(--teal);
      color: white;
    }

    button.warn {
      border-color: #c56a52;
      color: #8d331f;
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      padding: 18px;
      min-height: 0;
    }

    .stage {
      display: grid;
      grid-template-columns: minmax(300px, 700px) minmax(280px, 420px);
      align-content: start;
      align-items: start;
      justify-content: center;
      gap: 18px;
      min-width: 0;
    }

    .viewport, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .viewport {
      padding: 14px;
      min-width: 0;
    }

    .viewport-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 720;
      letter-spacing: 0;
    }

    .value {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    .canvas-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: var(--black);
      border: 1px solid #222;
      border-radius: 6px;
      overflow: hidden;
    }

    canvas {
      width: 100%;
      height: 100%;
      display: block;
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      cursor: crosshair;
    }

    .crosshair {
      position: absolute;
      width: 1px;
      height: 1px;
      border: 1px solid rgba(255,255,255,0.8);
      outline: 1px solid rgba(0,0,0,0.55);
      pointer-events: none;
      display: none;
    }

    aside {
      min-width: 0;
      padding: 14px;
      overflow: auto;
    }

    .section {
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }

    .section:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .section-title {
      margin: 0 0 10px 0;
      font-size: 12px;
      font-weight: 760;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    details.section > summary.section-title {
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    details.section > summary.section-title::-webkit-details-marker {
      display: none;
    }

    details.section > summary.section-title::after {
      content: "+";
      color: var(--teal);
      font-size: 14px;
      line-height: 1;
    }

    details.section[open] > summary.section-title::after {
      content: "-";
    }

    label {
      display: grid;
      gap: 6px;
      margin: 10px 0;
      color: var(--ink);
      font-size: 12px;
      font-weight: 650;
    }

    label span {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-weight: 560;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--teal);
    }

    select, input[type="number"] {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 0 10px;
      background: white;
      color: var(--ink);
    }

    .seg {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .info-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
      min-width: 0;
    }

    .setup-panel {
      grid-column: 1 / -1;
      background: var(--panel-2);
      border-color: #c9c0b1;
      box-shadow: none;
    }

    .setup-panel .section-title {
      color: var(--teal);
    }

    .selected-panel {
      align-self: start;
      overflow: hidden;
    }

    .selected-panel .spec-list {
      grid-template-columns: 1fr;
    }

    .selected-panel .experiment-title,
    .selected-panel .experiment-summary,
    .selected-panel .spec-list dd {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .selected-panel .spec-list div {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      align-items: baseline;
      gap: 8px;
    }

    .selected-panel .spec-list dt {
      font-size: 11px;
    }

    .selected-panel .spec-list dd {
      margin-top: 0;
    }

    .spec-value-with-help {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
      min-width: 0;
    }

    .horizon-help {
      display: inline-block;
      min-width: 0;
    }

    .horizon-help[open] {
      flex-basis: 100%;
    }

    .horizon-help-summary {
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      min-height: 20px;
      border: 1px solid #c6a86d;
      border-radius: 999px;
      padding: 1px 8px;
      background: #fffdf8;
      color: var(--gold);
      font-size: 10px;
      font-weight: 760;
      line-height: 1.4;
      list-style: none;
      text-transform: uppercase;
    }

    .horizon-help-summary:hover,
    .horizon-help[open] .horizon-help-summary {
      background: #f7f0df;
    }

    .horizon-help-summary::-webkit-details-marker {
      display: none;
    }

    .horizon-help-body {
      margin-top: 6px;
      border: 1px solid #d8ceb9;
      border-radius: 6px;
      padding: 8px;
      background: #fffdf8;
      color: var(--ink);
      font-size: 11px;
      line-height: 1.35;
    }

    .experiment-media {
      display: grid;
      gap: 10px;
      margin-top: 4px;
    }

    .media-block {
      display: grid;
      gap: 6px;
      min-width: 0;
    }

    .media-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .media-title {
      color: var(--muted);
      font-size: 10px;
      font-weight: 720;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .media-zoom {
      height: 26px;
      padding: 0 8px;
      border-radius: 6px;
      color: var(--teal);
      font-size: 11px;
      font-weight: 700;
    }

    .experiment-media img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      cursor: zoom-in;
    }

    .media-dialog {
      position: fixed;
      inset: 0;
      z-index: 80;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(25, 24, 23, 0.68);
    }

    .media-dialog.open {
      display: flex;
    }

    .media-dialog-card {
      width: min(96vw, 1400px);
      max-height: 94vh;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }

    .media-dialog-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }

    .media-dialog-title {
      min-width: 0;
      color: var(--ink);
      font-size: 13px;
      font-weight: 760;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .media-dialog-close {
      width: 30px;
      height: 30px;
      padding: 0;
      flex: 0 0 auto;
      font-size: 16px;
      line-height: 1;
    }

    .media-dialog img {
      display: block;
      width: 100%;
      max-height: calc(94vh - 52px);
      object-fit: contain;
      background: white;
    }

    .media-empty {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .setup-copy {
      margin: 0;
      color: var(--ink);
      font-size: 13px;
      line-height: 1.5;
    }

    .setup-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
      margin-top: 12px;
    }

    .setup-item,
    .experiment-row {
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }

    .setup-item strong,
    .experiment-title {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      font-weight: 760;
      color: var(--ink);
    }

    .setup-item span,
    .experiment-summary {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    .formula {
      display: block;
      margin-top: 8px;
      padding: 10px 12px;
      background: #fffdf8;
      border: 1px solid #cfc5b3;
      border-left: 4px solid var(--teal);
      border-radius: 6px;
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      line-height: 1.7;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }

    .inline-code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      color: var(--ink);
      font-size: 11px;
    }

    .tok-key { color: #065f59; font-weight: 800; }
    .tok-fn { color: #7a2e87; font-weight: 760; }
    .tok-num { color: #a43f20; font-weight: 780; }
    .tok-var { color: #174f8a; font-weight: 760; }
    .tok-op { color: #815d00; font-weight: 800; }
    .tok-comment { color: var(--muted); }

    .experiment-list {
      display: grid;
      gap: 12px;
    }

    .experiment-row {
      display: grid;
      gap: 8px;
    }

    .experiment-row.active .experiment-title {
      color: var(--teal);
    }


    .probe-architecture-list {
      margin: 2px 0 4px;
      border: 1px solid #cfc5b3;
      border-radius: 7px;
      background: #fffdf8;
      overflow: hidden;
    }

    .probe-architecture-summary {
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 34px;
      padding: 8px 10px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0;
      list-style: none;
    }

    .probe-architecture-summary:hover {
      background: #f7f0df;
    }

    .probe-architecture-summary::-webkit-details-marker {
      display: none;
    }

    .probe-architecture-summary::after {
      content: "Show";
      flex: 0 0 auto;
      border: 1px solid #c6a86d;
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--gold);
      font-size: 10px;
      font-weight: 760;
      line-height: 1.4;
      text-transform: uppercase;
    }

    .probe-architecture-list[open] .probe-architecture-summary {
      border-bottom: 1px solid #d8ceb9;
      background: #f7f0df;
    }

    .probe-architecture-list[open] .probe-architecture-summary::after {
      content: "Hide";
    }

    .probe-architecture-body {
      display: grid;
      gap: 8px;
      padding: 10px;
    }

    .probe-architecture {
      min-width: 0;
    }

    .probe-architecture-title {
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 720;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .probe-formula {
      margin-top: 0;
      border-left-color: var(--gold);
    }

    .spec-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 10px;
      margin: 0;
    }

    .spec-list div {
      min-width: 0;
    }

    .spec-list dt {
      color: var(--muted);
      font-size: 10px;
      font-weight: 720;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .spec-list dd {
      margin: 2px 0 0;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }

    .status {
      min-height: 20px;
      margin-top: 10px;
      color: var(--rust);
      font-size: 12px;
      line-height: 1.35;
    }

    .accent {
      color: var(--gold);
    }

    @media (max-width: 1040px) {
      main {
        grid-template-columns: 1fr;
      }

      aside {
        order: -1;
      }
    }

    @media (max-width: 720px) {
      header {
        align-items: stretch;
        flex-direction: column;
      }

      .toolbar {
        justify-content: flex-start;
      }

      .stage {
        grid-template-columns: 1fr;
      }

      .setup-panel {
        grid-column: auto;
      }

      .setup-grid,
      .spec-list {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Evolving NCAs with Epiplexity</h1>
      <div class="toolbar">
        <button id="runBtn" class="primary" title="Run or pause the rollout">Run</button>
        <button id="stepBtn" title="Advance a fixed number of NCA steps">Step</button>
        <button id="resetBtn" class="warn" title="Reset to a fresh initial condition">Reset</button>
      </div>
    </header>

    <main>
      <section class="stage">
        <div class="info-panel setup-panel">
          <p class="section-title">Experiment Setup</p>
          <p class="setup-copy">
            NCAs are evolved by an elitist strategy with candidates rolled out from random initial states.
            Runs are selected for dynamics that stay non-collapsed while remaining hard for probe models to predict.
          </p>
          <div class="setup-grid">
            <div class="setup-item">
              <strong>NCA rule</strong>
              <span>State is 3 channels on an <span class="inline-code">H x W</span> torus.</span>
              <code class="formula"><span class="tok-fn">circular_pad_1</span>
<span class="tok-op">-></span> <span class="tok-key">Conv3x3</span>   <span class="tok-num">3</span> <span class="tok-op">-></span> <span class="tok-num">4</span>
<span class="tok-op">-></span> <span class="tok-key">Conv1x1</span>   <span class="tok-num">4</span> <span class="tok-op">-></span> <span class="tok-num">16</span>
<span class="tok-op">-></span> <span class="tok-key">ReLU</span>
<span class="tok-op">-></span> <span class="tok-key">Conv1x1</span>  <span class="tok-num">16</span> <span class="tok-op">-></span> <span class="tok-num">3</span>

<span class="tok-var">x[t+1]</span> <span class="tok-op">=</span> <span class="tok-fn">clamp</span>(<span class="tok-var">x[t]</span> <span class="tok-op">+</span> <span class="tok-num">0.05</span> <span class="tok-op">*</span> <span class="tok-fn">f</span>(<span class="tok-var">x[t]</span>), <span class="tok-num">0</span>, <span class="tok-num">1</span>)</code>
            </div>
            <div class="setup-item">
              <strong>Fitness</strong>
              <code class="formula"><span class="tok-var">fitness</span> <span class="tok-op">=</span> <span class="tok-var">preq</span> <span class="tok-op">*</span> <span class="tok-num">1</span>[<span class="tok-var">gzip_ratio</span> <span class="tok-op">&gt;</span> <span class="tok-var">threshold</span>]

<span class="tok-var">preq</span> <span class="tok-op">=</span> <span class="tok-fn">sum_t</span> <span class="tok-fn">max</span>(<span class="tok-var">loss[t]</span> <span class="tok-op">-</span> <span class="tok-var">loss_final</span>, <span class="tok-num">0</span>)
<span class="tok-var">gzip_ratio</span> <span class="tok-op">=</span> <span class="tok-fn">gzip_bytes</span>(<span class="tok-var">quantized_full_rollout</span>) <span class="tok-op">/</span> <span class="tok-fn">raw_bytes</span>(<span class="tok-var">quantized_full_rollout</span>)</code>
              <span>Fitness combines a gzip complexity floor with prequential coding as the epiplexity heuristic. <span class="inline-code">preq</span> estimates information content as the area under the probe model's loss curve above its final loss over 50 Adam probe steps; <span class="inline-code">gzip_ratio</span> is computed across the whole rollout tensor and gates out collapsed dynamics.</span>
            </div>

          </div>
        </div>

        <div class="viewport">
          <div class="viewport-head">
            <h2>Live State</h2>
            <div id="stepReadout" class="value">step 0 | seed --</div>
          </div>
          <div class="canvas-wrap">
            <canvas id="stateCanvas" width="32" height="32"></canvas>
            <div id="patchBox" class="crosshair"></div>
          </div>
        </div>

        <div class="info-panel selected-panel">
          <p class="section-title">Selected Experiment</p>
          <div id="experimentList" class="experiment-list"></div>
        </div>

      </section>

      <aside>
        <div class="section">
          <p class="section-title">NCA</p>
          <label>
            <span><span>Checkpoint</span><output id="modelCount">0</output></span>
            <select id="modelSelect"></select>
          </label>
          <label>
            <span><span>Viewing grid</span><output id="viewGridOut">trained</output></span>
            <select id="viewGrid"></select>
          </label>
        </div>

        <div class="section">
          <p class="section-title">Perturbation</p>
          <label>
            <span><span>Patch size</span><output id="patchSizeOut">5</output></span>
            <input id="patchSize" type="range" min="1" max="21" step="2" value="5" />
          </label>
          <label>
            <span><span>Strength</span><output id="strengthOut">1.00</output></span>
            <input id="strength" type="range" min="0" max="1" step="0.01" value="1" />
          </label>
          <label>
            <span><span>Mode</span></span>
            <select id="noiseMode">
              <option value="uniform">Uniform noise</option>
              <option value="zero">Zero patch</option>
              <option value="one">One patch</option>
              <option value="invert">Invert patch</option>
            </select>
          </label>
        </div>

        <details class="section">
          <summary class="section-title">Rollout</summary>
          <label>
            <span><span>Steps per tick</span><output id="stepsOut">1</output></span>
            <input id="stepsPerTick" type="range" min="1" max="32" step="1" value="1" />
          </label>
          <label>
            <span><span>Tick interval</span><output id="intervalOut">80 ms</output></span>
            <input id="tickMs" type="range" min="20" max="500" step="10" value="80" />
          </label>

        </details>
        <div id='status' class='status'></div>
      </aside>
    </main>
  </div>

  <div id="mediaDialog" class="media-dialog" role="dialog" aria-modal="true" aria-hidden="true">
    <div class="media-dialog-card">
      <div class="media-dialog-head">
        <div id="mediaDialogTitle" class="media-dialog-title"></div>
        <button id="mediaDialogClose" class="media-dialog-close" type="button" title="Close enlarged plot">x</button>
      </div>
      <img id="mediaDialogImg" alt="" />
    </div>
  </div>

  <script>
    const stateCanvas = document.getElementById('stateCanvas');
    const stateCtx = stateCanvas.getContext('2d');
    stateCtx.imageSmoothingEnabled = false;

    const el = (id) => document.getElementById(id);
    const controls = {
      modelSelect: el('modelSelect'),
      viewGrid: el('viewGrid'),
      patchSize: el('patchSize'),
      strength: el('strength'),
      noiseMode: el('noiseMode'),
      stepsPerTick: el('stepsPerTick'),
      tickMs: el('tickMs'),
    };

    let running = false;
    let timer = null;
    let busy = false;
    let switchingModel = false;
    let requestEpoch = 0;
    let grid = 32;
    let trainedGrid = 32;
    let lastPoint = null;
    let modelOptions = [];
    let renderedExperimentId = null;
    const probeArchitectureOpen = new Map();

    function setStatus(msg) {
      el('status').textContent = msg || '';
    }

    function delay(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function fmt(v, digits = 4) {
      return Number(v).toFixed(digits);
    }

    function syncOutputs() {
      el('patchSizeOut').value = controls.patchSize.value;
      el('strengthOut').value = Number(controls.strength.value).toFixed(2);
      el('stepsOut').value = controls.stepsPerTick.value;
      el('intervalOut').value = `${controls.tickMs.value} ms`;
      el('viewGridOut').value = controls.viewGrid.value ? `${controls.viewGrid.value}x${controls.viewGrid.value}` : 'trained';
    }

    function viewGridChoices(baseGrid) {
      const base = Math.max(1, Number(baseGrid) || 32);
      const choices = [];
      for (let value = base; value <= 128; value *= 2) {
        choices.push(value);
        if (value === 128) break;
      }
      return choices;
    }

    function syncViewGridControl(payload) {
      const nextTrained = Number(payload.trained_grid || payload.grid || trainedGrid);
      const nextGrid = Number(payload.grid || nextTrained);
      const existing = Array.from(controls.viewGrid.options).map((option) => Number(option.value));
      const choices = viewGridChoices(nextTrained);
      const needsRebuild = trainedGrid !== nextTrained || choices.length !== existing.length || choices.some((value, idx) => value !== existing[idx]);
      trainedGrid = nextTrained;
      if (needsRebuild) {
        controls.viewGrid.replaceChildren();
        for (const value of choices) {
          const option = document.createElement('option');
          option.value = String(value);
          option.textContent = value === nextTrained ? `${value} x ${value} (trained)` : `${value} x ${value}`;
          controls.viewGrid.appendChild(option);
        }
      }
      controls.viewGrid.value = String(nextGrid);
    }

    async function postJSON(path, body = {}) {
      const response = await fetch(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    async function getJSON(path) {
      const response = await fetch(path);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    function drawDataUrl(canvas, ctx, dataUrl) {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0);
      };
      img.src = dataUrl;
    }

    function openMediaDialog(title, url) {
      const dialog = el('mediaDialog');
      const image = el('mediaDialogImg');
      el('mediaDialogTitle').textContent = title;
      image.src = url;
      image.alt = title;
      dialog.classList.add('open');
      dialog.setAttribute('aria-hidden', 'false');
      el('mediaDialogClose').focus();
    }

    function closeMediaDialog() {
      const dialog = el('mediaDialog');
      dialog.classList.remove('open');
      dialog.setAttribute('aria-hidden', 'true');
      el('mediaDialogImg').removeAttribute('src');
    }

    function updateView(payload) {
      grid = payload.grid || grid;
      syncViewGridControl(payload);
      drawDataUrl(stateCanvas, stateCtx, payload.state_image);
      const seedText = payload.seed === undefined ? '' : ` | seed ${payload.seed}`;
      el('stepReadout').textContent = `step ${payload.step}${seedText}`;

      if (payload.active_model_id && controls.modelSelect.value !== payload.active_model_id) {
        controls.modelSelect.value = payload.active_model_id;
      }
      renderExperiments(payload.active_model_id || controls.modelSelect.value);
      syncOutputs();
    }

    function rememberProbeArchitectureState(list) {
      const details = list ? list.querySelector('.probe-architecture-list') : null;
      if (renderedExperimentId && details) {
        probeArchitectureOpen.set(renderedExperimentId, details.open);
      }
    }

    function renderExperiments(activeId = controls.modelSelect.value, force = false) {
      const list = el('experimentList');
      if (!list) return;
      if (!force && renderedExperimentId === activeId) return;
      rememberProbeArchitectureState(list);
      list.replaceChildren();
      const model = modelOptions.find((item) => item.id === activeId);
      if (!model) {
        renderedExperimentId = null;
        const empty = document.createElement('div');
        empty.className = 'experiment-summary';
        empty.textContent = 'Select an NCA checkpoint to see its experiment setup.';
        list.appendChild(empty);
        return;
      }
      renderedExperimentId = activeId;

      const row = document.createElement('div');
      row.className = 'experiment-row active';

      const title = document.createElement('div');
      title.className = 'experiment-title';
      title.textContent = model.label;
      row.appendChild(title);

      if (model.probe_architectures && model.probe_architectures.length) {
        const archDetails = document.createElement('details');
        archDetails.className = 'probe-architecture-list';
        archDetails.open = probeArchitectureOpen.get(activeId) === true;
        archDetails.addEventListener('toggle', () => {
          probeArchitectureOpen.set(activeId, archDetails.open);
        });
        const archSummary = document.createElement('summary');
        archSummary.className = 'probe-architecture-summary';
        archSummary.textContent = model.probe_architectures.length === 1 ? 'Probe architecture' : 'Probe architectures';
        const archBody = document.createElement('div');
        archBody.className = 'probe-architecture-body';
        for (const arch of model.probe_architectures) {
          const block = document.createElement('div');
          block.className = 'probe-architecture';
          const archTitle = document.createElement('div');
          archTitle.className = 'probe-architecture-title';
          archTitle.textContent = arch.label || 'Probe architecture';
          const code = document.createElement('code');
          code.className = 'formula probe-formula';
          code.textContent = arch.body || '';
          block.appendChild(archTitle);
          block.appendChild(code);
          archBody.appendChild(block);
        }
        archDetails.appendChild(archSummary);
        archDetails.appendChild(archBody);
        row.appendChild(archDetails);
      }

      const specs = document.createElement('dl');
      specs.className = 'spec-list';
      for (const item of model.details || []) {
        const wrap = document.createElement('div');
        const term = document.createElement('dt');
        const desc = document.createElement('dd');
        term.textContent = item.label;
        if (item.label === 'Horizon') {
          const valueWrap = document.createElement('div');
          valueWrap.className = 'spec-value-with-help';
          const value = document.createElement('span');
          value.textContent = item.value;
          const help = document.createElement('details');
          help.className = 'horizon-help';
          const summary = document.createElement('summary');
          summary.className = 'horizon-help-summary';
          summary.textContent = 'K horizon';
          const body = document.createElement('div');
          body.className = 'horizon-help-body';
          body.textContent = 'K is the prediction horizon: direct K predicts x[t+K] from x[t] in one probe call, while autoregressive K applies the probe K one-step calls.';
          help.appendChild(summary);
          help.appendChild(body);
          valueWrap.appendChild(value);
          valueWrap.appendChild(help);
          desc.appendChild(valueWrap);
        } else {
          desc.textContent = item.value;
        }
        wrap.appendChild(term);
        wrap.appendChild(desc);
        specs.appendChild(wrap);
      }
      row.appendChild(specs);


      const media = document.createElement('div');
      media.className = 'experiment-media';
      const addMedia = (titleText, url) => {
        if (!url) return;
        const block = document.createElement('div');
        block.className = 'media-block';
        const head = document.createElement('div');
        head.className = 'media-head';
        const mediaTitle = document.createElement('div');
        mediaTitle.className = 'media-title';
        mediaTitle.textContent = titleText;
        const zoom = document.createElement('button');
        zoom.type = 'button';
        zoom.className = 'media-zoom';
        zoom.textContent = 'Enlarge';
        const dialogTitle = `${model.label} - ${titleText}`;
        zoom.addEventListener('click', () => openMediaDialog(dialogTitle, url));
        head.appendChild(mediaTitle);
        head.appendChild(zoom);
        const img = document.createElement('img');
        img.src = url;
        img.alt = `${model.label} ${titleText}`;
        img.loading = 'lazy';
        img.tabIndex = 0;
        img.setAttribute('role', 'button');
        img.title = 'Open larger plot';
        img.addEventListener('click', () => openMediaDialog(dialogTitle, url));
        img.addEventListener('keydown', (evt) => {
          if (evt.key === 'Enter' || evt.key === ' ') {
            evt.preventDefault();
            openMediaDialog(dialogTitle, url);
          }
        });
        block.appendChild(head);
        block.appendChild(img);
        media.appendChild(block);
      };
      addMedia('Training curve', model.training_curve_url);
      addMedia('Probe loss curves', model.probe_curve_url);
      if (!media.children.length) {
        const empty = document.createElement('div');
        empty.className = 'media-empty';
        empty.textContent = 'Curve images are not available for this checkpoint.';
        media.appendChild(empty);
      }
      row.appendChild(media);
      list.appendChild(row);
    }

    async function loadModels() {
      try {
        const payload = await getJSON('/api/models');
        modelOptions = payload.models || [];
        controls.modelSelect.replaceChildren();
        for (const model of modelOptions) {
          const option = document.createElement('option');
          option.value = model.id;
          option.textContent = model.label;
          option.title = model.source;
          controls.modelSelect.appendChild(option);
        }
        el('modelCount').value = `${modelOptions.length}`;
        if (payload.active_id) {
          controls.modelSelect.value = payload.active_id;
        }
        renderExperiments(payload.active_id || controls.modelSelect.value, true);
      } catch (err) {
        setStatus(err.message);
      }
    }

    async function refresh() {
      try {
        updateView(await getJSON('/api/state'));
        setStatus('');
      } catch (err) {
        setStatus(err.message);
      }
    }

    async function stepOnce() {
      if (busy || switchingModel) return;
      const epoch = requestEpoch;
      busy = true;
      try {
        const payload = await postJSON('/api/step', {
          steps: Number(controls.stepsPerTick.value)
        });
        if (epoch !== requestEpoch || switchingModel) return;
        updateView(payload);
        setStatus('');
      } catch (err) {
        if (epoch === requestEpoch && !switchingModel) setStatus(err.message);
      } finally {
        busy = false;
      }
    }

    function scheduleRun() {
      if (timer) clearInterval(timer);
      if (!running) return;
      timer = setInterval(stepOnce, Number(controls.tickMs.value));
    }

    async function injectAt(x, y) {
      try {
        const payload = await postJSON('/api/inject', {
          x, y,
          size: Number(controls.patchSize.value),
          strength: Number(controls.strength.value),
          mode: controls.noiseMode.value
        });
        updateView(payload);
        setStatus('');
      } catch (err) {
        setStatus(err.message);
      }
    }

    function canvasCell(evt) {
      const rect = stateCanvas.getBoundingClientRect();
      const x = Math.floor((evt.clientX - rect.left) / rect.width * grid);
      const y = Math.floor((evt.clientY - rect.top) / rect.height * grid);
      return {
        x: Math.max(0, Math.min(grid - 1, x)),
        y: Math.max(0, Math.min(grid - 1, y))
      };
    }

    function drawPatchBox(point) {
      const box = el('patchBox');
      const rect = stateCanvas.getBoundingClientRect();
      const size = Number(controls.patchSize.value);
      const cell = rect.width / grid;
      box.style.display = 'block';
      box.style.width = `${Math.max(1, size * cell)}px`;
      box.style.height = `${Math.max(1, size * cell)}px`;
      box.style.left = `${(point.x + 0.5 - size / 2) * cell}px`;
      box.style.top = `${(point.y + 0.5 - size / 2) * cell}px`;
    }

    stateCanvas.addEventListener('pointerdown', async (evt) => {
      stateCanvas.setPointerCapture(evt.pointerId);
      lastPoint = canvasCell(evt);
      drawPatchBox(lastPoint);
      await injectAt(lastPoint.x, lastPoint.y);
    });

    stateCanvas.addEventListener('pointermove', async (evt) => {
      const point = canvasCell(evt);
      drawPatchBox(point);
      if (evt.buttons !== 1) return;
      if (!lastPoint || point.x !== lastPoint.x || point.y !== lastPoint.y) {
        lastPoint = point;
        await injectAt(point.x, point.y);
      }
    });

    stateCanvas.addEventListener('pointerleave', () => {
      el('patchBox').style.display = 'none';
    });

    stateCanvas.addEventListener('pointerup', () => {
      lastPoint = null;
    });

    function setRunning(next) {
      running = next;
      el('runBtn').textContent = running ? 'Pause' : 'Run';
      el('runBtn').classList.toggle('primary', !running);
      scheduleRun();
    }

    el('runBtn').addEventListener('click', () => setRunning(!running));

    controls.modelSelect.addEventListener('change', async () => {
      const targetId = controls.modelSelect.value;
      const switchEpoch = requestEpoch + 1;
      requestEpoch = switchEpoch;
      switchingModel = true;
      setRunning(false);
      let loaded = false;
      try {
        while (busy && switchEpoch === requestEpoch) {
          await delay(10);
        }
        if (switchEpoch !== requestEpoch) return;
        const payload = await postJSON('/api/load', {
          id: targetId
        });
        if (switchEpoch !== requestEpoch) return;
        updateView(payload);
        await loadModels();
        setStatus('');
        loaded = true;
      } catch (err) {
        if (switchEpoch === requestEpoch) setStatus(err.message);
      } finally {
        if (switchEpoch === requestEpoch) {
          switchingModel = false;
          if (loaded) setRunning(true);
        }
      }
    });

    controls.viewGrid.addEventListener('change', async () => {
      const targetGrid = Number(controls.viewGrid.value);
      const switchEpoch = requestEpoch + 1;
      const wasRunning = running;
      requestEpoch = switchEpoch;
      switchingModel = true;
      setRunning(false);
      let loaded = false;
      try {
        while (busy && switchEpoch === requestEpoch) {
          await delay(10);
        }
        if (switchEpoch !== requestEpoch) return;
        const payload = await postJSON('/api/view-grid', {
          view_grid: targetGrid
        });
        if (switchEpoch !== requestEpoch) return;
        updateView(payload);
        setStatus('');
        loaded = true;
      } catch (err) {
        if (switchEpoch === requestEpoch) setStatus(err.message);
      } finally {
        if (switchEpoch === requestEpoch) {
          switchingModel = false;
          if (loaded && wasRunning) setRunning(true);
        }
      }
    });

    el('stepBtn').addEventListener('click', stepOnce);
    el('resetBtn').addEventListener('click', async () => {
      try {
        const payload = await postJSON('/api/reset', {
        });
        updateView(payload);
        setStatus(payload.seed === undefined ? '' : `Reset seed ${payload.seed}`);
      } catch (err) {
        setStatus(err.message);
      }
    });

    controls.tickMs.addEventListener('input', () => {
      syncOutputs();
      scheduleRun();
    });

    for (const input of Object.values(controls)) {
      input.addEventListener('input', syncOutputs);
    }

    el('mediaDialogClose').addEventListener('click', closeMediaDialog);
    el('mediaDialog').addEventListener('click', (evt) => {
      if (evt.target === el('mediaDialog')) closeMediaDialog();
    });
    window.addEventListener('keydown', (evt) => {
      if (evt.key === 'Escape' && el('mediaDialog').classList.contains('open')) {
        closeMediaDialog();
      }
    });

    syncOutputs();
    loadModels().then(refresh).then(() => setRunning(true));
  </script>
</body>
</html>
"""


def _pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is not available")
    return device


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _png_data_url(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=False)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not torch.is_tensor(v)}
    if torch.is_tensor(value):
        return {"tensor_shape": list(value.shape)}
    return str(value)


def _checkpoint_params(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "params" in ckpt:
        return ckpt["params"]
    if "best_ever_params" in ckpt:
        return ckpt["best_ever_params"]
    if "best_in_gen_params" in ckpt:
        return ckpt["best_in_gen_params"]
    raise KeyError("checkpoint does not contain params")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _model_label(path: Path) -> str:
    if path.name == "best_ever_params.pt":
        return f"{path.parent.name} / best"
    if path.parent.name == "checkpoints":
        return f"{path.parent.parent.name} / {path.stem}"
    return _display_path(path)


def _model_entry(idx: int, path: Path, label: str | None = None) -> dict[str, str]:
    path = path.resolve()
    return {
        "id": f"m{idx}",
        "label": label or _model_label(path),
        "source": _display_path(path),
        "path": str(path),
    }


def _fmt_float(value: Any, digits: int = 3) -> str | None:
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return None


def _infer_k(text: str) -> str | None:
    match = re.search(r"(?:^|[_-])K(\d+)(?:\D|$)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"directK(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else None


def _infer_floor(text: str) -> str | None:
    match = re.search(r"floor(\d+)", text, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1)
    if raw.startswith("0") and len(raw) > 1:
        return _fmt_float(float("0." + raw[1:]), 2)
    return _fmt_float(float(raw), 2)


def _infer_probe(label: str, source: str, meta: dict[str, Any]) -> str | None:
    probe_archs = meta.get("probe_archs")
    if isinstance(probe_archs, list) and probe_archs:
        return ", ".join(str(item) for item in probe_archs)
    if meta.get("probe_arch"):
        return str(meta["probe_arch"])
    text = f"{label} {source}".lower()
    if "multi_observer" in text or "observer_balanced" in text:
        return "linear, mlp_wide, deep_mlp, transformer"
    for probe in ("transformer", "deep_mlp", "deeper_mlp", "mlp_wide", "mlp_small", "linear"):
        if probe in text:
            return "deep_mlp" if probe == "deeper_mlp" else probe
    return None


_PROBE_DEFAULT_HIDDEN = {
    "linear": 0,
    "mlp_small": 8,
    "mlp_wide": 32,
    "deep_mlp": 16,
    "transformer": 32,
}


def _probe_arch_names(probe: str | None) -> list[str]:
    if not probe:
        return []
    names: list[str] = []
    for raw in str(probe).split(","):
        name = raw.strip()
        if name == "deeper_mlp":
            name = "deep_mlp"
        if name in _PROBE_DEFAULT_HIDDEN and name not in names:
            names.append(name)
    return names


def _meta_int(meta: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(meta.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _probe_hidden(arch: str, meta: dict[str, Any]) -> int:
    default = _PROBE_DEFAULT_HIDDEN[arch]
    hidden = _meta_int(meta, "probe_hidden", 0)
    if arch != "linear" and hidden > 0:
        return hidden
    return default


def _probe_architecture_entry(arch: str, meta: dict[str, Any]) -> dict[str, str] | None:
    d_state = _meta_int(meta, "d_state", 3)
    hidden = _probe_hidden(arch, meta)
    if arch == "linear":
        return {
            "label": "linear probe",
            "body": (
                "circular_pad_1\n"
                f"-> Conv3x3   {d_state} -> {d_state}\n\n"
                "y_hat = f(x)"
            ),
        }
    if arch in {"mlp_small", "mlp_wide"}:
        return {
            "label": f"{arch} probe (H={hidden})",
            "body": (
                "residual local MLP\n"
                "circular_pad_1\n"
                f"-> Conv3x3   {d_state} -> {hidden}\n"
                "-> ReLU\n"
                f"-> Conv1x1  {hidden} -> {d_state}\n\n"
                "y_hat = x + d(x)"
            ),
        }
    if arch == "deep_mlp":
        return {
            "label": f"deep_mlp probe (H={hidden})",
            "body": (
                "residual deep local MLP\n"
                "circular_pad_1\n"
                f"-> Conv3x3   {d_state} -> {hidden}\n"
                "-> ReLU\n"
                "circular_pad_1\n"
                f"-> Conv3x3  {hidden} -> {hidden}\n"
                "-> ReLU\n"
                f"-> Conv1x1  {hidden} -> {d_state}\n\n"
                "y_hat = x + d(x)"
            ),
        }
    if arch == "transformer":
        ffn_hidden = hidden * 4
        return {
            "label": f"transformer probe (H={hidden})",
            "body": (
                "grid transformer probe\n"
                f"-> Conv1x1   {d_state} -> {hidden} + position\n"
                "-> LayerNorm\n"
                "-> 2-head self-attention\n"
                "-> LayerNorm\n"
                f"-> FFN       {hidden} -> {ffn_hidden} -> {hidden}\n"
                "-> LayerNorm\n"
                f"-> Conv1x1  {hidden} -> {d_state}\n\n"
                "y_hat = x + d(x)"
            ),
        }
    return None


def _probe_architecture_entries(probe: str | None, meta: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for arch in _probe_arch_names(probe):
        entry = _probe_architecture_entry(arch, meta)
        if entry is not None:
            entries.append(entry)
    return entries


def _infer_horizon(label: str, source: str, meta: dict[str, Any]) -> str | None:
    text = f"{label} {source}"
    mode = str(meta.get("horizon_mode") or "").strip()
    k = meta.get("probe_horizon") or _infer_k(text)
    if not mode:
        low = text.lower()
        if "autoregressive" in low:
            mode = "autoregressive"
        elif "direct" in low:
            mode = "direct"
        elif "multi" in low:
            mode = "multi"
    if mode == "multi" and meta.get("multi_ks"):
        return f"multi K={k}; ks={', '.join(str(v) for v in meta['multi_ks'])}" if k else f"multi ks={', '.join(str(v) for v in meta['multi_ks'])}"
    if mode and k:
        return f"{mode} K={k}"
    if k:
        return f"K={k}"
    return mode or None


def _model_experiment_info(entry: dict[str, str]) -> dict[str, Any]:
    path = Path(entry["path"])
    label = entry["label"]
    source = entry["source"]
    meta: dict[str, Any] = {}
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        meta = {
            str(k): _jsonable(v)
            for k, v in ckpt.items()
            if k not in {"params", "best_ever_params", "best_in_gen_params"}
        }
    except Exception as exc:
        meta = {"metadata_error": str(exc)}

    text = f"{label} {source}"
    probe = _infer_probe(label, source, meta)
    probe_architectures = _probe_architecture_entries(probe, meta)
    horizon = _infer_horizon(label, source, meta)
    gzip_floor = meta.get("gzip_threshold")
    if gzip_floor is None:
        gzip_floor = _infer_floor(text)
    pop_size = meta.get("pop_size")
    if pop_size is None:
        match = re.search(r"pop(\d+)", text, re.IGNORECASE)
        pop_size = match.group(1) if match else None

    details: list[dict[str, str]] = []

    def add(key: str, value: Any) -> None:
        if value is None or value == "":
            return
        if isinstance(value, float):
            value = _fmt_float(value)
        details.append({"label": key, "value": str(value)})

    add("Probe", probe)
    add("Horizon", horizon)
    if meta.get("grid"):
        add("Grid", f"{meta['grid']}x{meta['grid']}")
    if meta.get("rollout_steps"):
        add("Rollout", f"{meta['rollout_steps']} steps")

    if gzip_floor is not None:
        add("Gzip", f"threshold > {gzip_floor}")
    add("Population", pop_size)
    add("Generations", meta.get("n_generations"))
    add("Best score", meta.get("best_combined"))

    parts = []
    if probe:
        parts.append(f"{probe} probe")
    if horizon:
        parts.append(horizon)
    if meta.get("grid"):
        parts.append(f"{meta['grid']}x{meta['grid']} grid")
    if meta.get("rollout_steps"):
        parts.append(f"{meta['rollout_steps']}-step rollout")
    summary = "; ".join(parts) + "." if parts else source
    result: dict[str, Any] = {"summary": summary, "details": details}
    if probe_architectures:
        result["probe_architectures"] = probe_architectures
    return result


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path.resolve()
    return None


def _checkpoint_media_dirs(entry: dict[str, str]) -> list[Path]:
    checkpoint_path = Path(entry["path"])
    parent = checkpoint_path.parent
    dirs: list[Path] = [parent]
    if parent.name.startswith("preview_") or parent.name == "checkpoints":
        dirs.append(parent.parent)
    return dirs


def _gh_pages_media_stem(entry: dict[str, str]) -> str | None:
    text = f"{entry.get('label', '')} {entry.get('source', '')}".lower()
    if "mlp_small_direct_k8" in text or "sweep_mlp_small_direct_k8" in text:
        return None
    mappings = [
        (("deeper_mlp_autoregressive", "deep_mlp_autoregressive", "sweep_deep_mlp_autoregressive"), "deep_mlp_ar"),
        (("deeper_mlp_direct", "deep_mlp_direct", "sweep_deep_mlp_direct"), "deep_mlp_direct"),
        (("mlp_small_autoregressive", "sweep_mlp_small_autoregressive"), "mlp_small_ar"),
        (("mlp_small_direct", "sweep_mlp_small_direct"), "mlp_small_direct"),
        (("mlp_wide_autoregressive", "sweep_mlp_wide_autoregressive"), "mlp_wide_ar"),
        (("mlp_wide_direct", "sweep_mlp_wide_direct"), "mlp_wide_direct"),
        (("linear_autoregressive", "sweep_linear_autoregressive"), "linear_ar"),
        (("linear_direct", "sweep_linear_direct"), "linear_direct"),
        (("transformer_direct", "sweep_transformer_direct"), "transformer_direct"),
    ]
    for needles, stem in mappings:
        if any(needle in text for needle in needles):
            return stem
    return None


def _model_media_assets(entry: dict[str, str]) -> dict[str, Path]:
    media_dirs = _checkpoint_media_dirs(entry)
    training = _first_existing([
        folder / filename
        for folder in media_dirs
        for filename in (
            "evolve_nca_preq_gzip_curve_torch.png",
            "evolve_nca_preq_curve_torch.png",
        )
    ])
    probe = _first_existing([
        folder / "probe_curves_snapshots.png"
        for folder in media_dirs
    ])

    stem = _gh_pages_media_stem(entry)
    if stem:
        base = Path(".gh-pages-worktree") / "img" / "sweep_pdrop0"
        if training is None:
            training = _first_existing([base / f"{stem}_curve.png"])
        if probe is None:
            probe = _first_existing([base / f"{stem}_probe.png"])

    assets: dict[str, Path] = {}
    if training is not None:
        assets["training"] = training
    if probe is not None:
        assets["probe"] = probe
    return assets


def _model_media_png(path: Path, kind: str) -> bytes:
    with Image.open(path) as img:
        img.load()
        if kind == "training":
            width, height = img.size
            crop_width = max(1, int(width * 2 / 3))
            img = img.crop((0, 0, crop_width, height))
        if img.mode not in {"RGB", "RGBA"}:
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()

def _public_model_entry(entry: dict[str, str]) -> dict[str, Any]:
    public: dict[str, Any] = {
        "id": entry["id"],
        "label": entry["label"],
        "source": entry["source"],
    }
    public.update(_model_experiment_info(entry))
    media_assets = _model_media_assets(entry)
    if "training" in media_assets:
        public["training_curve_url"] = f"/api/model-media?id={entry['id']}&kind=training"
    if "probe" in media_assets:
        public["probe_curve_url"] = f"/api/model-media?id={entry['id']}&kind=probe"
    return public


def _resolve_model_path(path_text: str, config_path: Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists() or config_path is None:
        return cwd_path
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return cwd_path


def _dedupe_paths(paths: list[tuple[Path, str | None]]) -> list[tuple[Path, str | None]]:
    seen: set[Path] = set()
    unique: list[tuple[Path, str | None]] = []
    for path, label in paths:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        unique.append((path, label))
    return unique


def _read_curated_model_entries(config_path: Path, initial_checkpoint: Path) -> list[dict[str, str]]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_models = data.get("models", []) if isinstance(data, dict) else data
    if not isinstance(raw_models, list):
        raise ValueError(f"model list must contain a JSON list: {config_path}")

    candidates: list[tuple[Path, str | None]] = []
    for item in raw_models:
        if isinstance(item, str):
            candidates.append((_resolve_model_path(item, config_path), None))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"model entry must be a string or object: {item!r}")
        if item.get("enabled", True) is False:
            continue
        path_text = item.get("path") or item.get("checkpoint")
        if not path_text:
            raise ValueError(f"model entry missing path: {item!r}")
        label = item.get("label")
        candidates.append((_resolve_model_path(str(path_text), config_path), str(label) if label else None))

    unique = _dedupe_paths(candidates)
    if not unique:
        unique = [(initial_checkpoint.resolve(), _model_label(initial_checkpoint))]
    return [_model_entry(idx, path, label) for idx, (path, label) in enumerate(unique)]


def _discover_model_entries(initial_checkpoint: Path, roots: list[str]) -> list[dict[str, str]]:
    candidates: list[tuple[Path, str | None]] = [(initial_checkpoint.resolve(), None)]
    for root_text in roots:
        root = Path(root_text)
        if root.is_file() and root.suffix == ".pt":
            candidates.append((root.resolve(), None))
        elif root.is_dir():
            candidates.extend((p.resolve(), None) for p in root.rglob("best_ever_params.pt"))
    unique = _dedupe_paths(candidates)
    unique.sort(key=lambda item: _display_path(item[0]).lower())
    return [_model_entry(idx, path, label) for idx, (path, label) in enumerate(unique)]


def _load_model_entries(
    initial_checkpoint: Path,
    model_list_path: Path | None,
    roots: list[str],
    discover_models: bool,
) -> list[dict[str, str]]:
    if not discover_models and model_list_path is not None and model_list_path.is_file():
        return _read_curated_model_entries(model_list_path, initial_checkpoint)
    return _discover_model_entries(initial_checkpoint, roots)

class NCAViewerState:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        seed: int,
        p_drop: float | None,
        diff_gain: float,
        model_id: str,
        model_label: str,
    ) -> None:
        self.device = device
        self.default_seed = int(seed)
        self.seed = self.default_seed
        self.diff_gain = float(diff_gain)
        self.model_id = model_id
        self.model_label = model_label
        self.lock = threading.Lock()
        self._load_checkpoint(checkpoint_path, p_drop)
        self.reset()

    def _load_checkpoint(self, checkpoint_path: Path, p_drop: float | None) -> None:
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        missing = [key for key in ("d_state", "grid", "dt") if key not in ckpt]
        if missing:
            raise ValueError(f"checkpoint missing metadata fields: {', '.join(missing)}")
        params = _checkpoint_params(ckpt)
        self.checkpoint_path = checkpoint_path
        self.d_state = int(ckpt["d_state"])
        self.trained_grid = int(ckpt["grid"])
        self.grid = self.trained_grid
        self.dt = float(ckpt["dt"])
        self.p_drop = float(ckpt.get("p_drop", 0.0) if p_drop is None else p_drop)
        self.metadata = {
            k: _jsonable(v)
            for k, v in ckpt.items()
            if k not in {"params", "best_ever_params", "best_in_gen_params"}
        }

        self.net = NCANetworkTorch(d_state=self.d_state).to(self.device)
        self.net.load_state_dict({k: v.to(self.device) for k, v in params.items()})
        self.net.eval()

    def _coerce_view_grid(self, view_grid: int | float | str | None) -> int:
        if view_grid is None or view_grid == "":
            return self.trained_grid
        grid = int(view_grid)
        max_grid = max(self.trained_grid, 128)
        grid = max(self.trained_grid, min(grid, max_grid))
        return grid

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model_id: str,
        model_label: str,
        p_drop: float | None = None,
        diff_gain: float | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self.model_id = model_id
            self.model_label = model_label
            self._load_checkpoint(checkpoint_path, p_drop)
            if diff_gain is not None:
                self.diff_gain = float(np.clip(diff_gain, 1.0, 100.0))
            self.reset(self.default_seed)
            return self.snapshot()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.init_rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.step_rng = torch.Generator(device=self.device).manual_seed(self.seed * 1009 + 17)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed * 9176 + 53)
        state = torch.rand(
            (1, self.d_state, self.grid, self.grid),
            generator=self.init_rng,
            device=self.device,
        )
        self.state = state
        self.reference = state.clone()
        self.step_count = 0

    @torch.no_grad()
    def _advance_one(self) -> None:
        if self.p_drop <= 0.0:
            mask = torch.ones((1, 1, self.grid, self.grid), device=self.device)
        else:
            mask = (
                torch.rand(
                    (1, 1, self.grid, self.grid),
                    generator=self.step_rng,
                    device=self.device,
                )
                < (1.0 - self.p_drop)
            ).to(torch.float32)
        live_delta = self.net(self.state)
        ref_delta = self.net(self.reference)
        self.state = torch.clamp(self.state + live_delta * self.dt * mask, 0.0, 1.0)
        self.reference = torch.clamp(self.reference + ref_delta * self.dt * mask, 0.0, 1.0)
        self.step_count += 1

    def step(self, n_steps: int, p_drop: float | None = None, diff_gain: float | None = None) -> dict[str, Any]:
        with self.lock:
            if p_drop is not None:
                self.p_drop = float(np.clip(p_drop, 0.0, 0.99))
            if diff_gain is not None:
                self.diff_gain = float(np.clip(diff_gain, 1.0, 100.0))
            for _ in range(max(1, min(int(n_steps), 2048))):
                self._advance_one()
            return self.snapshot()

    def set_options(self, p_drop: float | None = None, diff_gain: float | None = None) -> dict[str, Any]:
        with self.lock:
            if p_drop is not None:
                self.p_drop = float(np.clip(p_drop, 0.0, 0.99))
            if diff_gain is not None:
                self.diff_gain = float(np.clip(diff_gain, 1.0, 100.0))
            return self.snapshot()

    def set_view_grid(self, view_grid: int | float | str | None, diff_gain: float | None = None) -> dict[str, Any]:
        with self.lock:
            if diff_gain is not None:
                self.diff_gain = float(np.clip(diff_gain, 1.0, 100.0))
            self.grid = self._coerce_view_grid(view_grid)
            self.reset(self.seed)
            return self.snapshot()

    def reset_and_snapshot(self, p_drop: float | None = None, diff_gain: float | None = None) -> dict[str, Any]:
        with self.lock:
            if p_drop is not None:
                self.p_drop = float(np.clip(p_drop, 0.0, 0.99))
            if diff_gain is not None:
                self.diff_gain = float(np.clip(diff_gain, 1.0, 100.0))
            self.seed += 1
            self.reset(self.seed)
            return self.snapshot()

    @torch.no_grad()
    def inject(
        self,
        x: int,
        y: int,
        size: int,
        strength: float,
        mode: str,
        diff_gain: float | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if diff_gain is not None:
                self.diff_gain = float(np.clip(diff_gain, 1.0, 100.0))
            size = max(1, min(int(size), self.grid))
            if size % 2 == 0 and size > 1:
                size -= 1
            radius = size // 2
            strength = float(np.clip(strength, 0.0, 1.0))
            xs = torch.tensor(
                [(int(x) + dx) % self.grid for dx in range(-radius, radius + 1)],
                device=self.device,
                dtype=torch.long,
            )
            ys = torch.tensor(
                [(int(y) + dy) % self.grid for dy in range(-radius, radius + 1)],
                device=self.device,
                dtype=torch.long,
            )
            patch = self.state[:, :, ys][:, :, :, xs]
            mode = str(mode)
            if mode == "add":
                noise = (
                    torch.rand(patch.shape, generator=self.noise_rng, device=self.device) * 2.0 - 1.0
                )
                updated = patch + noise * strength
            elif mode == "zero":
                updated = patch * (1.0 - strength)
            elif mode == "one":
                updated = patch * (1.0 - strength) + strength
            elif mode == "invert":
                updated = patch * (1.0 - strength) + (1.0 - patch) * strength
            else:
                noise = torch.rand(patch.shape, generator=self.noise_rng, device=self.device)
                updated = patch * (1.0 - strength) + noise * strength
            updated = torch.clamp(updated, 0.0, 1.0)
            self.state[:, :, ys[:, None], xs[None, :]] = updated
            return self.snapshot()

    def snapshot_threadsafe(self) -> dict[str, Any]:
        with self.lock:
            return self.snapshot()

    def _rgb_u8(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor[0].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 2:
            arr = np.concatenate([arr, np.zeros_like(arr[:, :, :1])], axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return (arr * 255.0 + 0.5).astype(np.uint8)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "grid": self.grid,
            "view_grid": self.grid,
            "trained_grid": self.trained_grid,
            "d_state": self.d_state,
            "dt": self.dt,
            "active_model_id": self.model_id,
            "model_label": self.model_label,
            "checkpoint_name": self.model_label or self.checkpoint_path.parent.name,
            "checkpoint_source": _display_path(self.checkpoint_path),
            "seed": self.seed,
            "default_seed": self.default_seed,
            "state_image": _png_data_url(self._rgb_u8(self.state)),
            "metadata": self.metadata,
        }

class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "NCANoiseViewer/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    @property
    def viewer(self) -> NCAViewerState:
        return self.server.viewer  # type: ignore[attr-defined]

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(_json_bytes(payload), "application/json", status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _handle_error(self, exc: Exception) -> None:
        self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            try:
                query = parse_qs(parsed.query)
                diff_gain = query.get("diff_gain", [None])[0]
                payload = self.viewer.set_options(
                    diff_gain=float(diff_gain) if diff_gain is not None else None,
                )
                self._send_json(payload)
            except Exception as exc:
                self._handle_error(exc)
            return
        if parsed.path == "/api/models":
            entries = _load_model_entries(
                self.server.initial_checkpoint_path,  # type: ignore[attr-defined]
                self.server.model_list_path,  # type: ignore[attr-defined]
                self.server.model_roots,  # type: ignore[attr-defined]
                self.server.discover_models,  # type: ignore[attr-defined]
            )
            active_path = self.viewer.checkpoint_path.resolve()
            active_entry = next(
                (entry for entry in entries if Path(entry["path"]).resolve() == active_path),
                None,
            )
            if active_entry is None:
                active_entry = _model_entry(len(entries), active_path, f"{_model_label(active_path)} (current)")
                entries.append(active_entry)
            self.viewer.model_id = active_entry["id"]
            self.viewer.model_label = active_entry["label"]
            self.server.models_by_id = {entry["id"]: entry for entry in entries}  # type: ignore[attr-defined]
            self.server.public_models = [  # type: ignore[attr-defined]
                _public_model_entry(e)
                for e in entries
            ]
            self._send_json({
                "models": self.server.public_models,  # type: ignore[attr-defined]
                "active_id": self.viewer.model_id,
            })
            return
        if parsed.path == "/api/model-media":
            try:
                query = parse_qs(parsed.query)
                model_id = query.get("id", [""])[0]
                kind = query.get("kind", [""])[0]
                if kind not in {"training", "probe"}:
                    raise ValueError(f"unknown media kind: {kind}")
                models_by_id = getattr(self.server, "models_by_id", {})
                if model_id not in models_by_id:
                    entries = _load_model_entries(
                        self.server.initial_checkpoint_path,  # type: ignore[attr-defined]
                        self.server.model_list_path,  # type: ignore[attr-defined]
                        self.server.model_roots,  # type: ignore[attr-defined]
                        self.server.discover_models,  # type: ignore[attr-defined]
                    )
                    models_by_id = {entry["id"]: entry for entry in entries}
                    self.server.models_by_id = models_by_id  # type: ignore[attr-defined]
                entry = models_by_id.get(model_id)
                if entry is None:
                    raise KeyError(f"unknown model id: {model_id}")
                media_path = _model_media_assets(entry).get(kind)
                if media_path is None:
                    self._send_json({"error": "media not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send(_model_media_png(media_path, kind), "image/png")
            except Exception as exc:
                self._handle_error(exc)
            return
        if parsed.path == "/favicon.ico":
            self._send(b"", mimetypes.types_map.get(".ico", "image/x-icon"))
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
            if parsed.path == "/api/step":
                payload = self.viewer.step(
                    int(data.get("steps", 1)),
                    p_drop=data.get("p_drop"),
                    diff_gain=data.get("diff_gain"),
                )
            elif parsed.path == "/api/inject":
                payload = self.viewer.inject(
                    int(data.get("x", self.viewer.grid // 2)),
                    int(data.get("y", self.viewer.grid // 2)),
                    int(data.get("size", 5)),
                    float(data.get("strength", 1.0)),
                    str(data.get("mode", "uniform")),
                    diff_gain=data.get("diff_gain"),
                )
            elif parsed.path == "/api/reset":
                payload = self.viewer.reset_and_snapshot(
                    p_drop=data.get("p_drop"),
                    diff_gain=data.get("diff_gain"),
                )
            elif parsed.path == "/api/options":
                payload = self.viewer.set_options(
                    p_drop=data.get("p_drop"),
                    diff_gain=data.get("diff_gain"),
                )
            elif parsed.path == "/api/view-grid":
                payload = self.viewer.set_view_grid(
                    data.get("view_grid"),
                    diff_gain=data.get("diff_gain"),
                )
            elif parsed.path == "/api/load":
                model_id = str(data.get("id", ""))
                models_by_id = self.server.models_by_id  # type: ignore[attr-defined]
                if model_id not in models_by_id:
                    raise KeyError(f"unknown model id: {model_id}")
                entry = models_by_id[model_id]
                payload = self.viewer.load_checkpoint(
                    Path(entry["path"]),
                    entry["id"],
                    entry["label"],
                    p_drop=data.get("p_drop"),
                    diff_gain=data.get("diff_gain"),
                )
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)
        except Exception as exc:
            self._handle_error(exc)

def _resolve_checkpoint(run_dir: str, checkpoint: str) -> Path:
    run_path = Path(run_dir)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_path / ckpt_path
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    return ckpt_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=str, default="best_ever_params.pt")
    parser.add_argument(
        "--model-list",
        type=str,
        default=str(MODEL_WEIGHTS_PATH),
        help="JSON file containing the manually curated model list.",
    )
    parser.add_argument(
        "--discover-models",
        action="store_true",
        help="Ignore --model-list and scan --model-root directories recursively instead.",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        default=None,
        help="Discovery mode root to scan recursively for best_ever_params.pt. Can be passed more than once.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=12353)
    parser.add_argument("--p-drop", type=float, default=None)
    parser.add_argument("--diff-gain", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint_path = _resolve_checkpoint(args.run_dir, args.checkpoint)
    device = _pick_device(args.device)
    model_roots = list(args.model_root or DEFAULT_MODEL_ROOTS)
    if args.run_dir not in model_roots:
        model_roots.insert(0, args.run_dir)
    model_list_path = Path(args.model_list).resolve() if args.model_list else None
    model_entries = _load_model_entries(
        checkpoint_path,
        model_list_path,
        model_roots,
        args.discover_models,
    )
    active_entry = next(
        (entry for entry in model_entries if Path(entry["path"]).resolve() == checkpoint_path.resolve()),
        model_entries[0],
    )
    checkpoint_path = Path(active_entry["path"]).resolve()
    public_models = [
        _public_model_entry(e)
        for e in model_entries
    ]
    viewer = NCAViewerState(
        checkpoint_path=checkpoint_path,
        device=device,
        seed=args.seed,
        p_drop=args.p_drop,
        diff_gain=args.diff_gain,
        model_id=active_entry["id"],
        model_label=active_entry["label"],
    )
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.viewer = viewer  # type: ignore[attr-defined]
    server.initial_checkpoint_path = checkpoint_path  # type: ignore[attr-defined]
    server.model_list_path = model_list_path  # type: ignore[attr-defined]
    server.model_roots = model_roots  # type: ignore[attr-defined]
    server.discover_models = args.discover_models  # type: ignore[attr-defined]
    server.models_by_id = {entry["id"]: entry for entry in model_entries}  # type: ignore[attr-defined]
    server.public_models = public_models  # type: ignore[attr-defined]
    server.quiet = args.quiet  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"loaded {checkpoint_path}")
    print(f"models={len(model_entries)} roots={', '.join(model_roots)}")
    print(f"device={device} grid={viewer.grid} d_state={viewer.d_state} dt={viewer.dt}")
    print(f"open {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
