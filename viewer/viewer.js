(function () {
  "use strict";

  var MODEL_URL = "models.json";
  var DEFAULT_SEED = 12353;
  var DEFAULT_VIEW_GRID = 32;

  var canvas = document.getElementById("stateCanvas");
  var ctx = canvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;

  var controls = {
    modelSelect: document.getElementById("modelSelect"),
    gridSelect: document.getElementById("gridSelect"),
    seedInput: document.getElementById("seedInput"),
    pDrop: document.getElementById("pDrop"),
    diffGain: document.getElementById("diffGain"),
    patchSize: document.getElementById("patchSize"),
    strength: document.getElementById("strength"),
    noiseMode: document.getElementById("noiseMode"),
    stepsPerTick: document.getElementById("stepsPerTick"),
    tickMs: document.getElementById("tickMs")
  };

  var models = [];
  var activeModel = null;
  var weights = null;
  var grid = 32;
  var trainedGrid = 32;
  var dState = 3;
  var dt = 0.05;
  var seed = DEFAULT_SEED;
  var stepCount = 0;
  var running = false;
  var runTimer = 0;
  var dragging = false;
  var lastPoint = null;

  var state = new Float32Array(0);
  var reference = new Float32Array(0);
  var nextState = new Float32Array(0);
  var nextReference = new Float32Array(0);
  var conv3 = new Float32Array(0);
  var hidden = new Float32Array(0);
  var imageData = null;

  var initRng = makeRng(seed);
  var stepRng = makeRng(seed * 1009 + 17);
  var noiseRng = makeRng(seed * 9176 + 53);

  function el(id) {
    return document.getElementById(id);
  }

  function setStatus(message) {
    el("status").textContent = message || "";
  }

  function clamp01(value) {
    return value < 0 ? 0 : value > 1 ? 1 : value;
  }

  function selectedSeed() {
    var parsed = Number(controls.seedInput.value);
    if (!isFinite(parsed)) return DEFAULT_SEED;
    return Math.max(0, Math.floor(parsed));
  }

  function makeRng(seedValue) {
    var a = seedValue >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function idx(x, y, c) {
    return ((y * grid + x) * dState) + c;
  }

  function cellBase(x, y) {
    return (y * grid + x) * dState;
  }

  function paramData(params, name) {
    if (!params[name] || !params[name].data) {
      throw new Error("model is missing " + name);
    }
    return new Float32Array(params[name].data);
  }

  function unpackWeights(model) {
    var params = model.params || {};
    return {
      conv3w: paramData(params, "conv3.weight"),
      conv3b: paramData(params, "conv3.bias"),
      conv1aw: paramData(params, "conv1a.weight"),
      conv1ab: paramData(params, "conv1a.bias"),
      conv1bw: paramData(params, "conv1b.weight"),
      conv1bb: paramData(params, "conv1b.bias")
    };
  }

  function defaultViewGrid(baseGrid) {
    return Math.max(Number(baseGrid) || DEFAULT_VIEW_GRID, DEFAULT_VIEW_GRID);
  }

  function viewGridChoices(baseGrid) {
    var choices = [];
    var value = Math.max(1, baseGrid);
    while (value <= 128) {
      choices.push(value);
      value *= 2;
    }
    if (choices.indexOf(128) < 0) choices.push(128);
    return choices;
  }

  function syncControls() {
    el("gridOut").value = grid + "x" + grid;
    el("seedOut").value = String(seed);
    el("pDropOut").value = Number(controls.pDrop.value).toFixed(2);
    el("diffGainOut").value = controls.diffGain.value + "x";
    el("patchSizeOut").value = controls.patchSize.value;
    el("strengthOut").value = Number(controls.strength.value).toFixed(2);
    el("stepsOut").value = controls.stepsPerTick.value;
    el("intervalOut").value = controls.tickMs.value + " ms";
  }

  function syncReadout() {
    el("stepReadout").value = "step " + stepCount + " | seed " + seed;
  }

  function renderModelPanel() {
    if (!activeModel) return;
    el("modelTitle").textContent = activeModel.label;
    el("modelSummary").textContent = activeModel.summary || activeModel.source || "exported checkpoint";

    renderProbeArchitectures(activeModel);

    var details = activeModel.details && activeModel.details.length ? activeModel.details.map(function (item) {
      return [item.label, item.value];
    }) : [
      ["grid", String(activeModel.grid)],
      ["d_state", String(activeModel.d_state)],
      ["dt", fmt(activeModel.dt)],
      ["p_drop", fmt(activeModel.p_drop || 0)],
      ["probe", activeModel.metadata && activeModel.metadata.probe_archs ? activeModel.metadata.probe_archs.join(", ") : activeModel.metadata && activeModel.metadata.probe_arch],
      ["horizon", horizonLabel(activeModel.metadata || {})],
      ["gzip", gzipLabel(activeModel.metadata || {})]
    ];
    details.push(["source", activeModel.source]);

    var root = el("modelStats");
    root.replaceChildren();
    details.forEach(function (item) {
      if (item[1] === undefined || item[1] === null || item[1] === "") return;
      var dtEl = document.createElement("dt");
      var ddEl = document.createElement("dd");
      dtEl.textContent = item[0];
      ddEl.textContent = item[1];
      root.appendChild(dtEl);
      root.appendChild(ddEl);
    });

    renderModelMedia(activeModel);
  }

  function renderProbeArchitectures(model) {
    var root = el("probeArchitectures");
    root.replaceChildren();
    (model.probe_architectures || []).forEach(function (arch) {
      var details = document.createElement("details");
      var summary = document.createElement("summary");
      var body = document.createElement("code");
      summary.textContent = arch.label;
      body.className = "formula";
      body.textContent = arch.body;
      details.appendChild(summary);
      details.appendChild(body);
      root.appendChild(details);
    });
  }

  function renderModelMedia(model) {
    var root = el("modelMedia");
    root.replaceChildren();
    addMedia(root, model, "Training curve", model.training_curve_url);
    addMedia(root, model, "Probe loss curves", model.probe_curve_url);
    if (!root.children.length) {
      var empty = document.createElement("div");
      empty.className = "media-empty";
      empty.textContent = "Curve images are not available for this checkpoint.";
      root.appendChild(empty);
    }
  }

  function addMedia(root, model, title, url) {
    if (!url) return;
    var block = document.createElement("div");
    var head = document.createElement("div");
    var label = document.createElement("div");
    var zoom = document.createElement("button");
    var image = document.createElement("img");
    var dialogTitle = model.label + " - " + title;

    block.className = "media-block";
    head.className = "media-head";
    label.className = "media-title";
    label.textContent = title;
    zoom.type = "button";
    zoom.className = "media-zoom";
    zoom.textContent = "Enlarge";
    zoom.addEventListener("click", function () {
      openMediaDialog(dialogTitle, url);
    });
    image.src = url;
    image.alt = dialogTitle;
    image.loading = "lazy";
    image.addEventListener("click", function () {
      openMediaDialog(dialogTitle, url);
    });

    head.appendChild(label);
    head.appendChild(zoom);
    block.appendChild(head);
    block.appendChild(image);
    root.appendChild(block);
  }

  function openMediaDialog(title, url) {
    var dialog = el("mediaDialog");
    var image = el("mediaDialogImg");
    el("mediaDialogTitle").textContent = title;
    image.src = url;
    image.alt = title;
    dialog.classList.add("open");
    dialog.setAttribute("aria-hidden", "false");
    el("mediaDialogClose").focus();
  }

  function closeMediaDialog() {
    var dialog = el("mediaDialog");
    dialog.classList.remove("open");
    dialog.setAttribute("aria-hidden", "true");
    el("mediaDialogImg").removeAttribute("src");
  }

  function fmt(value) {
    if (typeof value !== "number") return String(value);
    if (!isFinite(value)) return String(value);
    return Math.abs(value) >= 10 ? value.toFixed(2) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }

  function horizonLabel(meta) {
    if (!meta.horizon_mode && !meta.probe_horizon && !meta.multi_ks) return "";
    if (meta.horizon_mode === "multi" && meta.multi_ks) return "multi K=" + meta.multi_ks.join(",");
    if (meta.horizon_mode && meta.probe_horizon) return meta.horizon_mode + " K=" + meta.probe_horizon;
    return meta.horizon_mode || "K=" + meta.probe_horizon;
  }

  function gzipLabel(meta) {
    if (meta.gzip_mode === "band") return "target " + fmt(meta.gzip_target);
    if (meta.gzip_threshold !== undefined && meta.gzip_threshold !== null) return "floor " + fmt(meta.gzip_threshold);
    return "";
  }

  function resetGridOptions() {
    controls.gridSelect.replaceChildren();
    viewGridChoices(trainedGrid).forEach(function (value) {
      var option = document.createElement("option");
      option.value = String(value);
      option.textContent = value === trainedGrid ? value + " x " + value + " (trained)" : value + " x " + value;
      controls.gridSelect.appendChild(option);
    });
    controls.gridSelect.value = String(grid);
  }

  function allocateArrays() {
    var cells = grid * grid;
    state = new Float32Array(cells * dState);
    reference = new Float32Array(cells * dState);
    nextState = new Float32Array(cells * dState);
    nextReference = new Float32Array(cells * dState);
    conv3 = new Float32Array(cells * 4);
    hidden = new Float32Array(cells * 16);
    canvas.width = grid;
    canvas.height = grid;
    imageData = ctx.createImageData(grid, grid);
    applyDelta._lastMasks = new Array(cells).fill(1);
  }

  function resetState(nextSeed) {
    if (typeof nextSeed === "number") seed = nextSeed;
    initRng = makeRng(seed);
    stepRng = makeRng(seed * 1009 + 17);
    noiseRng = makeRng(seed * 9176 + 53);
    allocateArrays();
    for (var i = 0; i < state.length; i += 1) {
      state[i] = initRng();
      reference[i] = state[i];
    }
    stepCount = 0;
    controls.seedInput.value = String(seed);
    draw();
    syncReadout();
    syncControls();
  }

  function loadModel(modelId, resetSeed) {
    var found = models.find(function (item) { return item.id === modelId; }) || models[0];
    if (!found) throw new Error("no exported models found");
    activeModel = found;
    controls.modelSelect.value = activeModel.id;
    weights = unpackWeights(found);
    trainedGrid = Number(found.grid || 32);
    dState = Number(found.d_state || 3);
    dt = Number(found.dt || 0.05);
    grid = defaultViewGrid(trainedGrid);
    controls.pDrop.value = String(Number(found.p_drop || 0));
    resetGridOptions();
    renderModelPanel();
    resetState(typeof resetSeed === "number" ? resetSeed : selectedSeed());
    syncControls();
  }

  function stepOne() {
    var pDrop = Number(controls.pDrop.value);
    var keepProb = 1 - pDrop;

    evalConv(state);
    applyDelta(state, nextState, pDrop, keepProb, true);
    evalConv(reference);
    applyDelta(reference, nextReference, pDrop, keepProb, false);

    var swap = state;
    state = nextState;
    nextState = swap;
    swap = reference;
    reference = nextReference;
    nextReference = swap;
    stepCount += 1;
  }

  function evalConv(source) {
    var conv3w = weights.conv3w;
    var conv3b = weights.conv3b;
    var conv1aw = weights.conv1aw;
    var conv1ab = weights.conv1ab;

    for (var y = 0; y < grid; y += 1) {
      for (var x = 0; x < grid; x += 1) {
        var cell = y * grid + x;
        for (var oc = 0; oc < 4; oc += 1) {
          var acc = conv3b[oc];
          for (var ic = 0; ic < dState; ic += 1) {
            for (var ky = 0; ky < 3; ky += 1) {
              var sy = (y + ky + grid - 1) % grid;
              for (var kx = 0; kx < 3; kx += 1) {
                var sx = (x + kx + grid - 1) % grid;
                var wIndex = (((oc * dState + ic) * 3 + ky) * 3 + kx);
                acc += conv3w[wIndex] * source[idx(sx, sy, ic)];
              }
            }
          }
          conv3[cell * 4 + oc] = acc;
        }

        for (var h = 0; h < 16; h += 1) {
          var hAcc = conv1ab[h];
          for (var c4 = 0; c4 < 4; c4 += 1) {
            hAcc += conv1aw[h * 4 + c4] * conv3[cell * 4 + c4];
          }
          hidden[cell * 16 + h] = hAcc > 0 ? hAcc : 0;
        }
      }
    }
  }

  function applyDelta(source, target, pDrop, keepProb, advanceMask) {
    var conv1bw = weights.conv1bw;
    var conv1bb = weights.conv1bb;
    var cellCount = grid * grid;

    for (var cell = 0; cell < cellCount; cell += 1) {
      var mask;
      if (pDrop <= 0) {
        mask = 1;
      } else if (advanceMask) {
        mask = stepRng() < keepProb ? 1 : 0;
        applyDelta._lastMasks[cell] = mask;
      } else {
        mask = applyDelta._lastMasks[cell] === undefined ? 1 : applyDelta._lastMasks[cell];
      }

      for (var oc = 0; oc < dState; oc += 1) {
        var acc = conv1bb[oc];
        for (var h = 0; h < 16; h += 1) {
          acc += conv1bw[oc * 16 + h] * hidden[cell * 16 + h];
        }
        var i = cell * dState + oc;
        target[i] = clamp01(source[i] + acc * dt * mask);
      }
    }
  }
  applyDelta._lastMasks = [];

  function stepMany(count) {
    var n = Math.max(1, Math.min(512, count | 0));
    if (applyDelta._lastMasks.length !== grid * grid) {
      applyDelta._lastMasks = new Array(grid * grid).fill(1);
    }
    for (var i = 0; i < n; i += 1) {
      stepOne();
    }
    draw();
    syncReadout();
  }

  function draw() {
    if (!imageData) return;
    var data = imageData.data;
    var p = 0;
    for (var y = 0; y < grid; y += 1) {
      for (var x = 0; x < grid; x += 1) {
        var base = cellBase(x, y);
        var r = state[base];
        var g = dState > 1 ? state[base + 1] : r;
        var b = dState > 2 ? state[base + 2] : 0;
        data[p] = Math.round(clamp01(r) * 255);
        data[p + 1] = Math.round(clamp01(g) * 255);
        data[p + 2] = Math.round(clamp01(b) * 255);
        data[p + 3] = 255;
        p += 4;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function runTick() {
    if (!running) return;
    stepMany(Number(controls.stepsPerTick.value));
  }

  function restartRunTimer() {
    if (runTimer) {
      clearInterval(runTimer);
      runTimer = 0;
    }
    if (!running) return;
    runTimer = setInterval(runTick, Math.max(20, Number(controls.tickMs.value) || 80));
  }

  function setRunning(next) {
    running = next;
    el("runBtn").textContent = running ? "Pause" : "Run";
    restartRunTimer();
  }

  function canvasPoint(evt) {
    var rect = canvas.getBoundingClientRect();
    var x = Math.floor((evt.clientX - rect.left) / rect.width * grid);
    var y = Math.floor((evt.clientY - rect.top) / rect.height * grid);
    return {
      x: Math.max(0, Math.min(grid - 1, x)),
      y: Math.max(0, Math.min(grid - 1, y))
    };
  }

  function drawPatchBox(point) {
    var box = el("patchBox");
    var rect = canvas.getBoundingClientRect();
    var size = Number(controls.patchSize.value);
    var cell = rect.width / grid;
    box.style.display = "block";
    box.style.width = Math.max(1, size * cell) + "px";
    box.style.height = Math.max(1, size * cell) + "px";
    box.style.left = ((point.x + 0.5 - size / 2) * cell) + "px";
    box.style.top = ((point.y + 0.5 - size / 2) * cell) + "px";
  }

  function samePoint(a, b) {
    return a && b && a.x === b.x && a.y === b.y;
  }

  function injectAt(point) {
    var size = Number(controls.patchSize.value) | 0;
    if (size % 2 === 0) size += 1;
    var radius = Math.floor(size / 2);
    var strength = Number(controls.strength.value);
    var mode = controls.noiseMode.value;

    for (var dy = -radius; dy <= radius; dy += 1) {
      var y = (point.y + dy + grid) % grid;
      for (var dx = -radius; dx <= radius; dx += 1) {
        var x = (point.x + dx + grid) % grid;
        var base = cellBase(x, y);
        for (var c = 0; c < dState; c += 1) {
          var current = state[base + c];
          var updated;
          if (mode === "add") {
            updated = current + (noiseRng() * 2 - 1) * strength;
          } else if (mode === "zero") {
            updated = current * (1 - strength);
          } else if (mode === "one") {
            updated = current * (1 - strength) + strength;
          } else if (mode === "invert") {
            updated = current * (1 - strength) + (1 - current) * strength;
          } else {
            updated = current * (1 - strength) + noiseRng() * strength;
          }
          state[base + c] = clamp01(updated);
        }
      }
    }
    draw();
  }

  function bindEvents() {
    el("runBtn").addEventListener("click", function () {
      setRunning(!running);
    });
    el("stepBtn").addEventListener("click", function () {
      stepMany(Number(controls.stepsPerTick.value));
    });
    el("resetBtn").addEventListener("click", function () {
      resetState(seed + 1);
    });

    controls.modelSelect.addEventListener("change", function () {
      setRunning(false);
      loadModel(controls.modelSelect.value, DEFAULT_SEED);
      setStatus("");
      setRunning(true);
    });
    controls.seedInput.addEventListener("change", function () {
      setRunning(false);
      resetState(selectedSeed());
    });
    controls.gridSelect.addEventListener("change", function () {
      var wasRunning = running;
      setRunning(false);
      grid = Number(controls.gridSelect.value);
      resetState(seed);
      syncControls();
      if (wasRunning) setRunning(true);
    });
    controls.pDrop.addEventListener("input", syncControls);
    controls.diffGain.addEventListener("input", function () {
      syncControls();
      draw();
    });
    controls.patchSize.addEventListener("input", syncControls);
    controls.strength.addEventListener("input", syncControls);
    controls.stepsPerTick.addEventListener("input", syncControls);
    controls.tickMs.addEventListener("input", function () {
      syncControls();
      restartRunTimer();
    });

    el("mediaDialogClose").addEventListener("click", closeMediaDialog);
    el("mediaDialog").addEventListener("click", function (evt) {
      if (evt.target === el("mediaDialog")) closeMediaDialog();
    });
    window.addEventListener("keydown", function (evt) {
      if (evt.key === "Escape" && el("mediaDialog").classList.contains("open")) {
        closeMediaDialog();
      }
    });

    canvas.addEventListener("pointerdown", function (evt) {
      canvas.setPointerCapture(evt.pointerId);
      dragging = true;
      lastPoint = canvasPoint(evt);
      drawPatchBox(lastPoint);
      injectAt(lastPoint);
    });
    canvas.addEventListener("pointermove", function (evt) {
      var point = canvasPoint(evt);
      drawPatchBox(point);
      if (!dragging || samePoint(point, lastPoint)) return;
      lastPoint = point;
      injectAt(point);
    });
    canvas.addEventListener("pointerup", function () {
      dragging = false;
      lastPoint = null;
    });
    canvas.addEventListener("pointerleave", function () {
      el("patchBox").style.display = "none";
      dragging = false;
      lastPoint = null;
    });
  }

  function initModelList() {
    controls.modelSelect.replaceChildren();
    models.forEach(function (model) {
      var option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      option.title = model.source || model.id;
      controls.modelSelect.appendChild(option);
    });
    el("modelCount").value = String(models.length);
  }

  function boot() {
    bindEvents();
    fetch(MODEL_URL)
      .then(function (response) {
        if (!response.ok) throw new Error("could not load " + MODEL_URL);
        return response.json();
      })
      .then(function (payload) {
        models = payload.models || [];
        if (!models.length) throw new Error("models.json has no models");
        initModelList();
        loadModel(models[0].id, DEFAULT_SEED);
        setRunning(true);
        setStatus("");
      })
      .catch(function (err) {
        setStatus(err.message);
        el("modelTitle").textContent = "Model load failed";
        el("modelSummary").textContent = "The static model export is missing or malformed.";
      });
  }

  boot();
}());
