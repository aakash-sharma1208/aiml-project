/**
 * script.js
 * ---------
 * Frontend logic for the Handwritten Digit Recognition dashboard.
 *
 * Responsibilities:
 *   - SPA navigation (sidebar links)
 *   - Drawing canvas (mouse + touch)
 *   - Image preprocessing (crop → resize → invert → scale 0-16)
 *   - API calls to Flask backend
 *   - Rendering charts (Chart.js): scores, per-class metrics, class dist, CM
 *   - Dataset gallery and example digit rendering
 */

"use strict";

/* ═══════════════════════════════════════════════════════════
   1. NAVIGATION
   ═══════════════════════════════════════════════════════════ */

const navItems = document.querySelectorAll(".nav-item");
const pages    = document.querySelectorAll(".page");
const sidebar  = document.getElementById("sidebar");
const menuBtn  = document.getElementById("menuBtn");

// Track which pages have been loaded
const loaded = {};

function showPage(pageId) {
  pages.forEach(p => p.classList.remove("active"));
  navItems.forEach(n => n.classList.remove("active"));

  const target = document.getElementById(`page-${pageId}`);
  const navEl  = document.querySelector(`.nav-item[data-page="${pageId}"]`);
  if (target) target.classList.add("active");
  if (navEl)  navEl.classList.add("active");

  // Lazy-load page data once
  if (!loaded[pageId]) {
    loaded[pageId] = true;
    switch (pageId) {
      case "overview":   loadOverview();   break;
      case "evaluation": loadEvaluation(); break;
      case "dataset":    loadDataset();    break;
      case "recognition":loadExamples();   break;
    }
  }

  // Close mobile sidebar
  sidebar.classList.remove("open");
}

navItems.forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    showPage(item.dataset.page);
  });
});

menuBtn?.addEventListener("click", () => sidebar.classList.toggle("open"));

// Close sidebar when clicking outside on mobile
document.addEventListener("click", e => {
  if (window.innerWidth <= 900 &&
      !sidebar.contains(e.target) &&
      !menuBtn.contains(e.target)) {
    sidebar.classList.remove("open");
  }
});

// Initial page load
showPage("overview");


/* ═══════════════════════════════════════════════════════════
   2. DRAWING CANVAS
   ═══════════════════════════════════════════════════════════ */

const canvas   = document.getElementById("digitCanvas");
const ctx      = canvas.getContext("2d");

// Canvas state
let drawing   = false;
let strokes   = [];      // array of arrays of points for undo
let currStroke = null;

// Drawing settings
const LINE_WIDTH = 22;
const LINE_COLOR = "#000000";

function initCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth   = LINE_WIDTH;
  ctx.lineCap     = "round";
  ctx.lineJoin    = "round";
  ctx.strokeStyle = LINE_COLOR;
}

initCanvas();

// ── Coordinate helpers ─────────────────────────────────────

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  if (e.touches) {
    const t = e.touches[0];
    return {
      x: (t.clientX - rect.left) * scaleX,
      y: (t.clientY - rect.top)  * scaleY,
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  };
}

// ── Drawing events ─────────────────────────────────────────

function startDraw(e) {
  e.preventDefault();
  drawing     = true;
  currStroke  = [];
  const pos   = getPos(e);
  currStroke.push(pos);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  // Draw a dot for click-only
  ctx.arc(pos.x, pos.y, LINE_WIDTH / 2, 0, Math.PI * 2);
  ctx.fillStyle = LINE_COLOR;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const pos = getPos(e);
  currStroke.push(pos);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function endDraw(e) {
  if (!drawing) return;
  drawing = false;
  if (currStroke && currStroke.length > 0) strokes.push(currStroke);
  currStroke = null;
  ctx.beginPath();
}

canvas.addEventListener("mousedown",  startDraw);
canvas.addEventListener("mousemove",  draw);
canvas.addEventListener("mouseup",    endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove",  draw,      { passive: false });
canvas.addEventListener("touchend",   endDraw);

// ── Clear & Undo ───────────────────────────────────────────

document.getElementById("btnClear").addEventListener("click", () => {
  strokes = [];
  initCanvas();
  resetResult();
});

document.getElementById("btnUndo").addEventListener("click", () => {
  if (strokes.length === 0) return;
  strokes.pop();
  initCanvas();
  // Redraw remaining strokes
  strokes.forEach(stroke => {
    if (stroke.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(stroke[0].x, stroke[0].y);
    stroke.forEach(pt => {
      ctx.lineTo(pt.x, pt.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pt.x, pt.y);
    });
    // Also draw dot for single-point strokes
    if (stroke.length === 1) {
      ctx.arc(stroke[0].x, stroke[0].y, LINE_WIDTH / 2, 0, Math.PI * 2);
      ctx.fillStyle = LINE_COLOR;
      ctx.fill();
    }
    ctx.beginPath();
  });
  resetResult();
});


/* ═══════════════════════════════════════════════════════════
   3. IMAGE PREPROCESSING
   ═══════════════════════════════════════════════════════════ */

/**
 * Convert the canvas drawing to 64 pixel values (0–16)
 * compatible with the sklearn digits dataset.
 *
 * Steps:
 *   1. Get raw image data
 *   2. Convert to grayscale
 *   3. Crop the non-white bounding box with padding
 *   4. Resize to 8×8 using an off-screen canvas (bilinear via browser)
 *   5. Invert (white bg → dark bg) and scale to 0–16
 *
 * Returns null if the canvas appears blank.
 */
function extractPixels() {
  const w = canvas.width;
  const h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data      = imageData.data;  // RGBA flat array

  // Convert to grayscale values (0=white, 255=black equivalent)
  // We store the "ink" amount: 0 = white (no ink), 255 = black (full ink)
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    // Standard luminance
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    // "Ink" = inverted luminance (white bg → 0 ink, black stroke → 255 ink)
    gray[i] = 255 - lum;
  }

  // Find bounding box of non-blank pixels
  const THRESHOLD = 10;  // ink > 10 counts as drawn
  let minX = w, maxX = 0, minY = h, maxY = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (gray[y * w + x] > THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Check if canvas is blank
  if (minX > maxX || minY > maxY) return null;

  // Add proportional padding around the drawn stroke
  const pad = Math.max(12, Math.round(Math.max(maxX - minX, maxY - minY) * 0.14));
  minX = Math.max(0,  minX - pad);
  minY = Math.max(0,  minY - pad);
  maxX = Math.min(w - 1, maxX + pad);
  maxY = Math.min(h - 1, maxY + pad);

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;

  // Draw the cropped region onto an intermediate 8x8 canvas
  const SIZE = 8;
  const offCanvas = document.createElement("canvas");
  offCanvas.width  = SIZE;
  offCanvas.height = SIZE;
  const offCtx    = offCanvas.getContext("2d");

  // Fill with white background
  offCtx.fillStyle = "#ffffff";
  offCtx.fillRect(0, 0, SIZE, SIZE);

  // Aspect-ratio preserving fit into 6.5 x 6.5 inner area inside 8x8
  const TARGET = 6.4;
  let dw, dh, dx, dy;
  if (bw > bh) {
    dw = TARGET;
    dh = Math.max(1, (TARGET * bh) / bw);
  } else {
    dh = TARGET;
    dw = Math.max(1, (TARGET * bw) / bh);
  }
  dx = (SIZE - dw) / 2;
  dy = (SIZE - dh) / 2;

  // Draw cropped section of original canvas scaled down
  offCtx.drawImage(canvas, minX, minY, bw, bh, dx, dy, dw, dh);

  // Read 8×8 pixel data from the off-screen canvas
  const offData = offCtx.getImageData(0, 0, SIZE, SIZE).data;

  // Measure peak ink
  let maxInk = 0;
  const rawInks = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const r   = offData[i * 4];
    const g   = offData[i * 4 + 1];
    const b   = offData[i * 4 + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;  // 0=black, 255=white
    const ink = Math.max(0, 255 - lum);              // inverted: 255=ink
    rawInks[i] = ink;
    if (ink > maxInk) maxInk = ink;
  }

  // Reject blank canvas or tiny smudge
  if (maxInk < 20) return null;

  // Contrast-normalize so the peak ink reaches 16.0
  const pixels = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    const val = Math.round((rawInks[i] / maxInk) * 16);
    pixels.push(Math.max(0, Math.min(16, val)));
  }

  return { pixels, previewCanvas: offCanvas };
}


/* ═══════════════════════════════════════════════════════════
   4. PREDICTION
   ═══════════════════════════════════════════════════════════ */

let predictInProgress = false;
let scoresChartInstance = null;

document.getElementById("btnPredict").addEventListener("click", predictDigit);
document.getElementById("btnTryAgain").addEventListener("click", () => {
  strokes = [];
  initCanvas();
  resetResult();
});
document.getElementById("btnErrorRetry").addEventListener("click", () => {
  resetResult();
});

async function predictDigit() {
  if (predictInProgress) return;

  if (strokes.length === 0) {
    showResultError("The canvas appears to be blank. Please draw a digit first.");
    return;
  }

  predictInProgress = true;
  document.getElementById("btnPredict").disabled = true;
  showResultLoading();

  try {
    const dataUrl = canvas.toDataURL("image/png");
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    const data = await response.json();

    if (!response.ok) {
      showResultError(data.error || `Server error: ${response.status}`);
      return;
    }

    showResultSuccess(data, null, data.preprocessed_pixels || []);
  } catch (err) {
    console.error("Prediction error:", err);
    showResultError("Could not connect to the server. Make sure Flask is running.");
  } finally {
    predictInProgress = false;
    document.getElementById("btnPredict").disabled = false;
  }
}

function resetResult() {
  setResultState("placeholder");
}
function showResultLoading() {
  setResultState("loading");
}
function showResultError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  setResultState("error");
}

function setResultState(state) {
  document.getElementById("resultPlaceholder").classList.add("hidden");
  document.getElementById("resultContent").classList.add("hidden");
  document.getElementById("resultError").classList.add("hidden");
  document.getElementById("resultLoading").classList.add("hidden");

  if (state === "placeholder") document.getElementById("resultPlaceholder").classList.remove("hidden");
  if (state === "content")     document.getElementById("resultContent").classList.remove("hidden");
  if (state === "error")       document.getElementById("resultError").classList.remove("hidden");
  if (state === "loading")     document.getElementById("resultLoading").classList.remove("hidden");
}

function showResultSuccess(data, previewOff, pixels) {
  const digit = data.predicted_digit;
  const scores = data.decision_scores || [];

  document.getElementById("predictedDigit").textContent = digit;
  document.getElementById("predictedMsg").textContent   =
    `The model recognises this as digit ${digit}`;

  // Display the calibrated 8×8 image that the model evaluated
  const preview = document.getElementById("previewCanvas");
  const pCtx    = preview.getContext("2d");
  pCtx.imageSmoothingEnabled = false;
  const evalPixels = data.preprocessed_pixels || pixels;
  drawDigitImageFromFlat(pCtx, evalPixels, preview.width);

  // Scores bar chart
  if (scores.length === 10) {
    renderScoresChart(scores, digit);
  }

  setResultState("content");
}

function renderScoresChart(scores, predictedDigit) {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js is not loaded. Skipping scores chart.");
    return;
  }
  if (scoresChartInstance) {
    scoresChartInstance.destroy();
    scoresChartInstance = null;
  }
  const scoreCtx = document.getElementById("scoresChart");
  const labels = Array.from({ length: 10 }, (_, i) => `Digit ${i}`);
  const colors  = scores.map((_, i) =>
    i === predictedDigit
      ? "rgba(139,92,246,0.85)"
      : "rgba(99,102,241,0.3)"
  );

  scoresChartInstance = new Chart(scoreCtx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Decision Score",
        data: scores,
        backgroundColor: colors,
        borderColor: scores.map((_, i) =>
          i === predictedDigit ? "rgba(139,92,246,1)" : "rgba(99,102,241,0.5)"
        ),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9ca3af", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#9ca3af", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}


/* ═══════════════════════════════════════════════════════════
   5. API DATA LOADERS
   ═══════════════════════════════════════════════════════════ */

// ── 5a. Overview ────────────────────────────────────────────

async function loadOverview() {
  try {
    const res = await fetch("/api/overview");
    const d   = await res.json();

    document.getElementById("ovTotalSamples").textContent = d.n_samples?.toLocaleString()   ?? "—";
    document.getElementById("ovClasses").textContent      = d.n_classes                      ?? "—";
    document.getElementById("ovAccuracy").textContent     = pct(d.accuracy);
    document.getElementById("ovTrain").textContent        = d.n_train?.toLocaleString()      ?? "—";
    document.getElementById("ovTest").textContent         = d.n_test?.toLocaleString()       ?? "—";
    document.getElementById("ovF1").textContent           = pct(d.f1_score);
  } catch (err) {
    console.error("Overview load failed:", err);
  }
}

// ── 5b. Evaluation ──────────────────────────────────────────

let perClassChartInstance   = null;

async function loadEvaluation() {
  try {
    const res = await fetch("/api/evaluation");
    const d   = await res.json();

    document.getElementById("evalAccuracy").textContent  = pct(d.accuracy);
    document.getElementById("evalPrecision").textContent = pct(d.precision);
    document.getElementById("evalRecall").textContent    = pct(d.recall);
    document.getElementById("evalF1").textContent        = pct(d.f1_score);

    renderPerClassChart(d.per_class);
    renderConfusionMatrix(d.confusion_matrix);
  } catch (err) {
    console.error("Evaluation load failed:", err);
    document.getElementById("cmWrapper").innerHTML = '<p style="color:var(--accent-red)">Failed to load evaluation data.</p>';
  }
}

function renderPerClassChart(perClass) {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js is not loaded. Skipping per-class chart.");
    return;
  }
  if (perClassChartInstance) {
    perClassChartInstance.destroy();
    perClassChartInstance = null;
  }
  const ctx2 = document.getElementById("perClassChart");
  const labels = Array.from({ length: 10 }, (_, i) => `Digit ${i}`);

  perClassChartInstance = new Chart(ctx2, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Precision", data: perClass.precision.map(v => +(v * 100).toFixed(2)), backgroundColor: "rgba(139,92,246,0.7)", borderRadius: 4 },
        { label: "Recall",    data: perClass.recall.map(v    => +(v * 100).toFixed(2)), backgroundColor: "rgba(99,102,241,0.7)",  borderRadius: 4 },
        { label: "F1-Score",  data: perClass.f1.map(v        => +(v * 100).toFixed(2)), backgroundColor: "rgba(20,184,166,0.7)",  borderRadius: 4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#9ca3af", font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` } },
      },
      scales: {
        x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: {
          min: 0, max: 100,
          ticks: { color: "#9ca3af", callback: v => v + "%" },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
      },
    },
  });
}

function renderConfusionMatrix(cm) {
  const wrapper = document.getElementById("cmWrapper");
  if (!cm || !Array.isArray(cm)) {
    wrapper.innerHTML = '<p class="cm-loading">No confusion matrix data.</p>';
    return;
  }

  const n = cm.length;
  // Find max value for colour scaling (excluding diagonal)
  let maxOff = 1;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j && cm[i][j] > maxOff) maxOff = cm[i][j];

  let html = '<table class="cm-table"><thead><tr><th>Act \\ Pred</th>';
  for (let j = 0; j < n; j++) html += `<th>${j}</th>`;
  html += "</tr></thead><tbody>";

  for (let i = 0; i < n; i++) {
    html += `<tr><td class="row-label">${i}</td>`;
    for (let j = 0; j < n; j++) {
      const v = cm[i][j];
      let cls;
      if (i === j) {
        cls = "cm-diag";
      } else if (v === 0) {
        cls = "cm-cell-0";
      } else {
        const ratio = v / maxOff;
        if (ratio < 0.25) cls = "cm-cell-lo";
        else if (ratio < 0.6) cls = "cm-cell-mid";
        else cls = "cm-cell-hi";
      }
      html += `<td class="${cls}" title="Actual ${i}, Predicted ${j}: ${v}">${v}</td>`;
    }
    html += "</tr>";
  }

  html += "</tbody></table>";
  wrapper.innerHTML = html;
}

// ── 5c. Dataset ─────────────────────────────────────────────

let classDistChartInstance = null;

async function loadDataset() {
  try {
    const res = await fetch("/api/dataset");
    const d   = await res.json();

    document.getElementById("dsTotalSamples").textContent   = d.n_samples?.toLocaleString() ?? "—";
    document.getElementById("dsImageShape").textContent     = d.image_shape ? `${d.image_shape[0]}×${d.image_shape[1]}` : "—";
    document.getElementById("dsIntensityRange").textContent = d.intensity_range ? `${d.intensity_range[0]}–${d.intensity_range[1]}` : "—";
    document.getElementById("dsClasses").textContent        = d.n_classes ?? "—";

    renderClassDistChart(d.class_distribution);
    renderGallery(d.samples);
  } catch (err) {
    console.error("Dataset load failed:", err);
  }
}

function renderClassDistChart(dist) {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js is not loaded. Skipping class distribution chart.");
    return;
  }
  if (classDistChartInstance) {
    classDistChartInstance.destroy();
    classDistChartInstance = null;
  }
  const ctx3  = document.getElementById("classDistChart");
  const labels = Object.keys(dist).sort((a, b) => +a - +b).map(k => `Digit ${k}`);
  const vals   = labels.map(l => dist[l.replace("Digit ", "")] ?? 0);

  classDistChartInstance = new Chart(ctx3, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Image Count",
        data: vals,
        backgroundColor: "rgba(139,92,246,0.65)",
        borderColor:     "rgba(139,92,246,1)",
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

function renderGallery(samples) {
  const container = document.getElementById("galleryContainer");
  if (!samples || samples.length === 0) {
    container.innerHTML = '<p class="gallery-loading">No sample data.</p>';
    return;
  }

  // Group by label
  const byLabel = {};
  samples.forEach(s => {
    if (!byLabel[s.label]) byLabel[s.label] = [];
    byLabel[s.label].push(s);
  });

  let html = "";
  for (let label = 0; label <= 9; label++) {
    const group = byLabel[label] || [];
    html += `<div class="gallery-row">
      <span class="gallery-row-title">Digit ${label}</span>
      <div class="gallery-images" id="gallery-row-${label}"></div>
    </div>`;
  }
  container.innerHTML = html;

  // Render each image as a canvas
  for (let label = 0; label <= 9; label++) {
    const group = byLabel[label] || [];
    const rowEl = document.getElementById(`gallery-row-${label}`);
    group.forEach(s => {
      const wrap = document.createElement("div");
      wrap.className = "gallery-img-wrap";
      const c = document.createElement("canvas");
      c.className = "gallery-canvas";
      c.width  = 48;
      c.height = 48;
      const gCtx = c.getContext("2d");
      drawDigitImage(gCtx, s.image, 48);
      wrap.appendChild(c);
      rowEl.appendChild(wrap);
    });
  }
}

// ── 5d. Example digits (Recognition page) ──────────────────

async function loadExamples() {
  try {
    const res = await fetch("/api/dataset");
    const d   = await res.json();
    renderExamples(d.samples);
  } catch (err) {
    document.getElementById("examplesGrid").innerHTML = '<span class="examples-loading">Failed to load examples.</span>';
  }
}

function renderExamples(samples) {
  const grid = document.getElementById("examplesGrid");
  if (!samples || samples.length === 0) {
    grid.innerHTML = '<span class="examples-loading">No examples.</span>';
    return;
  }

  // Pick first sample per digit for the grid
  const picked = {};
  samples.forEach(s => {
    if (!picked[s.label]) picked[s.label] = s;
  });

  grid.innerHTML = "";
  for (let i = 0; i <= 9; i++) {
    const s = picked[i];
    if (!s) continue;

    const item = document.createElement("div");
    item.className = "example-item";
    item.title     = `Click to test digit ${i}`;

    const c = document.createElement("canvas");
    c.className = "example-canvas";
    c.width  = 56;
    c.height = 56;
    const eCtx = c.getContext("2d");
    drawDigitImage(eCtx, s.image, 56);

    const label = document.createElement("span");
    label.className   = "example-label";
    label.textContent = `Digit ${i}`;

    item.appendChild(c);
    item.appendChild(label);

    // When clicked, send this sample's flat pixel array to the predict API
    item.addEventListener("click", () => testExampleDigit(s));

    grid.appendChild(item);
  }
}

async function testExampleDigit(sample) {
  if (predictInProgress) return;
  predictInProgress = true;
  document.getElementById("btnPredict").disabled = true;
  showResultLoading();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: sample.flat }),
    });
    const data = await response.json();

    if (!response.ok) {
      showResultError(data.error || `Server error ${response.status}`);
      return;
    }

    // Create an 8×8 preview from the flat pixel data
    const previewOff = document.createElement("canvas");
    previewOff.width  = 8;
    previewOff.height = 8;
    const pCtx = previewOff.getContext("2d");
    drawDigitImageFromFlat(pCtx, sample.flat, 8);

    showResultSuccess(data, previewOff, sample.flat);
  } catch (err) {
    showResultError("Could not connect to the server.");
  } finally {
    predictInProgress = false;
    document.getElementById("btnPredict").disabled = false;
  }
}


/* ═══════════════════════════════════════════════════════════
   6. CANVAS RENDERING HELPERS
   ═══════════════════════════════════════════════════════════ */

/**
 * Draw a digit from its 8×8 two-dimensional array (values 0–16)
 * onto a canvas context, scaled to `size` pixels.
 */
function drawDigitImage(ctx, image8x8, size) {
  const cellSize = size / 8;
  ctx.fillStyle = "#1a1d30";
  ctx.fillRect(0, 0, size, size);

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const val       = image8x8[row][col];
      const intensity = Math.round((val / 16) * 255);
      ctx.fillStyle   = `rgb(${intensity},${intensity},${intensity})`;
      ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    }
  }
}

/**
 * Same as drawDigitImage but from a flat 64-element array.
 */
function drawDigitImageFromFlat(ctx, flat64, size) {
  const cellSize = size / 8;
  ctx.fillStyle = "#1a1d30";
  ctx.fillRect(0, 0, size, size);

  for (let i = 0; i < 64; i++) {
    const row = Math.floor(i / 8);
    const col = i % 8;
    const val       = flat64[i];
    const intensity = Math.round((val / 16) * 255);
    ctx.fillStyle   = `rgb(${intensity},${intensity},${intensity})`;
    ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
  }
}


/* ═══════════════════════════════════════════════════════════
   7. UTILITIES
   ═══════════════════════════════════════════════════════════ */

function pct(val) {
  if (val === undefined || val === null) return "—";
  return (val * 100).toFixed(2) + "%";
}
