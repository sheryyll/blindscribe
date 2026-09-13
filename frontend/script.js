// When this page is served BY the FastAPI backend itself (the recommended
// deployment - see the StaticFiles mount at the bottom of backend/app.py),
// API calls are same-origin, so a relative path ("" + "/predict") is
// correct and there's nothing to configure. The only case that needs an
// explicit host is opening this HTML file directly from disk (file://)
// during local development, where there's no server for the page itself.
const API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "";
const LOW_CONFIDENCE_THRESHOLD = 0.6;
const PLACEHOLDER_TILE_COUNT = 8;
const CANVAS_SIZES = {
  small: [420, 140],
  medium: [640, 200],
  large: [900, 280],
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const trashBtn = document.getElementById("trash-btn");
const undoBtn = document.getElementById("undo-btn");
const redoBtn = document.getElementById("redo-btn");
const sizeSelect = document.getElementById("size-select");
const resultStringEl = document.getElementById("result-string");
const resultCharsEl = document.getElementById("result-chars");
const explainPanel = document.getElementById("explain-panel");
const explainCaption = document.getElementById("explain-caption");
const heatmapCanvas = document.getElementById("heatmap-canvas");
const heatmapCtx = heatmapCanvas.getContext("2d");
const warningEl = document.getElementById("warning");
const overallConfidenceValue = document.getElementById("overall-confidence-value");
const overallConfidenceLabel = document.getElementById("overall-confidence-label");
const thicknessButtons = document.querySelectorAll(".dot-btn");

// Palette constants - must match the CSS custom properties in style.css.
// Kept as plain values here (not read from CSS) because canvas 2D
// rendering needs literal color strings, not var() references.
const CANVAS_BG = "#fbf7ec";   // --cream-2
const INK_COLOR = "#23303a";  // --ink
const GRID_LINE_COLOR = "rgba(147, 184, 201, 0.5)"; // --grid-line (placid blue tint)
const GRID_LINE_SPACING = 40; // px, matches the line-height a writer would expect

let drawing = false;
let hasInk = false;
let lastCharacters = [];
let lastImageDataUrl = null;
let lastImageElement = null;
let currentLineWidth = 2; // thin default, matches the active "Thin" toolbar preset

// Undo/redo: each entry is a snapshot of the canvas pixels PLUS the hasInk
// flag at that point, captured right before a stroke (or a clear) happens.
let historyStack = [];
let redoStack = [];

function setupCanvas() {
  // Background + guide lines are drawn directly on the canvas context
  // (not as a CSS background-image) because the ink fill below is fully
  // opaque and sits on top of any CSS background - a CSS-only approach
  // would make the guide lines invisible the moment the canvas is
  // painted, which was a real bug in the previous version.
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.strokeStyle = GRID_LINE_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 5]);
  for (let y = GRID_LINE_SPACING; y < canvas.height; y += GRID_LINE_SPACING) {
    ctx.beginPath();
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(canvas.width, y + 0.5);
    ctx.stroke();
  }
  ctx.restore();

  ctx.strokeStyle = INK_COLOR;
  ctx.lineWidth = currentLineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
setupCanvas();
renderPlaceholderTiles();

// ---------- Drawing ----------

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = evt.touches ? evt.touches[0] : evt;
  return {
    x: (point.clientX - rect.left) * scaleX,
    y: (point.clientY - rect.top) * scaleY,
  };
}

function captureState() {
  return { imageData: ctx.getImageData(0, 0, canvas.width, canvas.height), hasInk };
}

function pushHistory() {
  historyStack.push(captureState());
  if (historyStack.length > 50) historyStack.shift();
  redoStack = [];
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  undoBtn.disabled = historyStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
}

function undo() {
  if (!historyStack.length) return;
  redoStack.push(captureState());
  const prev = historyStack.pop();
  ctx.putImageData(prev.imageData, 0, 0);
  hasInk = prev.hasInk;
  updateUndoRedoButtons();
}

function redo() {
  if (!redoStack.length) return;
  historyStack.push(captureState());
  const next = redoStack.pop();
  ctx.putImageData(next.imageData, 0, 0);
  hasInk = next.hasInk;
  updateUndoRedoButtons();
}

function startDraw(evt) {
  evt.preventDefault();
  pushHistory(); // snapshot BEFORE this stroke, so undo reverts to just before it
  drawing = true;
  hasInk = true;
  const { x, y } = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const { x, y } = getPos(evt);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", endDraw);

canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", endDraw);

// ---------- Toolbar: thickness presets ----------

thicknessButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    thicknessButtons.forEach((b) => b.classList.remove("dot-btn--active"));
    btn.classList.add("dot-btn--active");
    currentLineWidth = Number(btn.dataset.thickness);
    ctx.lineWidth = currentLineWidth;
  });
});

// ---------- Toolbar: undo / redo / clear ----------

undoBtn.addEventListener("click", undo);
redoBtn.addEventListener("click", redo);

function clearCanvasAction() {
  pushHistory(); // clearing is itself undoable
  setupCanvas();
  hasInk = false;
  resetResultDisplay();
}

function resetResultDisplay() {
  warningEl.hidden = true;
  explainPanel.hidden = true;
  resultStringEl.textContent = "Your writing will appear here…";
  resultStringEl.classList.add("recognized-word--empty");
  overallConfidenceValue.textContent = "–";
  overallConfidenceLabel.textContent = "";
  renderPlaceholderTiles();
}

trashBtn.addEventListener("click", clearCanvasAction);
clearBtn.addEventListener("click", clearCanvasAction);

// ---------- Canvas size ----------

sizeSelect.addEventListener("change", () => {
  const [w, h] = CANVAS_SIZES[sizeSelect.value];
  canvas.width = w;
  canvas.height = h;
  canvas.style.aspectRatio = `${w} / ${h}`;
  historyStack = [];
  redoStack = [];
  hasInk = false;
  setupCanvas();
  updateUndoRedoButtons();
  resetResultDisplay();
});

// ---------- Predict ----------

predictBtn.addEventListener("click", async () => {
  if (!hasInk) {
    showWarning("Write something on the canvas first.");
    return;
  }
  warningEl.hidden = true;
  explainPanel.hidden = true;
  predictBtn.disabled = true;
  const originalLabel = predictBtn.innerHTML;
  predictBtn.textContent = "Reading…";

  try {
    const dataUrl = canvas.toDataURL("image/png");
    lastImageDataUrl = dataUrl;

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: dataUrl }),
    });

    if (!response.ok) {
      throw new Error(`Server responded ${response.status}`);
    }

    const data = await response.json();

    if (data.warning) {
      showWarning(data.warning);
      renderPlaceholderTiles();
      return;
    }

    await renderResult(data);
  } catch (err) {
    showWarning(`Could not reach the recognizer: ${err.message}. Is the backend running on ${API_BASE}?`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = originalLabel;
  }
});

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function showWarning(message) {
  warningEl.textContent = message;
  warningEl.hidden = false;
}

// ---------- Rendering results ----------

function renderPlaceholderTiles() {
  resultCharsEl.innerHTML = "";
  for (let i = 0; i < PLACEHOLDER_TILE_COUNT; i++) {
    const tile = document.createElement("div");
    tile.className = "char-tile char-tile--placeholder";
    tile.innerHTML = `<div class="char-tile__letter">–</div><div class="char-tile__confidence">–%</div>`;
    resultCharsEl.appendChild(tile);
  }
}

function overallConfidenceLabelFor(avg) {
  if (avg >= 0.85) return "high";
  if (avg >= 0.6) return "medium";
  return "low";
}

async function renderResult(data) {
  lastCharacters = data.characters;
  lastImageElement = await loadImage(lastImageDataUrl);

  resultStringEl.textContent = data.predicted_string || "(nothing recognized)";
  resultStringEl.classList.remove("recognized-word--empty");
  resultCharsEl.innerHTML = "";

  if (data.characters.length > 0) {
    const avg = data.characters.reduce((sum, c) => sum + c.confidence, 0) / data.characters.length;
    overallConfidenceValue.textContent = `${Math.round(avg * 100)}%`;
    overallConfidenceLabel.textContent = `(${overallConfidenceLabelFor(avg)})`;
  }

  data.characters.forEach((charResult, index) => {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "char-tile";
    if (charResult.confidence < LOW_CONFIDENCE_THRESHOLD) {
      tile.classList.add("char-tile--low");
    }
    tile.dataset.index = index;

    const cropCanvas = document.createElement("canvas");
    cropCanvas.className = "char-tile__crop";
    cropCanvas.width = 32;
    cropCanvas.height = 32;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.fillStyle = "#ffffff";
    cropCtx.fillRect(0, 0, 32, 32);

    const box = charResult.bbox;
    const margin = 4;
    const sx = Math.max(0, box.x0 - margin);
    const sy = Math.max(0, box.y0 - margin);
    const sw = Math.min(lastImageElement.width - sx, box.x1 - box.x0 + margin * 2);
    const sh = Math.min(lastImageElement.height - sy, box.y1 - box.y0 + margin * 2);
    if (sw > 0 && sh > 0) {
      cropCtx.drawImage(lastImageElement, sx, sy, sw, sh, 0, 0, 32, 32);
    }

    const letter = document.createElement("div");
    letter.className = "char-tile__letter";
    letter.textContent = charResult.char;

    const confidence = document.createElement("div");
    confidence.className = "char-tile__confidence";
    confidence.textContent = `${Math.round(charResult.confidence * 100)}%`;

    tile.appendChild(cropCanvas);
    tile.appendChild(letter);
    tile.appendChild(confidence);

    tile.addEventListener("click", () => selectCharacter(index, tile));

    resultCharsEl.appendChild(tile);
  });
}

async function selectCharacter(index, tileEl) {
  document.querySelectorAll(".char-tile").forEach((el) => el.classList.remove("char-tile--selected"));
  tileEl.classList.add("char-tile--selected");

  const charResult = lastCharacters[index];

  try {
    const response = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: lastImageDataUrl, char_index: index }),
    });
    if (!response.ok) throw new Error(`Server responded ${response.status}`);
    const data = await response.json();
    drawHeatmap(data.heatmap, charResult.bbox);
    explainCaption.textContent =
      `The model read this as "${charResult.char}" with ${Math.round(charResult.confidence * 100)}% confidence. ` +
      `The letter is shown below with a heat overlay: brighter red marks the pixels that most influenced ` +
      `that guess, unshaded areas had little to no influence.`;
    explainPanel.hidden = false;
  } catch (err) {
    console.warn("Could not load explanation:", err);
  }
}

function drawHeatmap(heatmap, bbox) {
  const size = heatmap.length;
  const canvasSize = heatmapCanvas.width;
  const margin = 4;

  heatmapCtx.fillStyle = "#ffffff";
  heatmapCtx.fillRect(0, 0, canvasSize, canvasSize);

  const sx = Math.max(0, bbox.x0 - margin);
  const sy = Math.max(0, bbox.y0 - margin);
  const sw = Math.min(lastImageElement.width - sx, bbox.x1 - bbox.x0 + margin * 2);
  const sh = Math.min(lastImageElement.height - sy, bbox.y1 - bbox.y0 + margin * 2);
  if (sw > 0 && sh > 0) {
    heatmapCtx.drawImage(lastImageElement, sx, sy, sw, sh, 0, 0, canvasSize, canvasSize);
  }

  const cell = canvasSize / size;
  const ATTENTION_FLOOR = 0.15;
  const MAX_OVERLAY_OPACITY = 0.75;
  const FLAG_RGB = "122, 33, 64"; // must match --merlot in style.css

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const value = heatmap[y][x];
      if (value < ATTENTION_FLOOR) continue;
      const alpha = Math.min(MAX_OVERLAY_OPACITY, value);
      heatmapCtx.fillStyle = `rgba(${FLAG_RGB}, ${alpha})`;
      heatmapCtx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
}
