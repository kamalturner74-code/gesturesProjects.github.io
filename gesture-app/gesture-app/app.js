/* ============================================================
   GESTURE CONSOLE — configuration
   Edit everything in this block to match YOUR model and audio.
   ============================================================ */
const CONFIG = {
  // Path to your exported ONNX model.
  MODEL_URL: "model/model.onnx",

  // Class names, in the exact order your model's output vector uses.
  // e.g. if output index 0 = "fist", index 1 = "peace", etc.
  LABELS: ["fist", "peace", "open_palm", "thumbs_up", "ok"],

  // One audio file per label. Missing entries are silently skipped.
  AUDIO_MAP: {
    fist: "audio/fist.mp3",
    peace: "audio/peace.mp3",
    open_palm: "audio/open_palm.mp3",
    thumbs_up: "audio/thumbs_up.mp3",
    ok: "audio/ok.mp3",
  },

  // Square input size your model expects (e.g. 224 for a 224x224 model).
  INPUT_SIZE: 224,

  // "NCHW" (channels first, typical for PyTorch/ONNX exports) or
  // "NHWC" (channels last, typical for raw TensorFlow/Keras exports).
  LAYOUT: "NCHW",

  // Per-channel normalization applied as (pixel/255 - MEAN) / STD.
  // Defaults below are the common ImageNet stats. Set MEAN to [0,0,0]
  // and STD to [1,1,1] if your model just wants raw 0-1 pixel values.
  MEAN: [0.485, 0.456, 0.406],
  STD: [0.229, 0.224, 0.225],

  // If your model's last layer is raw logits (not already softmaxed),
  // leave this true so confidence values are meaningful probabilities.
  APPLY_SOFTMAX: true,

  // Only acts on a prediction if confidence is above this.
  CONFIDENCE_THRESHOLD: 0.75,

  // Minimum time between two audio triggers, in milliseconds.
  // Prevents the same clip from firing every frame while a gesture is held.
  COOLDOWN_MS: 1800,

  // How often to run inference on the live feed, in milliseconds.
  INFERENCE_INTERVAL_MS: 200,
};
/* ============================================================
   End of config. Nothing below this line needs editing to get
   started — it's the plumbing that wires the config together.
   ============================================================ */

const els = {
  video: document.getElementById("video"),
  placeholder: document.getElementById("placeholder"),
  canvas: document.getElementById("hiddenCanvas"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  gestureReadout: document.getElementById("gestureReadout"),
  confidenceFill: document.getElementById("confidenceFill"),
  confidenceValue: document.getElementById("confidenceValue"),
  log: document.getElementById("log"),
  scanline: document.getElementById("scanline"),
  errorBanner: document.getElementById("errorBanner"),
};

let session = null;
let stream = null;
let inferenceTimer = null;
let audioCache = {};
let lastTriggered = { label: null, at: 0 };
let running = false;

function setStatus(mode) {
  // mode: "idle" | "loading" | "live" | "error"
  els.statusDot.className = "dot" + (mode === "live" ? " live" : mode === "error" ? " err" : "");
  els.statusText.textContent = {
    idle: "IDLE",
    loading: "LOADING…",
    live: "LIVE",
    error: "ERROR",
  }[mode] || mode.toUpperCase();
}

function showError(message) {
  els.errorBanner.textContent = "⚠ " + message;
  els.errorBanner.classList.add("show");
  setStatus("error");
}

function clearError() {
  els.errorBanner.classList.remove("show");
  els.errorBanner.textContent = "";
}

function logDetection(label, confidence) {
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `<span class="label">${label}</span><span class="time">${(confidence * 100).toFixed(0)}% · ${time}</span>`;
  els.log.prepend(entry);
  while (els.log.children.length > 30) {
    els.log.removeChild(els.log.lastChild);
  }
}

function getAudio(label) {
  if (audioCache[label]) return audioCache[label];
  const src = CONFIG.AUDIO_MAP[label];
  if (!src) return null;
  const audio = new Audio(src);
  audioCache[label] = audio;
  return audio;
}

function maybePlayAudio(label, confidence) {
  if (confidence < CONFIG.CONFIDENCE_THRESHOLD) return;
  const now = performance.now();
  const sameLabel = lastTriggered.label === label;
  const withinCooldown = now - lastTriggered.at < CONFIG.COOLDOWN_MS;
  if (sameLabel && withinCooldown) return;

  const audio = getAudio(label);
  if (audio) {
    audio.currentTime = 0;
    audio.play().catch((err) => {
      console.warn("Audio playback blocked or failed:", err);
    });
  }
  lastTriggered = { label, at: now };
  logDetection(label, confidence);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Draws the current video frame into a square, model-sized buffer
// (center-cropped so aspect ratio isn't distorted), then converts
// it into a Float32Array laid out per CONFIG.LAYOUT.
function frameToTensor() {
  const size = CONFIG.INPUT_SIZE;
  const ctx = els.canvas.getContext("2d");
  els.canvas.width = size;
  els.canvas.height = size;

  const vw = els.video.videoWidth;
  const vh = els.video.videoHeight;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;

  ctx.save();
  ctx.translate(size, 0);
  ctx.scale(-1, 1); // mirror to match the on-screen preview
  ctx.drawImage(els.video, sx, sy, side, side, 0, 0, size, size);
  ctx.restore();

  const { data } = ctx.getImageData(0, 0, size, size); // RGBA, 0-255
  const numPixels = size * size;
  const float32 = new Float32Array(3 * numPixels);

  if (CONFIG.LAYOUT === "NCHW") {
    for (let i = 0; i < numPixels; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      float32[i] = (r - CONFIG.MEAN[0]) / CONFIG.STD[0];
      float32[numPixels + i] = (g - CONFIG.MEAN[1]) / CONFIG.STD[1];
      float32[2 * numPixels + i] = (b - CONFIG.MEAN[2]) / CONFIG.STD[2];
    }
  } else {
    // NHWC
    for (let i = 0; i < numPixels; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      float32[i * 3] = (r - CONFIG.MEAN[0]) / CONFIG.STD[0];
      float32[i * 3 + 1] = (g - CONFIG.MEAN[1]) / CONFIG.STD[1];
      float32[i * 3 + 2] = (b - CONFIG.MEAN[2]) / CONFIG.STD[2];
    }
  }

  const dims = CONFIG.LAYOUT === "NCHW" ? [1, 3, size, size] : [1, size, size, 3];
  return new ort.Tensor("float32", float32, dims);
}

async function runInference() {
  if (!session || els.video.readyState < 2) return;
  try {
    const tensor = frameToTensor();
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);
    let output = Array.from(results[outputName].data);

    if (CONFIG.APPLY_SOFTMAX) output = softmax(output);

    let bestIdx = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[bestIdx]) bestIdx = i;
    }
    const label = CONFIG.LABELS[bestIdx] ?? `class_${bestIdx}`;
    const confidence = output[bestIdx];

    els.gestureReadout.innerHTML = label;
    els.confidenceFill.style.width = `${(confidence * 100).toFixed(1)}%`;
    els.confidenceValue.textContent = `confidence: ${(confidence * 100).toFixed(1)}%`;

    maybePlayAudio(label, confidence);
  } catch (err) {
    console.error("Inference error:", err);
    showError("Inference failed — check console for details.");
    stop();
  }
}

async function loadModel() {
  setStatus("loading");
  session = await ort.InferenceSession.create(CONFIG.MODEL_URL, {
    executionProviders: ["wasm"],
  });
}

async function start() {
  clearError();
  els.startBtn.disabled = true;
  try {
    if (!session) await loadModel();

    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    els.video.srcObject = stream;
    await els.video.play();

    els.placeholder.style.display = "none";
    els.scanline.classList.add("active");
    els.stopBtn.disabled = false;
    setStatus("live");
    running = true;

    inferenceTimer = setInterval(runInference, CONFIG.INFERENCE_INTERVAL_MS);
  } catch (err) {
    console.error(err);
    if (err.name === "NotAllowedError") {
      showError("Camera permission denied. Allow camera access and try again.");
    } else if (err.message && err.message.includes("model")) {
      showError("Couldn't load the model. Check MODEL_URL in app.js and that you're serving over http, not file://.");
    } else {
      showError(err.message || "Something went wrong starting the console.");
    }
    els.startBtn.disabled = false;
  }
}

function stop() {
  running = false;
  if (inferenceTimer) clearInterval(inferenceTimer);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  els.placeholder.style.display = "flex";
  els.scanline.classList.remove("active");
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.gestureReadout.innerHTML = '<span class="idle">— no signal —</span>';
  els.confidenceFill.style.width = "0%";
  els.confidenceValue.textContent = "confidence: —";
  setStatus("idle");
}

els.startBtn.addEventListener("click", start);
els.stopBtn.addEventListener("click", stop);
