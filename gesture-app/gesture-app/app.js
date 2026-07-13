// ---- Gesture -> chord mapping, in model output order (index 0..17) ----
const GESTURES = [
  "peace", "palm", "reverse peace", "call", "like", "dislike",
  "silent", "fist", "one", "three", "four", "three2",
  "mute", "stop", "stop-inverted", "ok", "two-up", "two-up inverted"
];

const CHORD_MAP = {
  "peace": "CM", "palm": "GM", "reverse peace": "DM", "call": "AM", "like": "EM",
  "dislike": "Amin", "silent": "Cmin", "fist": "Fmin",
  "one": "FM", "three": "BM", "four": "DsM", "three2": "FsM",
  "mute": "Dmin", "stop": "Gmin", "stop-inverted": "Bmin",
  "ok": "AsM", "two-up": "CsM", "two-up inverted": "GsM"
};

function chordLabel(code) {
  const isMinor = code.endsWith("min");
  let root = isMinor ? code.slice(0, -3) : code.slice(0, -1);
  if (root.endsWith("s")) root = root.slice(0, -1) + "#";
  return { root, quality: isMinor ? "minor" : "major" };
}

const wheelGrid = document.getElementById('wheelGrid');
const pegEls = {};
GESTURES.forEach(g => {
  const code = CHORD_MAP[g];
  const { root, quality } = chordLabel(code);
  const peg = document.createElement('div');
  peg.className = 'peg';
  peg.innerHTML = `<div class="g">${g}</div><div class="c">${root}${quality === 'minor' ? 'm' : ''}</div>`;
  wheelGrid.appendChild(peg);
  pegEls[g] = peg;
});

const audioCache = {};
function createAudio(code) {
  const mp3 = `audio/${encodeURIComponent(code)}.mp3`;
  const wav = `audio/${encodeURIComponent(code)}.wav`;
  const audio = new Audio(mp3);
  audio.addEventListener('error', () => {
    if (audio.src.endsWith('.mp3')) {
      audio.src = wav;
      audio.load();
    }
  }, { once: true });
  return audio;
}

function playChord(gesture) {
  const code = CHORD_MAP[gesture];
  if (!code) return;
  if (!audioCache[code]) {
    audioCache[code] = createAudio(code);
  }
  const audio = audioCache[code];
  audio.currentTime = 0;
  audio.play().catch(() => {});
}

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('startBtn');
const statusLine = document.getElementById('statusLine');
const modelBanner = document.getElementById('modelBanner');
const currentGesture = document.getElementById('currentGesture');
const currentChord = document.getElementById('currentChord');
const confFill = document.getElementById('confFill');

const INPUT_SIZE = 240;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;
let running = false;
let lastGesture = null;
let stableCount = 0;
let lastPlayedAt = 0;
const STABLE_FRAMES_NEEDED = 3;
const RETRIGGER_COOLDOWN_MS = 900;
const INFERENCE_INTERVAL_MS = 220;

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

    const modelUrl = new URL(
      'https://github.com/kamalturner74-code/gesturesProjects.github.io/raw/refs/heads/main/gesture-app/gesture-app/model/model%202.0-20260713T195603Z-2-001/model%202.0/efficientnet_b0_phase2_best.onnx',
      window.location.href
    );

    console.log('Loading model from', modelUrl.href);
    session = await ort.InferenceSession.create(modelUrl.href, { executionProviders: ['wasm'] });
    statusLine.textContent = 'model loaded';
    modelBanner.classList.remove('show');
  } catch (err) {
    statusLine.textContent = 'model failed to load';
    modelBanner.classList.add('show');
    modelBanner.innerHTML =
      `Model didn't load: <code>${(err && err.message) || err}</code><br><br>` +
      `Ensure the model file exists at <code>${modelUrl.href}</code> and that you're serving this site over HTTP (not opening the file directly).`;
    console.error(err);
  }
}

async function startCamera() {
  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 480 } });
    video.srcObject = stream;
    await video.play();
    overlay.width = 480;
    overlay.height = 480;
    running = true;
    startBtn.textContent = 'Camera running';
    statusLine.textContent = session ? 'reading gestures…' : 'model not ready';
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    startBtn.textContent = 'Start camera';
    statusLine.textContent = 'camera permission denied';
    console.error(err);
  }
}

const workCanvas = document.createElement('canvas');
workCanvas.width = INPUT_SIZE;
workCanvas.height = INPUT_SIZE;
const workCtx = workCanvas.getContext('2d', { willReadFrequently: true });

function preprocessFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;
  workCtx.drawImage(video, sx, sy, side, side, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = workCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const plane = INPUT_SIZE * INPUT_SIZE;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = (r - MEAN[0]) / STD[0];
    chw[plane + i] = (g - MEAN[1]) / STD[1];
    chw[plane * 2 + i] = (b - MEAN[2]) / STD[2];
  }
  return chw;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function runInference() {
  const chw = preprocessFrame();
  if (!chw || !session) return;
  const tensor = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const results = await session.run({ input: tensor });
  const outputName = session.outputNames[0];
  const logits = Array.from(results[outputName].data);
  const probs = softmax(logits);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }
  const gesture = GESTURES[bestIdx];
  const conf = probs[bestIdx];

  currentGesture.textContent = gesture;
  const code = CHORD_MAP[gesture];
  const { root, quality } = chordLabel(code);
  currentChord.textContent = `plays ${root}${quality === 'minor' ? ' minor' : ' major'}`;
  confFill.style.width = `${Math.round(conf * 100)}%`;

  Object.values(pegEls).forEach(p => p.classList.remove('active'));
  if (pegEls[gesture]) pegEls[gesture].classList.add('active');

  if (gesture === lastGesture) {
    stableCount++;
  } else {
    stableCount = 1;
    lastGesture = gesture;
  }

  const now = performance.now();
  if (gesture !== 'silent' && stableCount === STABLE_FRAMES_NEEDED && (now - lastPlayedAt) > RETRIGGER_COOLDOWN_MS) {
    playChord(gesture);
    lastPlayedAt = now;
  }
}

let lastInferenceAt = 0;

function loop() {
  if (!running) return;
  const now = performance.now();
  if (session && now - lastInferenceAt > INFERENCE_INTERVAL_MS) {
    lastInferenceAt = now;
    runInference().catch(err => console.error(err));
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener('click', startCamera);
loadModel();
