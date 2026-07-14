const STD = [0.229, 0.224, 0.225];

// ---- MODEL CONFIGURATION ----
// Prefer a TensorFlow.js GraphModel (`model.json`) if available. Fallback to ONNX runtime otherwise.
const TF_MODEL_PATH = 'https://raw.githubusercontent.com/kamalturner74-code/gesturesProjects.github.io/0b089a547bcef294a6739f088cf6fa959669063f/gesture-app/gesture-app/model/model%202.0-20260713T195603Z-2-001/model%202.0/model.json';
const MODEL_PATH = 'https://raw.githubusercontent.com/kamalturner74-code/gesturesProjects.github.io/0b089a547bcef294a6739f088cf6fa959669063f/gesture-app/gesture-app/model/model%202.0-20260713T195603Z-2-001/model%202.0/efficientnet_b0_phase2_best.onnx';
const MODEL_DATA_PATH = 'https://raw.githubusercontent.com/kamalturner74-code/gesturesProjects.github.io/0b089a547bcef294a6739f088cf6fa959669063f/gesture-app/gesture-app/model/model%202.0-20260713T195603Z-2-001/model%202.0/efficientnet_b0_phase2_best.onnx.data';

let session = null; // onnxruntime session
let tfModel = null; // TensorFlow.js model
let running = false;
let lastGesture = null;
let stableCount = 0;
const INFERENCE_INTERVAL_MS = 220;

async function loadModel() {
  try {
    // Try TensorFlow.js model first (expects a model.json). If not present, fall back to ONNX.
    console.log('Checking for TF model at', TF_MODEL_PATH);
    const tfResp = await fetch(TF_MODEL_PATH, { method: 'HEAD' });
    if (tfResp.ok) {
      console.log('TensorFlow.js model found — loading tfjs and model');
      // dynamic import of tfjs from CDN
      await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
      // global `tf` should now be available
      tfModel = await tf.loadGraphModel(TF_MODEL_PATH);
      statusLine.textContent = 'TF model loaded';
      modelBanner.classList.remove('show');
      return;
    }

    // Fallback: attempt to load ONNX runtime model (existing approach)
    console.log('TF model not found; falling back to ONNX runtime');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

    console.log('Loading ONNX model from GitHub');
    console.log('Fetching:', MODEL_PATH);
    console.log('Fetching:', MODEL_DATA_PATH);

    const modelBytes = await (await fetch(MODEL_PATH)).arrayBuffer();
    const dataBytes = await (await fetch(MODEL_DATA_PATH)).arrayBuffer();

    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      externalData: [{ path: 'efficientnet_b0_phase2_best.onnx.data', data: dataBytes }]
    });

    statusLine.textContent = 'ONNX model loaded';
    modelBanner.classList.remove('show');
  } catch (err) {
    statusLine.textContent = 'model failed to load';
    console.error('Model load error:', err);
  }
}

function preprocessFrame() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;
  workCtx.drawImage(video, sx, sy, side, side, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = workCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const plane = INPUT_SIZE * INPUT_SIZE;
  const chw = new Float32Array(3 * plane);
  // for TF input (NHWC), we prepare a float32 array in row-major order
  const nhwc = new Float32Array(plane * 3);
  let p = 0;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    // CHW
    chw[i] = (r - MEAN[0]) / STD[0];
    chw[plane + i] = (g - MEAN[1]) / STD[1];
    chw[plane * 2 + i] = (b - MEAN[2]) / STD[2];
    // NHWC (r,g,b) per pixel
    nhwc[p++] = (r - MEAN[0]) / STD[0];
    nhwc[p++] = (g - MEAN[1]) / STD[1];
    nhwc[p++] = (b - MEAN[2]) / STD[2];
  }
  return { chw, nhwc };
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map(x => x / sum);
}

async function runInference() {
  const frames = preprocessFrame();
  if (!frames) return;
  let logits = null;

  if (tfModel) {
    // TensorFlow.js path (expects NHWC)
    await tf.ready();
    const inputTensor = tf.tensor4d(frames.nhwc, [1, INPUT_SIZE, INPUT_SIZE, 3], 'float32');
    try {
      // executeAsync for GraphModel; predict may also work for LayersModel
      let out = null;
      if (tfModel.executeAsync) {
        out = await tfModel.executeAsync(inputTensor);
      } else if (tfModel.predict) {
        out = tfModel.predict(inputTensor);
      } else {
        throw new Error('Unsupported TF model type');
      }
      // out may be a tensor or an array of tensors
      let outTensor = Array.isArray(out) ? out[0] : out;
      const data = await outTensor.data();
      logits = Array.from(data);
      // dispose tensors if necessary
      if (Array.isArray(out)) out.forEach(t => t.dispose && t.dispose());
      inputTensor.dispose();
    } catch (err) {
      console.error('TF inference error', err);
    }
  } else if (session) {
    // ONNX path
    const tensor = new ort.Tensor('float32', frames.chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const results = await session.run({ input: tensor });
    const outputName = session.outputNames[0];
    logits = Array.from(results[outputName].data);
  } else {
    return;
  }
  const probs = softmax(logits);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  const gesture = GESTURES[bestIdx];
  const confidence = (probs[bestIdx] * 100).toFixed(1);

  if (gesture === lastGesture) {
    stableCount++;
    if (stableCount > STABILITY_THRESHOLD) {
      displayResult(gesture, confidence, probs);
    }
  } else {
    lastGesture = gesture;
    stableCount = 1;
  }
}