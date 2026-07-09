# Gesture Console

A single-page site that runs your ONNX hand-gesture model live in the browser
(via [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)) and plays
an audio clip whenever it recognizes a gesture with enough confidence.

Everything runs client-side — no backend, no data leaves the browser.

## Folder layout

```
gesture-app/
├── .vscode/        Live Server + task config (VS Code only, safe to ignore otherwise)
├── index.html      the page/UI — usually no need to touch this
├── app.js          all logic + the CONFIG block you'll edit
├── package.json     lets "npm start" / the VS Code task serve the folder
├── model/
│   └── model.onnx  ← put your exported model here
└── audio/
    ├── fist.mp3     ← one clip per gesture label
    ├── peace.mp3
    └── ...
```

## 1. Add your files

- Copy your model into `model/model.onnx` (or update `MODEL_URL` in
  `app.js` if you'd rather name it something else).
- Copy your audio clips into `audio/`.

## 2. Edit the CONFIG block at the top of `app.js`

| Setting | What to set it to |
|---|---|
| `LABELS` | Your class names, **in the exact order your model outputs them** — this is the most common source of "wrong sound plays" bugs |
| `AUDIO_MAP` | Which audio file plays for each label |
| `INPUT_SIZE` | The square size your model expects, e.g. `224` |
| `LAYOUT` | `"NCHW"` (channel-first — typical for PyTorch/ONNX) or `"NHWC"` (channel-last — typical for raw Keras/TF exports) |
| `MEAN` / `STD` | Normalization your model was trained with. If it just expects raw 0–1 pixels, set `MEAN` to `[0,0,0]` and `STD` to `[1,1,1]` |
| `APPLY_SOFTMAX` | Leave `true` unless your model's output already sums to 1 |
| `CONFIDENCE_THRESHOLD` | How sure the model must be before a sound plays (0–1) |
| `COOLDOWN_MS` | Minimum gap between repeat triggers of the same gesture, so holding a pose doesn't spam the audio |
| `INFERENCE_INTERVAL_MS` | How often a frame is classified (lower = more responsive, higher CPU use) |

If you're not sure of your model's exact input requirements, check whatever
script/notebook you used to export it to ONNX — the preprocessing there
(resize size, `/255`, normalization) needs to match what happens in
`frameToTensor()` here.

## 3. Run it

Browsers block `fetch()` of local files opened as `file://`, so the model
won't load if you just double-click `index.html` — it needs to be served.

### Option A — VS Code + Live Server (easiest)

1. Open this folder in VS Code (`File > Open Folder…`).
2. When prompted, install the recommended **Live Server** extension (or grab
   it manually from the Extensions panel — publisher `ritwickdey`).
3. Right-click `index.html` in the file explorer → **Open with Live Server**.
   Your browser opens automatically at `http://127.0.0.1:5500`.
4. Hit **Start scanning** and allow camera access.

Live Server also auto-reloads the page whenever you save changes to `app.js`,
which is handy while you're tuning `CONFIG`.

### Option B — VS Code task (no extension needed)

Press **Ctrl+Shift+B** / **Cmd+Shift+B** (or `Terminal > Run Task… > Serve
gesture console`). This runs `npx serve` in the built-in terminal — the
first run downloads `serve` on the fly, so it needs internet access once.
Then open **http://localhost:8000**.

### Option C — plain terminal

```bash
cd gesture-app
python3 -m http.server 8000
```

Then open **http://localhost:8000** and hit **Start scanning**.

## Troubleshooting

- **"Couldn't load the model"** — usually means you opened the file directly
  instead of through a local server (see above), or `MODEL_URL` doesn't
  match where you put the file.
- **Predictions look random / always the same class** — almost always a
  preprocessing mismatch: wrong `INPUT_SIZE`, `LAYOUT`, or `MEAN`/`STD`. Match
  these to your training/export pipeline exactly.
- **No sound plays** — check `AUDIO_MAP` has an entry for the predicted
  label, and that `CONFIDENCE_THRESHOLD` isn't set higher than the
  confidences your model is actually producing (watch the meter in the UI).
- **Camera permission denied** — the site needs to be served over
  `http://localhost` or `https://`; camera access is blocked on plain `http://`
  for any non-localhost address.

## Deploying

Any static host works (GitHub Pages, Netlify, Vercel, S3 + CloudFront) since
there's no server logic — just upload the four items in the folder layout
above. Make sure it's served over **https** for camera access to work outside
of localhost.
