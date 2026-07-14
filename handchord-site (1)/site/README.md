# Handchord — hand gesture → major chord

A static, client-side site that runs your EfficientNet-B0 ONNX model in the browser
(via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)) to recognize a
hand gesture from the webcam and play the matching major chord.

Nothing is uploaded anywhere — the model, inference, and audio all run on-device.

## File layout

```
index.html
model/
  efficientnet_b0_phase2_best.onnx
  efficientnet_b0_phase2_best.onnx.data
chords/
  C.mp3  Cs.mp3  D.mp3  Ds.mp3  E.mp3  F.mp3  Fs.mp3  G.mp3  Gs.mp3  A.mp3  As.mp3  B.mp3
```
(`s` = sharp, e.g. `Cs.mp3` is C♯ major. Renamed from your originals so `#` never
has to appear in a URL.)

## Deploy to GitHub Pages

1. Create a new GitHub repo (or use an existing one).
2. Copy everything in this folder into the repo root (keep the `model/` and `chords/`
   folders as-is).
3. Commit and push.
4. In the repo, go to **Settings → Pages**, set **Source** to your default branch
   (root), and save.
5. Wait a minute, then open the URL GitHub gives you. Allow camera access when prompted.

No build step, no dependencies to install — it's plain HTML/JS plus the
`onnxruntime-web` script loaded from a CDN.

## How it works

1. The 18-class gesture model's output indices are assumed to follow **alphabetical
   order** of the class names (this is how PyTorch's `ImageFolder` assigns labels by
   default, which is the standard way this kind of classifier gets trained):
   `call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock,
   stop, stop_inverted, three, three2, two_up, two_up_inverted`.
2. Each video frame is center-cropped to a square, resized to 224×224, and normalized
   with standard ImageNet mean/std (`[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`) —
   the normalization EfficientNet-B0 transfer-learning models are almost always
   trained with.
3. 12 of the 18 gestures are mapped to the 12 major notes (the other 6 gestures have
   no chord and are shown as unrecognized). This mapping lives at the top of
   `index.html` in `GESTURE_TO_NOTE` — edit it any time:

   | Gesture | Chord | Gesture | Chord |
   |---|---|---|---|
   | one | C | like | G |
   | two_up | C♯ | dislike | G♯ |
   | three | D | rock | A |
   | four | D♯ | call | A♯ |
   | palm | E | peace | B |
   | fist | F | | |
   | ok | F♯ | | |

   Unused gestures: `mute`, `stop`, `stop_inverted`, `three2`, `two_up_inverted`,
   `peace_inverted`.
4. A prediction has to (a) clear a 55% confidence threshold and (b) repeat for 3
   consecutive frames (~1 second) before it's accepted — this keeps a chord from
   firing on a single blurry frame while your hand is moving between shapes.

## If predictions look wrong

Two assumptions above are the most likely source of trouble if gestures get
misclassified once it's live:

- **Class order** — if your training notebook printed something like
  `train_dataset.classes` or saved a `class_to_idx` dict, check it against the
  `CLASS_NAMES` array at the top of the script in `index.html` and fix the order
  if it differs.
- **Normalization** — if it still misfires after that, check what `transforms.Normalize(...)`
  values you used during training and update `IMAGENET_MEAN` / `IMAGENET_STD` in
  `index.html` to match.

Everything else (the mapping table, thresholds, timing) is also just plain
constants near the top of the `<script>` block in `index.html`.
