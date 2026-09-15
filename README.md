# puff-cam

A webcam toy: pinch your thumb and index finger like you're holding a
cigarette, vape, or cigar. Bring it to your mouth and it puffs smoke, sized
and angled to your actual hand.

No image assets — the items and the smoke are both drawn procedurally with
OpenCV, shaded and lit like real cylinders instead of flat cutouts. Hand and
mouth position come from MediaPipe's Tasks API (`HandLandmarker` +
`FaceLandmarker`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

MediaPipe model files (~12 MB) download into `models/` on first run.

## Run

```bash
.venv/bin/python main.py
```

| Key | Action |
| --- | --- |
| `1` | Cigarette |
| `2` | Vape |
| `3` | Cigar |
| `q` / `Esc` | Quit |

## How it works

- [hand_tracker.py](hand_tracker.py): finds your pinch point (midpoint of thumb and index fingertip) and hand angle
- [face_tracker.py](face_tracker.py): finds your mouth center and face width
- [gesture.py](gesture.py): pinch closer to your mouth than `0.35 × face width` counts as smoking
- [sprites.py](sprites.py): procedurally shaded cigarette/vape/cigar sprites, drawn at your pinch point, rotated and scaled to your hand
- [smoke.py](smoke.py): particle system — spawns, drifts, fades, and blurs into a soft smoke/vapor cloud while you're smoking

## Troubleshooting

- **`Could not open webcam`**: your user needs access to `/dev/video*`:
  `sudo usermod -aG video $USER`, then log out and back in.
  The app tries camera indices 0–3 on its own.
- **Black screen**: check the camera's privacy shutter or kill switch, and close
  any other app that's using the camera.
- **Smoke triggers too easily or not at all**: change `SMOKING_RATIO` in [gesture.py](gesture.py).
- **Item looks too big/small**: tweak the `scale` calculation in [main.py](main.py) (based on `hand_size`), or the base `length`/`width` in each `make_*` function in [sprites.py](sprites.py).
