# Virtual-Cigarette

Webcam toy: hold a virtual cigarette, vape, or cigar between your thumb and index
finger. Bring it to your mouth and it puffs smoke.

Built with OpenCV + MediaPipe (hand and face landmarks).

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

- `hand_tracker.py`: finds your pinch point (between thumb and index tip) and hand angle
- `face_tracker.py`: finds your mouth and face width
- `gesture.py`: pinch closer to mouth than 0.35 × face width means you're smoking
- `sprites.py`: draws the item at your pinch, rotated to match your hand
- `smoke.py`: particle smoke that comes off the tip while you're smoking

## Troubleshooting

- **`Could not open webcam`**: your user needs access to `/dev/video*`:
  `sudo usermod -aG video $USER`, then log out and back in.
  The app tries camera indices 0–3 on its own.
- **Black screen**: check the camera's privacy shutter or kill switch, and close
  any other app that's using the camera.
- **Smoke triggers too easily or not at all**: change `SMOKING_RATIO` in `gesture.py`.
