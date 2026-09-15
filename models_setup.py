"""Locates the MediaPipe .task model bundles, downloading them on first run."""
import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def ensure_model(filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(MODEL_URLS[filename], path)
    return path
