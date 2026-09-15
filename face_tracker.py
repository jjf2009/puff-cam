"""MediaPipe Tasks FaceLandmarker wrapper exposing mouth center + face-width scale reference."""
import math

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models_setup import ensure_model

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
FACE_LEFT = 234
FACE_RIGHT = 454


class FaceTracker:
    def __init__(self, detection_conf=0.5, tracking_conf=0.5):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model("face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame, frame_w, frame_h, timestamp_ms):
        """Returns None or dict: mouth (x,y), face_width (px)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        def pt(i):
            return (lm[i].x * frame_w, lm[i].y * frame_h)

        top, bottom = pt(MOUTH_TOP), pt(MOUTH_BOTTOM)
        mouth = ((top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2)

        left, right = pt(FACE_LEFT), pt(FACE_RIGHT)
        face_width = math.hypot(right[0] - left[0], right[1] - left[1])

        return {"mouth": mouth, "face_width": face_width}

    def close(self):
        self._landmarker.close()
