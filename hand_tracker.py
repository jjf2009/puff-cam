"""MediaPipe Tasks HandLandmarker wrapper exposing pinch point + grip angle."""
import math

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models_setup import ensure_model


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.5, tracking_conf=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame, frame_w, frame_h, timestamp_ms):
        """Returns None or dict: pinch (x,y), angle_deg, hand_size (px)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]

        def pt(i):
            return (lm[i].x * frame_w, lm[i].y * frame_h)

        thumb_tip = pt(4)
        index_tip = pt(8)
        wrist = pt(0)
        index_mcp = pt(5)

        pinch_x = (thumb_tip[0] + index_tip[0]) / 2
        pinch_y = (thumb_tip[1] + index_tip[1]) / 2

        angle = math.degrees(math.atan2(index_mcp[1] - wrist[1], index_mcp[0] - wrist[0]))
        hand_size = math.hypot(index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])

        return {
            "pinch": (pinch_x, pinch_y),
            "angle_deg": angle,
            "hand_size": hand_size,
        }

    def close(self):
        self._landmarker.close()
