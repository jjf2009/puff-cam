"""MediaPipe Face Mesh wrapper exposing mouth center + face-width scale reference."""
import math
import mediapipe as mp

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
FACE_LEFT = 234
FACE_RIGHT = 454


class FaceTracker:
    def __init__(self, detection_conf=0.6, tracking_conf=0.5):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, rgb_frame, frame_w, frame_h):
        """Returns None or dict: mouth (x,y), face_width (px)."""
        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        def pt(i):
            return (lm[i].x * frame_w, lm[i].y * frame_h)

        top, bottom = pt(MOUTH_TOP), pt(MOUTH_BOTTOM)
        mouth = ((top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2)

        left, right = pt(FACE_LEFT), pt(FACE_RIGHT)
        face_width = math.hypot(right[0] - left[0], right[1] - left[1])

        return {"mouth": mouth, "face_width": face_width}

    def close(self):
        self._mesh.close()
