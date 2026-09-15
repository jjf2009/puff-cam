"""MediaPipe Hands wrapper exposing pinch point + grip angle."""
import math
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.6, tracking_conf=0.5):
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, rgb_frame, frame_w, frame_h):
        """Returns None or dict: pinch (x,y), angle_deg, hand_size (px)."""
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0].landmark

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
        self._hands.close()
