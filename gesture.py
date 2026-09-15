"""Hand-to-mouth distance threshold -> gesture state."""
import math

IDLE = "idle"
HOLDING = "holding"
SMOKING = "smoking"

SMOKING_RATIO = 0.35  # distance(pinch, mouth) < ratio * face_width => smoking


def classify(hand_info, face_info):
    if hand_info is None:
        return IDLE

    if face_info is None:
        return HOLDING

    px, py = hand_info["pinch"]
    mx, my = face_info["mouth"]
    dist = math.hypot(px - mx, py - my)
    threshold = SMOKING_RATIO * face_info["face_width"]

    return SMOKING if dist < threshold else HOLDING
