"""Virtual Cigarette/Vape/Cigar: webcam overlay driven by hand + face tracking.

Controls:
  1 / 2 / 3  -> switch item (cigarette / vape / cigar)
  q / Esc    -> quit
"""
import time

import cv2
import numpy as np

from hand_tracker import HandTracker
from face_tracker import FaceTracker
import gesture
from smoke import SmokeSystem
from sprites import build_sprites, overlay_transparent, tip_position, ITEM_ORDER

KEY_TO_ITEM = {ord("1"): ITEM_ORDER[0], ord("2"): ITEM_ORDER[1], ord("3"): ITEM_ORDER[2]}


def open_camera(max_index=4):
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def main():
    cap = open_camera()
    if cap is None:
        raise RuntimeError("Could not open webcam (tried indices 0-3)")

    hand_tracker = HandTracker()
    face_tracker = FaceTracker()
    smoke = SmokeSystem()
    sprites = build_sprites()

    current_item = ITEM_ORDER[0]
    start = time.monotonic()
    last_ts = -1
    frame_idx = 0
    face_info = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # VIDEO running mode requires strictly increasing timestamps.
            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts

            hand_info = hand_tracker.process(rgb, w, h, ts)
            # Face barely moves between frames; halving its runs buys back ~10 ms/frame.
            if frame_idx % 2 == 0:
                face_info = face_tracker.process(rgb, w, h, ts)
            frame_idx += 1
            state = gesture.classify(hand_info, face_info)

            spawn_point = None
            if hand_info is not None:
                sprite_img, tip_offset = sprites[current_item]
                scale = max(0.5, min(2.0, hand_info["hand_size"] / 60))
                px, py = hand_info["pinch"]
                angle = -hand_info["angle_deg"]
                overlay_transparent(frame, sprite_img, px, py, angle, scale)
                spawn_point = tip_position(sprite_img, tip_offset, px, py, angle, scale)

            smoke.update(spawn_point, current_item, active=(state == gesture.SMOKING))
            smoke.render(frame)

            label = f"item: {current_item}  state: {state}"
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "1/2/3 switch item  q quit", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Virtual Cigarette", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in KEY_TO_ITEM:
                current_item = KEY_TO_ITEM[key]
    finally:
        cap.release()
        hand_tracker.close()
        face_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
