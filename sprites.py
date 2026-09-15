"""Procedural BGRA sprites for cigarette/vape/cigar and the transparent overlay blit helper."""
import cv2
import numpy as np

ITEM_CIGARETTE = "cigarette"
ITEM_VAPE = "vape"
ITEM_CIGAR = "cigar"

ITEM_ORDER = [ITEM_CIGARETTE, ITEM_VAPE, ITEM_CIGAR]


def _blank(w, h):
    return np.zeros((h, w, 4), dtype=np.uint8)


def make_cigarette(length=110, width=14):
    img = _blank(length, width * 2)
    cy = width
    body_len = int(length * 0.78)
    cv2.rectangle(img, (0, cy - width // 2), (body_len, cy + width // 2), (235, 235, 245, 255), -1)
    cv2.rectangle(img, (0, cy - width // 2), (int(body_len * 0.18), cy + width // 2), (150, 190, 210, 255), -1)
    cv2.rectangle(img, (body_len, cy - width // 2), (length, cy + width // 2), (40, 90, 200, 255), -1)
    return img, (length - 4, cy)


def make_vape(length=95, width=20):
    img = _blank(length, width * 2)
    cy = width
    cv2.rectangle(img, (0, cy - width // 2), (int(length * 0.75), cy + width // 2), (60, 45, 30, 255), -1)
    cv2.rectangle(img, (int(length * 0.75), cy - width // 3), (length, cy + width // 3), (20, 20, 20, 255), -1)
    cv2.circle(img, (int(length * 0.15), cy), 3, (0, 200, 255, 255), -1)
    return img, (length - 3, cy)


def make_cigar(length=120, width=22):
    img = _blank(length, width * 2)
    cy = width
    for x in range(length):
        t = x / length
        w = int(width * (0.55 + 0.45 * np.sin(min(t * 1.6, 1.0) * np.pi / 2)))
        color = (30, 70, 110, 255) if x < length * 0.9 else (60, 90, 160, 255)
        cv2.line(img, (x, cy - w // 2), (x, cy + w // 2), color, 1)
    return img, (length - 3, cy)


_BUILDERS = {
    ITEM_CIGARETTE: make_cigarette,
    ITEM_VAPE: make_vape,
    ITEM_CIGAR: make_cigar,
}


def build_sprites():
    """Returns {item_name: (bgra_image, (tip_x, tip_y))} with the tip = lit/mouthpiece end at x=0."""
    sprites = {}
    for name, builder in _BUILDERS.items():
        img, base_tip = builder()
        img = np.flip(img, axis=1).copy()
        h, w = img.shape[:2]
        tip = (0, base_tip[1])
        sprites[name] = (img, tip)
    return sprites


def overlay_transparent(frame, sprite_bgra, center_x, center_y, angle_deg, scale=1.0):
    """Rotate/scale sprite_bgra around its own center and alpha-blend onto frame at (center_x, center_y)."""
    h, w = sprite_bgra.shape[:2]
    if h == 0 or w == 0 or scale <= 0:
        return
    diag = int(np.ceil(np.hypot(w, h) * scale)) + 4
    canvas = np.zeros((diag, diag, 4), dtype=np.uint8)
    ox, oy = (diag - w) // 2, (diag - h) // 2
    canvas[oy:oy + h, ox:ox + w] = sprite_bgra
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle_deg, scale)
    rotated = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    x0 = int(center_x - diag / 2)
    y0 = int(center_y - diag / 2)
    fh, fw = frame.shape[:2]

    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = diag - max(0, (x0 + diag) - fw), diag - max(0, (y0 + diag) - fh)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

    if sx1 <= sx0 or sy1 <= sy0:
        return

    roi = frame[dy0:dy1, dx0:dx1]
    patch = rotated[sy0:sy1, sx0:sx1]
    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    roi[:] = (roi.astype(np.float32) * (1 - alpha) + patch[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
