"""Procedural BGRA sprites for cigarette/vape/cigar and the transparent overlay blit helper."""
import cv2
import numpy as np

ITEM_CIGARETTE = "cigarette"
ITEM_VAPE = "vape"
ITEM_CIGAR = "cigar"

ITEM_ORDER = [ITEM_CIGARETTE, ITEM_VAPE, ITEM_CIGAR]


def _blank(w, h):
    return np.zeros((h, w, 4), dtype=np.uint8)


def _cylinder_shade(base_bgr, half_w, highlight_strength=0.55, shadow_strength=0.45):
    """1D shading profile across a cylinder's cross-section: dark rim -> bright
    highlight (offset toward top, like a real light source) -> darker underside.
    Returns an array of (b,g,r) floats indexed by row offset from center, -half_w..half_w.
    """
    rows = np.arange(-half_w, half_w + 1, dtype=np.float32)
    n = rows / max(half_w, 1)
    # highlight band sits above center (toward -1), shadow builds up toward +1 (underside)
    highlight = np.exp(-((n + 0.35) ** 2) / 0.10) * highlight_strength
    shadow = np.clip(n, 0, 1) ** 1.5 * shadow_strength
    rim = (np.abs(n) ** 3) * 0.35
    factor = 1.0 + highlight - shadow - rim
    factor = np.clip(factor, 0.15, 1.5)
    base = np.array(base_bgr, dtype=np.float32)
    shaded = base[None, :] * factor[:, None]
    return np.clip(shaded, 0, 255)


def _draw_cylinder(img, x0, x1, cy, half_w, base_bgr, alpha=255):
    """Fills columns x0..x1 with a shaded cylinder cross-section (rounded lighting, not flat)."""
    profile = _cylinder_shade(base_bgr, half_w)
    y0, y1 = cy - half_w, cy + half_w
    h, w = img.shape[:2]
    xa, xb = max(0, x0), min(w, x1)
    if xb <= xa:
        return
    block = np.repeat(profile[:, None, :], xb - xa, axis=1)  # (rows, cols, 3)
    ya, yb = max(0, y0), min(h, y1 + 1)
    ra, rb = ya - y0, (y1 + 1 - y0) - (y1 + 1 - yb)
    img[ya:yb, xa:xb, :3] = block[ra:rb]
    img[ya:yb, xa:xb, 3] = alpha


def _round_cap(img, x, cy, half_w, base_bgr, direction=1, alpha=255):
    """Draws a rounded (hemispherical-looking) end cap so the tip isn't a flat edge."""
    profile = _cylinder_shade(base_bgr, half_w)
    h, w = img.shape[:2]
    span = half_w
    for dx in range(span):
        col = x + dx * direction
        if col < 0 or col >= w:
            continue
        shrink = np.sqrt(max(0.0, 1 - (dx / span) ** 2))
        local_half = max(1, int(half_w * shrink))
        rows = np.linspace(-half_w, half_w, 2 * local_half + 1)
        idx = np.clip(((rows + half_w)).astype(int), 0, len(profile) - 1)
        col_colors = profile[idx]
        ya, yb = cy - local_half, cy + local_half + 1
        ya_c, yb_c = max(0, ya), min(h, yb)
        ra, rb = ya_c - ya, (yb - ya) - (yb - yb_c)
        img[ya_c:yb_c, col, :3] = col_colors[ra:rb]
        img[ya_c:yb_c, col, 3] = alpha


def make_cigarette(length=78, width=8):
    """White paper filter cigarette with a cork-style filter tip and glowing ember."""
    pad = width
    img = _blank(length + pad, width * 2 + 4)
    cy = width + 2
    hw = width // 2

    filter_len = int(length * 0.28)
    body_x0 = filter_len

    # paper body: warm-white cylinder shading
    _draw_cylinder(img, body_x0, length, cy, hw, (238, 240, 244))
    # faint printed brand ring near the filter joint
    _draw_cylinder(img, body_x0, body_x0 + 3, cy, hw, (210, 205, 195))

    # cork filter: speckled tan/orange cylinder
    _draw_cylinder(img, 0, body_x0, cy, hw, (128, 168, 205))
    rng = np.random.RandomState(7)
    for _ in range(16):
        fx = rng.randint(2, body_x0 - 2)
        fy = cy + rng.randint(-hw + 1, hw - 1)
        speck = tuple(int(c) for c in (128, 168, 205) + rng.randint(-14, 14, 3))
        cv2.circle(img, (fx, fy), 1, speck, -1)

    _round_cap(img, 0, cy, hw, (128, 168, 205), direction=1)

    # lit ember tip with a soft glow gradient
    ember_len = max(3, int(length * 0.05))
    glow_hw = hw + 3
    for i in range(glow_hw, 0, -1):
        t = i / glow_hw
        color = (10, 60 + int(140 * (1 - t)), 140 + int(115 * (1 - t)))
        cv2.circle(img, (length - 1, cy), i, color, -1)
    cv2.circle(img, (length - 1, cy), max(2, hw // 2), (30, 110, 255), -1)
    img[:, :, 3] = np.where(img[:, :, 3] > 0, 255, img[:, :, 3])
    _draw_cylinder(img, length - ember_len, length, cy, hw, (25, 95, 220))

    return img, (length - 1, cy)


def make_vape(length=65, width=13):
    """Matte pen-style vape: brushed-metal battery/tank body, rounded dark
    mouthpiece at the smoke-emitting (rightmost pre-flip) end, glowing LED.
    """
    img = _blank(length + 4, width * 2 + 4)
    cy = width + 2
    hw = width // 2

    base_x1 = int(length * 0.15)
    tank_x0 = base_x1
    tank_x1 = int(length * 0.78)
    mouth_x0 = tank_x1

    _round_cap(img, 0, cy, hw + 2, (30, 24, 20), direction=1)
    _draw_cylinder(img, 0, base_x1, cy, hw + 2, (30, 24, 20))
    cv2.circle(img, (int(length * 0.07), cy), 2, (255, 190, 60, 255), -1)
    cv2.circle(img, (int(length * 0.07), cy), 4, (150, 120, 40, 120), -1)

    _draw_cylinder(img, tank_x0, tank_x1, cy, hw, (58, 46, 32))
    for gx in range(tank_x0 + 4, tank_x1 - 4, 5):
        cv2.line(img, (gx, cy - hw + 2), (gx, cy + hw - 2), (35, 28, 20, 255), 1)

    _draw_cylinder(img, mouth_x0, length, cy, max(2, hw - 3), (45, 35, 28))
    _round_cap(img, length - 1, cy, max(2, hw - 3), (45, 35, 28), direction=-1)

    return img, (length - 1, cy)


def make_cigar(length=92, width=11):
    """Tapered hand-rolled cigar: rounded cap, tobacco-leaf shading, foot end."""
    img = _blank(length + 4, width * 2 + 4)
    cy = width + 2

    def half_width_at(x):
        t = x / length
        head = np.clip(t / 0.08, 0, 1)
        taper = 1.0 - 0.35 * np.clip((t - 0.55) / 0.45, 0, 1)
        return max(2, int((width / 2) * head * taper))

    base = (35, 65, 95)
    for x in range(length):
        hw = half_width_at(x)
        profile = _cylinder_shade(base, hw)
        y0 = cy - hw
        for i, row in enumerate(profile):
            y = y0 + i
            if 0 <= y < img.shape[0]:
                img[y, x, :3] = row
                img[y, x, 3] = 255

    rng = np.random.RandomState(3)
    for _ in range(60):
        vx = rng.randint(int(length * 0.08), length - 2)
        hw = half_width_at(vx)
        vy = cy + rng.randint(-hw + 1, hw)
        img[vy, vx, :3] = np.clip(img[vy, vx, :3].astype(int) + rng.randint(-18, 8), 0, 255)

    band_x = int(length * 0.32)
    band_w = max(2, int(length * 0.05))
    for x in range(band_x, band_x + band_w):
        hw = half_width_at(x) + 1
        cv2.line(img, (x, cy - hw), (x, cy + hw), (40, 170, 220, 255), 1)
    cv2.line(img, (band_x, cy - half_width_at(band_x) - 1),
             (band_x + band_w, cy - half_width_at(band_x + band_w) - 1), (90, 210, 245, 255), 1)

    foot_hw = half_width_at(length - 2)
    cv2.ellipse(img, (length - 2, cy), (2, foot_hw), 0, -90, 90, (20, 40, 55, 255), -1)

    return img, (length - 1, cy)


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


def tip_position(sprite_bgra, tip, center_x, center_y, angle_deg, scale=1.0):
    """Where `tip` lands after the same rotate+scale `overlay_transparent` applies.

    Mirrors cv2.getRotationMatrix2D's convention, so call it with the identical
    angle/scale used for the blit.
    """
    h, w = sprite_bgra.shape[:2]
    dx = (tip[0] - w / 2) * scale
    dy = (tip[1] - h / 2) * scale
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    return (center_x + cos_a * dx + sin_a * dy,
            center_y - sin_a * dx + cos_a * dy)


def overlay_transparent(frame, sprite_bgra, center_x, center_y, angle_deg, scale=1.0):
    """Rotate/scale sprite_bgra around its own center and alpha-blend onto frame at (center_x, center_y)."""
    h, w = sprite_bgra.shape[:2]
    if h == 0 or w == 0 or scale <= 0:
        return
    # Must fit the sprite at its original size (for placement) AND at its
    # scaled size (for the warpAffine output), whichever is larger.
    diag = int(np.ceil(np.hypot(w, h) * max(scale, 1.0))) + 4
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
