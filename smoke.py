"""Particle-based smoke/vapor system, rendered onto a single blurred overlay layer per frame."""
import random
import numpy as np
import cv2

STYLES = {
    "cigarette": {"color": (200, 200, 200), "spawn_rate": 3, "max_radius": 14, "life": 55, "spread": 1.2},
    "vape": {"color": (245, 245, 245), "spawn_rate": 6, "max_radius": 22, "life": 45, "spread": 1.8},
    "cigar": {"color": (215, 210, 205), "spawn_rate": 4, "max_radius": 18, "life": 65, "spread": 1.0},
}

MAX_PARTICLES = 150


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "age", "life", "radius", "max_radius", "color")

    def __init__(self, x, y, style):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.6, 0.6) * style["spread"]
        self.vy = random.uniform(-1.8, -0.9)
        self.age = 0
        self.life = style["life"] + random.randint(-10, 10)
        self.radius = random.uniform(2, 4)
        self.max_radius = style["max_radius"]
        self.color = style["color"]

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.vx += random.uniform(-0.08, 0.08)
        self.vy *= 0.99
        self.age += 1
        t = self.age / self.life
        self.radius = 2 + (self.max_radius - 2) * min(t * 1.4, 1.0)

    @property
    def alive(self):
        return self.age < self.life

    @property
    def opacity(self):
        t = self.age / self.life
        return max(0.0, 1.0 - t) * 0.55


class SmokeSystem:
    def __init__(self):
        self.particles = []

    def update(self, spawn_point, item_type, active):
        style = STYLES[item_type]
        if active and spawn_point is not None and len(self.particles) < MAX_PARTICLES:
            for _ in range(style["spawn_rate"]):
                self.particles.append(Particle(spawn_point[0], spawn_point[1], style))

        for p in self.particles:
            p.step()
        self.particles = [p for p in self.particles if p.alive]

    def render(self, frame):
        if not self.particles:
            return
        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_mask = np.zeros((h, w), dtype=np.float32)

        for p in self.particles:
            cx, cy = int(p.x), int(p.y)
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            cv2.circle(overlay, (cx, cy), int(p.radius), p.color, -1)
            cv2.circle(alpha_mask, (cx, cy), int(p.radius), float(p.opacity), -1)

        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        alpha_mask = cv2.GaussianBlur(alpha_mask, (15, 15), 0)
        alpha_3c = alpha_mask[:, :, None]

        frame[:] = (frame.astype(np.float32) * (1 - alpha_3c) + overlay.astype(np.float32) * alpha_3c).astype(np.uint8)
