from __future__ import annotations

import argparse
import math
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont


BASE_W = 1280
BASE_H = 720
AUDIO_SAMPLE_RATE = 48_000

MOTION_SPEED = 0.5
OPENING_END = 3.20
ZOOM_END = 4.45
MMLU_END = 8.65
CHART_IN_TIME = 4.25
BASELINE_TIME = 4.85
IMPROVED_TIME = 6.05
DELTA_TIME = 7.35
LATE_STREAM_TIME = 7.75
CTA_TIME = 8.75

Color = tuple[int, int, int]
Point = tuple[float, float]


TEACHERS = [
    {
        "name": "Qwen3-4B",
        "icon": "Q",
        "pos": (265, 250),
        "color": (255, 72, 226),
        "accent": (180, 128, 255),
    },
    {
        "name": "Phi-4-Mini",
        "icon": "Phi",
        "pos": (640, 126),
        "color": (85, 214, 255),
        "accent": (120, 235, 255),
    },
    {
        "name": "Llama-3B",
        "icon": "L",
        "pos": (1018, 250),
        "color": (255, 207, 75),
        "accent": (255, 225, 130),
    },
]

METRICS = {
    "title": "MMLU",
    "baseline_label": "32.05",
    "baseline": 32.05,
    "improved_label": "46.32",
    "improved": 46.32,
    "delta_label": "+14.27",
}


@dataclass(frozen=True)
class SceneConfig:
    width: int = BASE_W
    height: int = BASE_H
    fps: int = 24
    duration: float = 10.0
    title: str = "Distilling different model families"
    student: str = "Llama-1B"
    cta: str = "Read the Paper"
    paper_url: str = "arxiv.org/pdf/2605.21699"


@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    radius: float
    alpha: int
    phase: float
    speed: float


@dataclass(frozen=True)
class GlobeNode:
    angle: float
    orbit: float
    lift: float
    alpha: int


class SceneAssets:
    def __init__(self, seed: int = 11) -> None:
        rng = random.Random(seed)
        self.bg_particles = [
            Particle(
                rng.uniform(0, BASE_W),
                rng.uniform(0, BASE_H),
                rng.uniform(0.7, 2.2),
                rng.randint(26, 115),
                rng.uniform(0, math.tau),
                rng.uniform(0.16, 0.75),
            )
            for _ in range(190)
        ]
        self.grid_ticks = [rng.uniform(0, 1) for _ in range(80)]
        self.globe_nodes = [
            GlobeNode(
                rng.uniform(0, math.tau),
                rng.uniform(0.16, 0.98),
                rng.uniform(-0.55, 0.55),
                rng.randint(58, 180),
            )
            for _ in range(145)
        ]


class FontBook:
    def __init__(self) -> None:
        self.cache: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def get(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (size, bold)
        if key in self.cache:
            return self.cache[key]

        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                try:
                    font = ImageFont.truetype(candidate, size=size)
                    self.cache[key] = font
                    return font
                except OSError:
                    continue

        font = ImageFont.load_default()
        self.cache[key] = font
        return font


FONTS = FontBook()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_out_back(t: float) -> float:
    t = clamp(t)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def motion_time(t: float) -> float:
    return t * MOTION_SPEED


def rgba(color: Color, alpha: float | int) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], int(clamp(float(alpha) / 255.0) * 255))


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: Point,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int] | Color,
    stroke_fill: tuple[int, int, int, int] | Color | None = None,
    stroke_width: int = 0,
) -> None:
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.text(
        (xy[0] - w / 2, xy[1] - h / 2 - box[1] / 2),
        text,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=stroke_width,
    )


def bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )


def alpha_composite_blur(
    img: Image.Image,
    draw_fn,
    blur_radius: float,
) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    if blur_radius > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur_radius))
    img.alpha_composite(layer)


def draw_glow_circle(
    img: Image.Image,
    center: Point,
    radius: float,
    color: Color,
    alpha: int = 150,
    width: int = 2,
    glow: float = 18,
    fill_alpha: int = 0,
) -> None:
    cx, cy = center

    def glow_shape(d: ImageDraw.ImageDraw) -> None:
        for i, scale in enumerate((1.0, 1.22, 1.48)):
            r = radius * scale
            d.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                outline=rgba(color, alpha / (i + 1.5)),
                width=max(width, int(width * 2.5)),
            )
        if fill_alpha:
            d.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=rgba(color, fill_alpha),
            )

    alpha_composite_blur(img, glow_shape, glow)
    d = ImageDraw.Draw(img)
    d.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=rgba(color, alpha),
        width=width,
    )
    if fill_alpha:
        fill_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        fd = ImageDraw.Draw(fill_layer)
        fd.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=rgba(color, fill_alpha),
        )
        img.alpha_composite(fill_layer)


def draw_glow_line(
    img: Image.Image,
    points: Sequence[Point],
    color: Color,
    width: int = 3,
    alpha: int = 190,
    glow: int = 14,
) -> None:
    if len(points) < 2:
        return

    def glow_shape(d: ImageDraw.ImageDraw) -> None:
        d.line(points, fill=rgba(color, alpha * 0.42), width=glow, joint="curve")

    alpha_composite_blur(img, glow_shape, max(2, glow // 2))
    d = ImageDraw.Draw(img)
    d.line(points, fill=rgba(color, alpha), width=width, joint="curve")


def draw_background(img: Image.Image, assets: SceneAssets, t: float, intensity: float = 1.0) -> None:
    mt = motion_time(t)
    d = ImageDraw.Draw(img)
    for y in range(BASE_H):
        u = y / BASE_H
        teal = int(22 + 18 * (1.0 - u))
        blue = int(32 + 30 * (1.0 - abs(u - 0.45)))
        d.line((0, y, BASE_W, y), fill=(2, teal, blue, 255))

    vignette = Image.new("L", img.size, 0)
    vd = ImageDraw.Draw(vignette)
    for r, a in ((760, 0), (640, 20), (520, 44), (400, 70)):
        vd.ellipse((BASE_W / 2 - r, BASE_H / 2 - r, BASE_W / 2 + r, BASE_H / 2 + r), fill=a)
    vignette = Image.eval(vignette, lambda p: 90 - p)
    dark = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dark.putalpha(vignette.filter(ImageFilter.GaussianBlur(60)))
    img.alpha_composite(dark)

    draw_floor_grid(img, t, intensity)
    draw_circuit_rain(img, assets, mt, intensity)

    pd = ImageDraw.Draw(img)
    for particle in assets.bg_particles:
        twinkle = 0.55 + 0.45 * math.sin(mt * particle.speed * math.tau + particle.phase)
        a = int(particle.alpha * twinkle * intensity)
        y = (particle.y + mt * particle.speed * 22) % BASE_H
        pd.ellipse(
            (particle.x - particle.radius, y - particle.radius, particle.x + particle.radius, y + particle.radius),
            fill=(120, 255, 244, a),
        )


def draw_floor_grid(img: Image.Image, t: float, intensity: float) -> None:
    d = ImageDraw.Draw(img)
    vp = (BASE_W / 2, 300)
    bottom_y = BASE_H + 55
    floor_y = 365
    color = (68, 242, 232)

    for i in range(16):
        p = i / 15
        y = floor_y + (bottom_y - floor_y) * (p**2.25)
        span = lerp(130, 760, p)
        alpha = int(40 * (1 - p) * intensity + 12)
        d.line((vp[0] - span, y, vp[0] + span, y), fill=rgba(color, alpha), width=1)

    for i in range(-10, 11):
        x = BASE_W / 2 + i * 78
        alpha = int((24 + 20 * (1 - abs(i) / 10)) * intensity)
        d.line((x, bottom_y, vp[0] + i * 6, vp[1]), fill=rgba(color, alpha), width=1)

    for r in (120, 172, 235, 304):
        box = (BASE_W / 2 - r * 1.9, 522 - r * 0.5, BASE_W / 2 + r * 1.9, 522 + r * 0.5)
        d.arc(box, start=180, end=360, fill=rgba((79, 249, 243), int(95 * intensity)), width=2)


def draw_circuit_rain(img: Image.Image, assets: SceneAssets, t: float, intensity: float) -> None:
    d = ImageDraw.Draw(img)
    for i, tick in enumerate(assets.grid_ticks):
        x = (tick * BASE_W + math.sin(i * 8.1) * 25) % BASE_W
        y = 20 + ((tick * 503 + t * 28 * (0.35 + tick)) % 330)
        length = 28 + 80 * ((i * 19) % 17) / 17
        alpha = int((18 + 48 * (1 - y / 370)) * intensity)
        d.line((x, y, x, y + length), fill=(82, 242, 226, alpha), width=1)
        if i % 4 == 0:
            d.line((x, y + length, x + 42, y + length), fill=(82, 242, 226, alpha // 2), width=1)


def draw_title(draw: ImageDraw.ImageDraw, cfg: SceneConfig, t: float) -> None:
    alpha = int(255 * ease(t / 0.45) * (1.0 - 0.65 * ease((t - (OPENING_END - 0.55)) / 0.55)))
    text_center(
        draw,
        (BASE_W / 2, 42),
        cfg.title,
        FONTS.get(34, True),
        (246, 255, 252, alpha),
        stroke_fill=(20, 48, 52, alpha // 2),
        stroke_width=1,
    )


def draw_model_card(
    img: Image.Image,
    center: Point,
    label: str,
    icon: str,
    color: Color,
    accent: Color,
    t: float,
    delay: float,
) -> None:
    appear = ease_out_back((t - delay) / 0.7)
    if appear <= 0:
        return
    cx, cy = center
    scale = 0.84 + 0.16 * appear
    w = 150 * scale
    h = 118 * scale
    rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def glow_shape(d: ImageDraw.ImageDraw) -> None:
        d.rounded_rectangle(rect, radius=int(18 * scale), outline=rgba(color, 130 * appear), width=7)

    alpha_composite_blur(img, glow_shape, 14)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(rect, radius=int(18 * scale), fill=(13, 31, 42, int(180 * appear)), outline=rgba(accent, 210 * appear), width=2)
    d.rounded_rectangle(
        (rect[0] + 7, rect[1] + 7, rect[2] - 7, rect[3] - 7),
        radius=int(13 * scale),
        outline=rgba((255, 255, 255), 38 * appear),
        width=1,
    )

    icon_r = 28 * scale
    draw_glow_circle(img, (cx, cy - 17 * scale), icon_r, color, int(150 * appear), width=max(1, int(2 * scale)), glow=10, fill_alpha=int(28 * appear))
    text_center(d, (cx, cy - 18 * scale), icon, FONTS.get(max(17, int(24 * scale)), True), (245, 255, 255, int(240 * appear)))
    text_center(d, (cx, cy + 42 * scale), label, FONTS.get(max(13, int(17 * scale)), True), (235, 248, 248, int(245 * appear)))


def stream_control_points(start: Point, end: Point, bend: float) -> tuple[Point, Point, Point, Point]:
    sx, sy = start
    ex, ey = end
    mid_y = (sy + ey) / 2
    return (
        start,
        (lerp(sx, ex, 0.34), mid_y - bend),
        (lerp(sx, ex, 0.74), mid_y + bend * 0.35),
        end,
    )


def draw_particle_stream(
    img: Image.Image,
    start: Point,
    end: Point,
    color: Color,
    t: float,
    delay: float,
    bend: float,
    reverse: bool = False,
    density: int = 78,
    width: int = 3,
) -> None:
    active = ease((t - delay) / 0.9)
    if active <= 0:
        return
    mt = motion_time(t)

    p0, p1, p2, p3 = stream_control_points(start, end, bend)
    if reverse:
        p0, p3 = p3, p0
        p1, p2 = p2, p1

    trail = [bezier(p0, p1, p2, p3, i / 80) for i in range(81)]
    draw_glow_line(img, trail, color, width=max(1, width - 1), alpha=int(64 * active), glow=18)
    d = ImageDraw.Draw(img)

    for i in range(density):
        phase = i / density
        pulse = (mt * 0.72 + phase * 1.45 + (i % 7) * 0.011) % 1.0
        progress = ease(pulse)
        x, y = bezier(p0, p1, p2, p3, progress)
        px, py = bezier(p0, p1, p2, p3, max(0, progress - 0.026))
        local_alpha = int(active * (70 + 160 * progress) * (0.65 + 0.35 * math.sin(i * 1.7 + mt * 5)))
        if progress < active + 0.08:
            d.line((px, py, x, y), fill=rgba(color, local_alpha), width=max(1, width))
            r = 1.6 + 2.3 * progress
            d.ellipse((x - r, y - r, x + r, y + r), fill=rgba(color, local_alpha))


def draw_central_orb(img: Image.Image, cfg: SceneConfig, t: float, zoom: float = 1.0, label_alpha: float = 1.0) -> None:
    cx, cy = BASE_W / 2, 345
    mt = motion_time(t)
    pulse = 0.5 + 0.5 * math.sin(mt * math.tau * 0.8)
    radius = 58 * zoom
    draw_glow_circle(img, (cx, cy), radius + 18 * pulse, (255, 197, 67), 120, width=max(2, int(3 * zoom)), glow=24, fill_alpha=14)
    draw_glow_circle(img, (cx, cy), radius, (82, 247, 242), 190, width=max(2, int(3 * zoom)), glow=18, fill_alpha=18)
    draw_glow_circle(img, (cx, cy), radius * 0.62, (255, 82, 232), 150, width=max(1, int(2 * zoom)), glow=14, fill_alpha=8)

    d = ImageDraw.Draw(img)
    for i in range(38):
        a = i * 0.73 + mt * 2.4
        rr = radius * (0.18 + 0.72 * ((i * 17) % 31) / 31)
        x = cx + math.cos(a) * rr
        y = cy + math.sin(a * 1.33) * rr * 0.62
        d.ellipse((x - 1.7, y - 1.7, x + 1.7, y + 1.7), fill=(255, 255, 255, 130))

    if label_alpha > 0:
        text_center(
            d,
            (cx, cy + radius + 52 * zoom),
            cfg.student,
            FONTS.get(max(16, int(25 * zoom)), True),
            (244, 255, 252, int(240 * label_alpha)),
            stroke_fill=(2, 16, 20, int(210 * label_alpha)),
            stroke_width=2,
        )


def draw_opening_scene(img: Image.Image, cfg: SceneConfig, assets: SceneAssets, t: float) -> None:
    draw_background(img, assets, t, 1.0)
    d = ImageDraw.Draw(img)
    draw_title(d, cfg, t)

    center = (BASE_W / 2, 345)
    for idx, teacher in enumerate(TEACHERS):
        tx, ty = teacher["pos"]
        start = (tx, ty + (48 if idx != 1 else 58))
        end = (center[0], center[1])
        bend = (-70 if idx == 0 else 70 if idx == 2 else 40)
        draw_particle_stream(
            img,
            start,
            end,
            teacher["color"],
            t,
            0.22 + idx * 0.08,
            bend,
            density=72,
            width=3,
        )

    draw_central_orb(img, cfg, t, 1.0, label_alpha=ease((t - 0.58) / 0.45))

    for idx, teacher in enumerate(TEACHERS):
        draw_model_card(
            img,
            teacher["pos"],
            teacher["name"],
            teacher["icon"],
            teacher["color"],
            teacher["accent"],
            t,
            0.05 + idx * 0.08,
        )


def draw_zoom_scene(img: Image.Image, cfg: SceneConfig, assets: SceneAssets, t: float) -> None:
    draw_background(img, assets, t, 0.82)
    local_t = t - OPENING_END
    progress = ease(local_t / (ZOOM_END - OPENING_END))
    center = (BASE_W / 2, 350)

    for teacher in TEACHERS:
        tx, ty = teacher["pos"]
        edge_x = -110 if tx < BASE_W / 2 else BASE_W + 110 if tx > BASE_W / 2 else BASE_W / 2
        edge_y = ty + 40
        draw_particle_stream(
            img,
            (edge_x, edge_y),
            center,
            teacher["color"],
            local_t,
            0.0,
            80 if tx > BASE_W / 2 else -80,
            density=90,
            width=4,
        )

    draw_central_orb(img, cfg, t, zoom=lerp(1.25, 2.35, progress), label_alpha=0.0)

    d = ImageDraw.Draw(img)
    text_center(
        d,
        (BASE_W / 2, 620 + 35 * (1 - progress)),
        cfg.student,
        FONTS.get(54, True),
        (248, 255, 253, int(220 * progress * (1 - progress * 0.5))),
        stroke_fill=(3, 18, 20, 190),
        stroke_width=3,
    )


def globe_project(node: GlobeNode, t: float, center: Point, radius: float) -> Point:
    angle = node.angle + motion_time(t) * (0.09 + node.orbit * 0.09)
    x = center[0] + math.cos(angle) * radius * node.orbit
    y = center[1] + math.sin(angle) * radius * 0.58 * node.orbit + node.lift * radius * 0.24
    return (x, y)


def draw_wire_globe(img: Image.Image, assets: SceneAssets, t: float, center: Point = (640, 365), radius: float = 312) -> None:
    draw_glow_circle(img, center, radius, (180, 245, 255), 75, width=2, glow=18, fill_alpha=5)
    draw_glow_circle(img, center, radius * 0.54, (255, 203, 91), 85, width=2, glow=15, fill_alpha=4)

    d = ImageDraw.Draw(img)
    for i in range(-4, 5):
        y_scale = abs(i) / 5
        y = center[1] + i * radius * 0.115
        rx = radius * math.sqrt(max(0.08, 1 - y_scale * y_scale))
        d.ellipse((center[0] - rx, y - 28, center[0] + rx, y + 28), outline=(195, 248, 255, 32), width=1)

    projected = [globe_project(node, t, center, radius) for node in assets.globe_nodes]
    for i, p in enumerate(projected):
        if i % 3 != 0:
            continue
        for j in range(i + 1, min(len(projected), i + 8)):
            q = projected[j]
            dist = math.dist(p, q)
            if dist < 82:
                alpha = int((1 - dist / 82) * 82)
                d.line((p[0], p[1], q[0], q[1]), fill=(210, 250, 255, alpha), width=1)

    for node, p in zip(assets.globe_nodes, projected):
        alpha = int(node.alpha * (0.75 + 0.25 * math.sin(motion_time(t) * 2 + node.angle)))
        r = 1.25 + node.orbit * 1.5
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(220, 255, 255, alpha))


def draw_chart_frame(img: Image.Image, alpha: float) -> None:
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 430, 150, 850, 560
    a = int(210 * alpha)
    color = (240, 250, 250, a)
    corner = 58
    width = 3
    d.line((left, top, left + corner, top), fill=color, width=width)
    d.line((left, top, left, top + corner), fill=color, width=width)
    d.line((right, top, right - corner, top), fill=color, width=width)
    d.line((right, top, right, top + corner), fill=color, width=width)
    d.line((left, bottom, left + corner, bottom), fill=color, width=width)
    d.line((left, bottom, left, bottom - corner), fill=color, width=width)
    d.line((right, bottom, right - corner, bottom), fill=color, width=width)
    d.line((right, bottom, right, bottom - corner), fill=color, width=width)


def draw_bar(
    img: Image.Image,
    x: float,
    baseline_y: float,
    width: float,
    height: float,
    color: Color,
    alpha: int,
    glow: bool = False,
) -> None:
    rect = (x - width / 2, baseline_y - height, x + width / 2, baseline_y)
    if glow:
        def glow_shape(d: ImageDraw.ImageDraw) -> None:
            d.rounded_rectangle(rect, radius=8, fill=rgba(color, alpha * 0.55))

        alpha_composite_blur(img, glow_shape, 15)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(rect, radius=7, fill=rgba(color, alpha))
    d.rectangle((rect[0], baseline_y - 9, rect[2], baseline_y), fill=rgba(tuple(max(0, c - 25) for c in color), alpha))


def draw_mmlu_scene(img: Image.Image, cfg: SceneConfig, assets: SceneAssets, t: float) -> None:
    draw_background(img, assets, t, 0.75)
    chart_in = ease((t - CHART_IN_TIME) / 0.55)
    draw_wire_globe(img, assets, t)
    draw_chart_frame(img, chart_in)

    d = ImageDraw.Draw(img)
    text_center(
        d,
        (BASE_W / 2, 178),
        METRICS["title"],
        FONTS.get(56, True),
        (255, 255, 250, int(245 * chart_in)),
        stroke_fill=(8, 18, 18, int(190 * chart_in)),
        stroke_width=3,
    )

    baseline_y = 522
    max_value = 55.0
    max_height = 255
    base_progress = ease((t - BASELINE_TIME) / 0.95)
    improved_progress = ease((t - IMPROVED_TIME) / 1.25)
    base_h = max_height * METRICS["baseline"] / max_value * base_progress
    improved_h = max_height * METRICS["improved"] / max_value * improved_progress

    draw_bar(img, 562, baseline_y, 68, base_h, (236, 239, 230), int(230 * chart_in), glow=False)
    draw_bar(img, 708, baseline_y, 70, improved_h, (43, 238, 210), int(235 * chart_in), glow=True)

    if base_progress > 0.8:
        text_center(
            d,
            (562, baseline_y - base_h - 24),
            METRICS["baseline_label"],
            FONTS.get(22, True),
            (255, 255, 255, int(230 * chart_in)),
            stroke_fill=(5, 15, 16, 180),
            stroke_width=2,
        )

    if improved_progress > 0.76:
        text_center(
            d,
            (708, baseline_y - improved_h - 28),
            METRICS["improved_label"],
            FONTS.get(35, True),
            (255, 255, 255, int(245 * improved_progress)),
            stroke_fill=(5, 15, 16, 220),
            stroke_width=2,
        )

    delta_in = ease((t - DELTA_TIME) / 0.42)
    if delta_in > 0:
        badge_w, badge_h = 84 * delta_in, 34 * delta_in
        bx, by = 614, baseline_y - base_h - 28
        rect = (bx - badge_w / 2, by - badge_h / 2, bx + badge_w / 2, by + badge_h / 2)

        def badge_glow(gd: ImageDraw.ImageDraw) -> None:
            gd.rounded_rectangle(rect, radius=8, fill=rgba((43, 238, 210), 150 * delta_in))

        alpha_composite_blur(img, badge_glow, 9)
        d.rounded_rectangle(rect, radius=8, fill=rgba((43, 238, 210), 235 * delta_in))
        text_center(d, (bx, by), METRICS["delta_label"], FONTS.get(max(10, int(19 * delta_in)), True), (7, 27, 30, int(255 * delta_in)))
        d.line((655, by, 687, baseline_y - improved_h + 8), fill=rgba((43, 238, 210), 180 * delta_in), width=2)

    if t > LATE_STREAM_TIME:
        for teacher in TEACHERS:
            tx, ty = teacher["pos"]
            edge_x = -90 if tx < BASE_W / 2 else BASE_W + 90 if tx > BASE_W / 2 else BASE_W / 2
            draw_particle_stream(
                img,
                (edge_x, ty + 10),
                (BASE_W / 2, 370),
                teacher["color"],
                t,
                LATE_STREAM_TIME,
                55 if tx > BASE_W / 2 else -55,
                density=42,
                width=2,
            )


def draw_paper_icon(img: Image.Image, center: Point, scale: float, alpha: float) -> None:
    cx, cy = center
    d = ImageDraw.Draw(img)
    w, h = 86 * scale, 108 * scale
    x0, y0 = cx - w / 2, cy - h / 2
    fold = 24 * scale
    points = [(x0, y0), (x0 + w - fold, y0), (x0 + w, y0 + fold), (x0 + w, y0 + h), (x0, y0 + h)]

    def glow_shape(gd: ImageDraw.ImageDraw) -> None:
        gd.polygon(points, fill=rgba((225, 255, 252), 120 * alpha))

    alpha_composite_blur(img, glow_shape, 15)
    d.polygon(points, fill=rgba((240, 255, 252), 235 * alpha), outline=rgba((255, 255, 255), 245 * alpha))
    d.polygon(
        [(x0 + w - fold, y0), (x0 + w - fold, y0 + fold), (x0 + w, y0 + fold)],
        fill=rgba((169, 231, 229), 235 * alpha),
    )
    for i in range(4):
        y = y0 + 42 * scale + i * 15 * scale
        d.line((x0 + 17 * scale, y, x0 + w - 18 * scale, y), fill=rgba((17, 58, 62), 190 * alpha), width=max(1, int(3 * scale)))


def draw_cta_scene(img: Image.Image, cfg: SceneConfig, assets: SceneAssets, t: float) -> None:
    draw_background(img, assets, t, 0.52)
    local = t - CTA_TIME
    cta_in = ease(local / 0.55)
    d = ImageDraw.Draw(img)

    for idx, radius in enumerate((76, 118, 164)):
        pulse = (motion_time(local) * 0.75 + idx * 0.27) % 1.0
        a = int((1.0 - pulse) * 120 * cta_in)
        draw_glow_circle(img, (BASE_W / 2, 404), radius + pulse * 38, (76, 242, 232), a, width=2, glow=8)

    text_center(
        d,
        (BASE_W / 2, 244),
        cfg.cta,
        FONTS.get(46, True),
        (248, 255, 253, int(245 * cta_in)),
        stroke_fill=(3, 16, 18, int(210 * cta_in)),
        stroke_width=3,
    )
    draw_paper_icon(img, (BASE_W / 2, 406), 0.92 * cta_in, cta_in)
    text_center(
        d,
        (BASE_W / 2, 530),
        cfg.paper_url,
        FONTS.get(24, False),
        (160, 238, 232, int(210 * cta_in)),
    )


def render_base_frame(cfg: SceneConfig, assets: SceneAssets, t: float) -> Image.Image:
    img = Image.new("RGBA", (BASE_W, BASE_H), (0, 0, 0, 255))
    if t < OPENING_END:
        draw_opening_scene(img, cfg, assets, t)
    elif t < ZOOM_END:
        draw_zoom_scene(img, cfg, assets, t)
    elif t < MMLU_END:
        draw_mmlu_scene(img, cfg, assets, t)
    else:
        draw_cta_scene(img, cfg, assets, t)

    fade_in = ease(t / 0.45)
    fade_out = 1.0 - ease((t - (cfg.duration - 0.55)) / 0.55)
    opacity = min(fade_in, fade_out)
    if opacity < 1.0:
        black = Image.new("RGBA", img.size, (0, 0, 0, int(255 * (1.0 - opacity))))
        img.alpha_composite(black)
    return img


def render_frame(cfg: SceneConfig, assets: SceneAssets, t: float) -> Image.Image:
    img = render_base_frame(cfg, assets, t)
    if cfg.width != BASE_W or cfg.height != BASE_H:
        img = img.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
    return img.convert("RGB")


def stereo_gains(pan: float) -> tuple[float, float]:
    angle = (clamp((pan + 1.0) / 2.0) * math.pi) / 2.0
    return math.cos(angle), math.sin(angle)


def event_envelope(local_t: float, length: float, attack: float, release: float) -> float:
    if local_t < 0.0 or local_t > length:
        return 0.0
    attack_gain = ease(local_t / max(attack, 0.001))
    release_gain = 1.0 - ease((local_t - (length - release)) / max(release, 0.001))
    return clamp(min(attack_gain, release_gain))


def add_whoosh(
    left: list[float],
    right: list[float],
    start: float,
    length: float,
    amp: float,
    pan: float,
    seed: int,
    rising: bool = True,
) -> None:
    rng = random.Random(seed)
    start_i = max(0, int(start * AUDIO_SAMPLE_RATE))
    end_i = min(len(left), int((start + length) * AUDIO_SAMPLE_RATE))
    lg, rg = stereo_gains(pan)
    noise = 0.0
    phase = 0.0

    for i in range(start_i, end_i):
        local = i / AUDIO_SAMPLE_RATE - start
        p = clamp(local / length)
        env = math.sin(math.pi * p) ** 0.55
        sweep = p if rising else 1.0 - p
        freq = 120.0 + 840.0 * sweep
        phase += math.tau * freq / AUDIO_SAMPLE_RATE
        noise = noise * 0.88 + (rng.random() * 2.0 - 1.0) * 0.12
        tone = math.sin(phase) * (0.35 + 0.65 * p)
        sample = (noise * 0.78 + tone * 0.22) * amp * env
        left[i] += sample * lg
        right[i] += sample * rg


def add_ping(
    left: list[float],
    right: list[float],
    start: float,
    freq: float,
    amp: float,
    pan: float,
    length: float = 0.72,
) -> None:
    start_i = max(0, int(start * AUDIO_SAMPLE_RATE))
    end_i = min(len(left), int((start + length) * AUDIO_SAMPLE_RATE))
    lg, rg = stereo_gains(pan)

    for i in range(start_i, end_i):
        local = i / AUDIO_SAMPLE_RATE - start
        env = math.exp(-local * 6.5) * ease(local / 0.018)
        shimmer = math.sin(math.tau * freq * local) + 0.35 * math.sin(math.tau * freq * 2.01 * local)
        sample = shimmer * amp * env
        left[i] += sample * lg
        right[i] += sample * rg


def add_thump(left: list[float], right: list[float], start: float, amp: float = 0.16) -> None:
    length = 0.55
    start_i = max(0, int(start * AUDIO_SAMPLE_RATE))
    end_i = min(len(left), int((start + length) * AUDIO_SAMPLE_RATE))

    for i in range(start_i, end_i):
        local = i / AUDIO_SAMPLE_RATE - start
        env = math.exp(-local * 9.0) * ease(local / 0.022)
        freq = 68.0 - 26.0 * clamp(local / length)
        sample = math.sin(math.tau * freq * local) * amp * env
        left[i] += sample
        right[i] += sample


def add_riser(left: list[float], right: list[float], start: float, length: float, amp: float = 0.08) -> None:
    start_i = max(0, int(start * AUDIO_SAMPLE_RATE))
    end_i = min(len(left), int((start + length) * AUDIO_SAMPLE_RATE))
    phase_l = 0.0
    phase_r = 0.5

    for i in range(start_i, end_i):
        local = i / AUDIO_SAMPLE_RATE - start
        p = clamp(local / length)
        env = event_envelope(local, length, 0.08, 0.18)
        freq = 180.0 + 920.0 * p * p
        phase_l += math.tau * freq / AUDIO_SAMPLE_RATE
        phase_r += math.tau * (freq * 1.012) / AUDIO_SAMPLE_RATE
        sample_l = math.sin(phase_l) * amp * env * (0.65 + 0.35 * p)
        sample_r = math.sin(phase_r) * amp * env * (0.65 + 0.35 * p)
        left[i] += sample_l
        right[i] += sample_r


def synthesize_audio(cfg: SceneConfig, audio_path: Path) -> Path:
    total_samples = int(round(cfg.duration * AUDIO_SAMPLE_RATE))
    left = [0.0] * total_samples
    right = [0.0] * total_samples

    for i in range(total_samples):
        t = i / AUDIO_SAMPLE_RATE
        fade = min(ease(t / 0.8), 1.0 - ease((t - (cfg.duration - 0.8)) / 0.8))
        wobble = 0.14 * math.sin(math.tau * 0.08 * t)
        bass = math.sin(math.tau * 45.0 * t + wobble) * 0.040
        low_mid = math.sin(math.tau * 91.0 * t + 0.45 * math.sin(math.tau * 0.05 * t)) * 0.020
        high = math.sin(math.tau * 242.0 * t + 0.7 * math.sin(math.tau * 0.13 * t)) * 0.006
        shimmer = math.sin(math.tau * 731.0 * t + math.sin(math.tau * 0.19 * t)) * 0.004
        drift = 0.92 + 0.08 * math.sin(math.tau * 0.045 * t)
        bed = (bass + low_mid + high + shimmer) * fade
        left[i] += bed * drift
        right[i] += bed * (1.0 / drift)

    add_ping(left, right, 0.20, 520, 0.040, -0.55, 0.42)
    add_ping(left, right, 0.36, 650, 0.038, 0.0, 0.42)
    add_ping(left, right, 0.52, 780, 0.040, 0.55, 0.42)
    add_whoosh(left, right, 0.28, 1.70, 0.090, -0.62, 3)
    add_whoosh(left, right, 0.40, 1.62, 0.078, 0.0, 7)
    add_whoosh(left, right, 0.52, 1.70, 0.090, 0.62, 11)
    add_riser(left, right, OPENING_END, ZOOM_END - OPENING_END, 0.070)
    add_thump(left, right, ZOOM_END, 0.110)
    add_ping(left, right, BASELINE_TIME, 410, 0.052, -0.20, 0.55)
    add_ping(left, right, IMPROVED_TIME, 615, 0.070, 0.22, 0.68)
    add_riser(left, right, DELTA_TIME - 0.30, 0.55, 0.050)
    add_ping(left, right, DELTA_TIME, 940, 0.075, 0.0, 0.82)
    add_whoosh(left, right, LATE_STREAM_TIME + 0.03, 0.85, 0.050, 0.0, 17, rising=False)
    add_thump(left, right, 8.56, 0.085)
    add_ping(left, right, 8.74, 690, 0.075, 0.0, 0.95)
    add_ping(left, right, 9.04, 1035, 0.045, 0.25, 0.80)

    peak = max(max(abs(v) for v in left), max(abs(v) for v in right), 0.001)
    gain = min(6.0, 0.70 / peak)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(AUDIO_SAMPLE_RATE)
        frames = bytearray()
        for l_sample, r_sample in zip(left, right):
            l_int = int(max(-1.0, min(1.0, l_sample * gain)) * 32767)
            r_int = int(max(-1.0, min(1.0, r_sample * gain)) * 32767)
            frames.extend(struct.pack("<hh", l_int, r_int))
        wav.writeframes(frames)
    return audio_path


def encode_video(frame_dir: Path, cfg: SceneConfig, out_path: Path, audio_path: Path | None = None) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(cfg.fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
    ]
    if audio_path:
        cmd.extend(["-i", str(audio_path)])
    cmd.extend([
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ])
    if audio_path:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.extend(["-movflags", "+faststart", str(out_path)])
    subprocess.run(cmd, check=True)


def render_video(cfg: SceneConfig, out_path: Path, keep_frames: bool = False, audio: bool = True) -> Path:
    assets = SceneAssets()
    total_frames = int(round(cfg.duration * cfg.fps))
    temp_parent = out_path.parent / ".frames"
    temp_parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix="distill_", dir=temp_parent))

    try:
        for index in range(total_frames):
            t = index / cfg.fps
            frame = render_frame(cfg, assets, t)
            frame.save(frame_dir / f"frame_{index + 1:04d}.png", optimize=False)
            if index == 0 or (index + 1) % cfg.fps == 0 or index + 1 == total_frames:
                print(f"rendered {index + 1:03d}/{total_frames} frames", flush=True)
        audio_path = None
        if audio:
            audio_path = synthesize_audio(cfg, frame_dir / "soundtrack.wav")
            print(f"rendered soundtrack {audio_path}", flush=True)
        encode_video(frame_dir, cfg, out_path, audio_path)
        print(f"wrote {out_path}", flush=True)
    finally:
        if keep_frames:
            print(f"kept frames in {frame_dir}", flush=True)
        else:
            shutil.rmtree(frame_dir, ignore_errors=True)
            try:
                temp_parent.rmdir()
            except OSError:
                pass

    return out_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a neon AI distillation explainer video.")
    parser.add_argument("--out", type=Path, default=Path("output/distillation_style.mp4"), help="Output MP4 path.")
    parser.add_argument("--duration", type=positive_float, default=10.0, help="Video duration in seconds.")
    parser.add_argument("--fps", type=positive_int, default=24, help="Frames per second.")
    parser.add_argument("--width", type=positive_int, default=BASE_W, help="Output width.")
    parser.add_argument("--height", type=positive_int, default=BASE_H, help="Output height.")
    parser.add_argument("--title", default="Distilling different model families", help="Opening title.")
    parser.add_argument("--student", default="Llama-1B", help="Central student model label.")
    parser.add_argument("--cta", default="Read the Paper", help="Final call-to-action.")
    parser.add_argument("--paper-url", default="arxiv.org/pdf/2605.21699", help="Small URL text shown under CTA.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep intermediate PNG frames.")
    parser.add_argument("--no-audio", action="store_true", help="Export a silent MP4.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = SceneConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
        title=args.title,
        student=args.student,
        cta=args.cta,
        paper_url=args.paper_url,
    )
    try:
        render_video(cfg, args.out, keep_frames=args.keep_frames, audio=not args.no_audio)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
