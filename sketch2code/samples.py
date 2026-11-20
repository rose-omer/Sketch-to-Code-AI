"""Deterministik örnek çizim ve sentetik "telefon fotoğrafı" üretimi.

Arayüzdeki "Örnekle dene" düğmesi ve testlerdeki fikstürler bu modülü kullanır;
repoya ikili fikstür dosyası eklemek gerekmez.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

PAPER_SIZE = (900, 1200)  # (genişlik, yükseklik)
INK = (52, 58, 66)
BLUE_PEN = (40, 90, 170)


def _wobbly_rect(canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], rng: np.random.Generator) -> None:
    """Elle çizilmiş gibi hafif titrek dikdörtgen."""
    (x0, y0), (x1, y1) = p1, p2
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    jitter = [(cx + int(rng.integers(-3, 4)), cy + int(rng.integers(-3, 4))) for cx, cy in corners]
    for start, end in zip(jitter, jitter[1:] + jitter[:1], strict=True):
        cv2.line(canvas, start, end, INK, 3, cv2.LINE_AA)


def _scribble(canvas: np.ndarray, x: int, y: int, length: int, rng: np.random.Generator) -> None:
    """Metin satırını temsil eden dalgalı çizgi."""
    xs = np.arange(x, x + length, 6)
    ys = y + (np.sin(xs / 9.0) * 2 + rng.normal(0, 0.6, len(xs))).astype(int)
    points = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(canvas, [points], False, INK, 2, cv2.LINE_AA)


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int], scale: float = 1.1, color=INK) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def make_sketch(seed: int = 7) -> np.ndarray:
    """Düz kâğıt üzerinde bir açılış sayfası taslağı (RGB)."""
    rng = np.random.default_rng(seed)
    width, height = PAPER_SIZE
    paper = np.full((height, width, 3), (247, 246, 241), dtype=np.uint8)
    paper = np.clip(paper.astype(np.int16) + rng.normal(0, 2.5, paper.shape).astype(np.int16), 0, 255).astype(np.uint8)

    # Üst çubuk: logo + menü
    _wobbly_rect(paper, (60, 60), (840, 150), rng)
    _wobbly_rect(paper, (85, 80), (215, 130), rng)
    _label(paper, "LOGO", (102, 118), 0.9)
    for i, item in enumerate(("Urunler", "Fiyat", "Iletisim")):
        _label(paper, item, (430 + i * 135, 115), 0.85)

    # Ana bölüm: başlık, metin, düğme + görsel
    _label(paper, "Taslaktan siteye", (80, 250), 1.5)
    for row in range(3):
        _scribble(paper, 80, 300 + row * 34, 340 - row * 40, rng)
    _wobbly_rect(paper, (80, 420), (300, 485), rng)
    _label(paper, "Kayit ol", (118, 465), 1.0, BLUE_PEN)
    _wobbly_rect(paper, (480, 200), (840, 500), rng)
    cv2.line(paper, (480, 200), (840, 500), INK, 2, cv2.LINE_AA)
    cv2.line(paper, (840, 200), (480, 500), INK, 2, cv2.LINE_AA)

    # Üç sütun
    for col in range(3):
        x0 = 60 + col * 265
        _wobbly_rect(paper, (x0, 570), (x0 + 245, 860), rng)
        _wobbly_rect(paper, (x0 + 25, 595), (x0 + 85, 655), rng)
        for row in range(4):
            _scribble(paper, x0 + 25, 700 + row * 32, 190 - (row % 2) * 50, rng)

    # Alt bilgi
    _wobbly_rect(paper, (60, 930), (840, 1120), rng)
    for row in range(2):
        _scribble(paper, 90, 990 + row * 40, 300, rng)
    _label(paper, "(c) 2026", (640, 1080), 0.8)
    return paper


def photograph(
    sketch: np.ndarray,
    corners: np.ndarray,
    canvas_size: tuple[int, int] = (1600, 1500),
    shadow: float = 0.45,
    seed: int = 3,
) -> np.ndarray:
    """Çizimi masa üstünde, verilen köşelere perspektifle yerleştirir; yumuşak gölge ekler.

    `corners`: (4, 2) sol-üst, sağ-üst, sağ-alt, sol-alt; canvas koordinatı.
    """
    rng = np.random.default_rng(seed)
    width, height = canvas_size
    yy, xx = np.mgrid[0:height, 0:width]
    grain = (np.sin(xx / 7.0 + np.sin(yy / 40.0) * 3) * 6).astype(np.int16)
    table = np.empty((height, width, 3), dtype=np.int16)
    table[..., 0], table[..., 1], table[..., 2] = 112 + grain, 84 + grain, 62 + grain
    table = np.clip(table + rng.normal(0, 3, table.shape).astype(np.int16), 0, 255).astype(np.uint8)

    sketch_h, sketch_w = sketch.shape[:2]
    source = np.array([[0, 0], [sketch_w - 1, 0], [sketch_w - 1, sketch_h - 1], [0, sketch_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, np.asarray(corners, dtype=np.float32))
    warped = cv2.warpPerspective(sketch, matrix, (width, height))
    mask = cv2.warpPerspective(np.full((sketch_h, sketch_w), 255, np.uint8), matrix, (width, height))
    photo = np.where(mask[..., None] > 127, warped, table)

    # Sol üstten sağ alta azalan ışık + köşede telefon gölgesi
    light = 1.0 - shadow * (xx / width * 0.6 + yy / height * 0.4)
    blob = np.exp(-(((xx - width * 0.75) / (width * 0.25)) ** 2 + ((yy - height * 0.2) / (height * 0.2)) ** 2))
    light = np.clip(light - shadow * 0.5 * blob, 0.25, 1.0)
    return np.clip(photo.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)


SAMPLE_CORNERS = np.array([[260, 140], [1300, 210], [1390, 1380], [170, 1300]], dtype=np.float32)


def sample_photo() -> Image.Image:
    """ "Örnekle dene" için hazır fotoğraf."""
    return Image.fromarray(photograph(make_sketch(), SAMPLE_CORNERS))
