"""Görüntüden baskın renk paleti çıkarma (KMeans)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

SAMPLE_SIDE = 120


def to_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = (int(round(float(c))) for c in rgb)  # noqa: RUF046 — NumPy skaleri
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_palette(rgb: np.ndarray, n_colors: int = 5, seed: int = 0) -> list[str]:
    """RGB dizisinden en sık `n_colors` rengi hex olarak döner (en baskından aza).

    Girdi her zaman 3 kanallı RGB olmalıdır (`vision.load_image` bunu garanti eder).
    Aynı girdi ve tohumla sonuç deterministiktir.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("extract_palette 3 kanallı RGB dizisi bekler.")
    step = max(1, max(rgb.shape[:2]) // SAMPLE_SIDE)
    pixels = rgb[::step, ::step].reshape(-1, 3).astype(np.float32)
    unique = np.unique(pixels, axis=0)
    clusters = max(1, min(n_colors, len(unique)))
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=4).fit(pixels)
    counts = np.bincount(model.labels_, minlength=clusters)
    order = np.argsort(-counts, kind="stable")
    return [to_hex(model.cluster_centers_[i]) for i in order]


def relative_luminance(hex_color: str) -> float:
    """WCAG göreli parlaklık (0–1); palet şeridinde metin rengi seçmek için."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
