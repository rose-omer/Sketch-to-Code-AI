"""Değerlendirme ölçütleri: görsel benzerlik (SSIM) ve blok konumu eşleşmesi."""

from __future__ import annotations

import cv2
import numpy as np

from sketch2code.vision import Box, iou

SSIM_WIDTH = 640  # karşılaştırma genişliği (px); iki görüntü de buna ölçeklenir


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def ssim(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Gri, yeniden boyutlu SSIM (Wang ve ark. 2004; 11×11 Gauss, σ=1.5). 1 = aynı.

    Aday, referansın en-boy oranına zorlanır: uzunluk farkı da benzerliği düşürür.
    """
    ref = _gray(reference)
    height = max(1, round(ref.shape[0] * SSIM_WIDTH / ref.shape[1]))
    a = cv2.resize(_gray(candidate), (SSIM_WIDTH, height), interpolation=cv2.INTER_AREA).astype(np.float64)
    b = cv2.resize(ref, (SSIM_WIDTH, height), interpolation=cv2.INTER_AREA).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(score.mean())


def block_match(candidate: list[Box], reference: list[Box]) -> float:
    """Blok konumu eşleşmesi (0–1).

    Kutular IoU'ya göre birebir, açgözlü eşlenir. Eşlenen IoU toplamı büyük listenin boyuna bölünür;
    eksik ve fazla blok aynı ölçüde cezalanır.
    """
    if not candidate and not reference:
        return 1.0
    pairs = sorted(((iou(c, r), i, j) for i, c in enumerate(candidate) for j, r in enumerate(reference)), reverse=True)
    used_c: set[int] = set()
    used_r: set[int] = set()
    total = 0.0
    for score, i, j in pairs:
        if score <= 0:
            break
        if i in used_c or j in used_r:
            continue
        used_c.add(i)
        used_r.add(j)
        total += score
    return total / max(len(candidate), len(reference))
