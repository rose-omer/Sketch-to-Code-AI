"""Kâğıt çizimi fotoğrafını düz, temiz ve renkli bir görüntüye çeviren CV hattı.

Adımlar: yükleme (EXIF + RGB + boyut sınırı) → kâğıt bulma ve perspektif düzeltme →
gölge/aydınlatma düzeltme + CLAHE → küçük eğim düzeltme → düzen ipucu (kutular).
Bütün fonksiyonlar saftır: girdi NumPy dizisi ya da PIL görüntüsü, çıktı yeni dizi.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 2048
MIN_PAPER_AREA = 0.2  # kâğıt, görüntü alanının en az %20'si olmalı
MAX_DESKEW_DEG = 10.0


class ImageLoadError(ValueError):
    """Yüklenen dosya görüntü olarak açılamadı."""


@dataclass(frozen=True)
class Box:
    """Normalize edilmiş dikdörtgen (0–1 aralığında, sol üst köşe + boyut)."""

    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "w": round(self.w, 3), "h": round(self.h, 3)}


@dataclass
class ProcessResult:
    original: np.ndarray  # RGB, yönü düzeltilmiş
    corrected: np.ndarray  # RGB, perspektif + aydınlatma düzeltilmiş
    binary: np.ndarray  # tek kanallı, mürekkep = 0
    boxes: list[Box] = field(default_factory=list)
    paper_found: bool = False
    corners: np.ndarray | None = None  # (4, 2), orijinal görüntü koordinatı


# --- Yükleme ---------------------------------------------------------------


def load_image(data: bytes, max_side: int = MAX_SIDE) -> Image.Image:
    """Baytları RGB PIL görüntüsüne çevirir; EXIF yönünü uygular, en uzun kenarı sınırlar.

    Gri (L), paletli (P), saydam (RGBA/LA) ve 16 bit görüntüler RGB'ye dönüşür.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = _to_rgb(image)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageLoadError("Dosya bir JPG ya da PNG görüntüsü olarak açılamadı.") from exc
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA", "PA"):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(image, dtype=np.float64)
        top = arr.max() or 1.0
        image = Image.fromarray((arr / top * 255).astype(np.uint8), mode="L")
    return image.convert("RGB")


# --- Kâğıt bulma ve perspektif ---------------------------------------------


def order_corners(points: np.ndarray) -> np.ndarray:
    """Dört noktayı sol-üst, sağ-üst, sağ-alt, sol-alt sırasına dizer."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(total)], pts[np.argmin(diff)], pts[np.argmax(total)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def find_paper(rgb: np.ndarray) -> np.ndarray | None:
    """En büyük dört köşeli açık renkli bölgeyi (kâğıt) bulur. Bulamazsa None döner."""
    height, width = rgb.shape[:2]
    scale = 800 / max(height, width) if max(height, width) > 800 else 1.0
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)

    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=2)

    # İki aday maske: parlaklık (Otsu) ve kenarlar (Canny). Gölge parlaklık maskesini kesebilir;
    # bu yüzden geçerli adaylardan alanı en büyük olan seçilir. Görüntünün tamamını kaplayan
    # dörtgen (çerçevenin kendisi) kâğıt sayılmaz.
    image_area = float(small.shape[0] * small.shape[1])
    candidates = [
        quad
        for quad in (_largest_quad(mask, MIN_PAPER_AREA * image_area) for mask in (bright, edges))
        if quad is not None and cv2.contourArea(quad) < 0.97 * image_area
    ]
    if not candidates:
        return None
    return order_corners(max(candidates, key=cv2.contourArea) / scale)


def _largest_quad(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < min_area:
            return None
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def warp_paper(rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Dört köşeyi dik bir dikdörtgene açar (perspektif düzeltme)."""
    tl, tr, br, bl = order_corners(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), target)
    return cv2.warpPerspective(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC)


def crop_fraction(rgb: np.ndarray, left: float, right: float, top: float, bottom: float) -> np.ndarray:
    """Elle kırpma: kenarlar 0–1 oranıyla verilir."""
    height, width = rgb.shape[:2]
    x0, x1 = sorted((int(left * width), int(right * width)))
    y0, y1 = sorted((int(top * height), int(bottom * height)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return rgb.copy()
    return rgb[y0:y1, x0:x1].copy()


# --- Aydınlatma ve eğim ------------------------------------------------------


def remove_shadows(rgb: np.ndarray) -> np.ndarray:
    """Arka planı (kâğıt) tahmin edip bölerek gölgeyi alır; L kanalına CLAHE uygular."""
    kernel_size = max(15, (min(rgb.shape[:2]) // 25) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    channels = []
    for channel in cv2.split(rgb):
        background = cv2.medianBlur(cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel), 21)
        channels.append(cv2.divide(channel, background, scale=255))
    flat = cv2.merge(channels)

    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def binarize(rgb: np.ndarray) -> np.ndarray:
    """Düzen ipucu için ikili görüntü. Mürekkep siyah (0), kâğıt beyaz (255)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    block = max(11, (min(gray.shape) // 40) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)


def skew_angle(binary: np.ndarray) -> float:
    """Mürekkep piksellerinden eğim açısını (derece) tahmin eder; ±10° dışını 0 sayar."""
    ink = np.column_stack(np.where(binary < 128))
    if len(ink) < 50:
        return 0.0
    angle = cv2.minAreaRect(ink[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle) if abs(angle) <= MAX_DESKEW_DEG else 0.0


def rotate(rgb: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return rgb
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# --- Düzen ipucu ---------------------------------------------------------------


def detect_boxes(binary: np.ndarray, min_area: float = 0.004, max_boxes: int = 24) -> list[Box]:
    """İkili görüntüdeki dikdörtgen benzeri konturları normalize kutulara çevirir."""
    height, width = binary.shape[:2]
    ink = cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    total = float(height * width)
    boxes: list[Box] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area * total or area > 0.95 * total:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if not 4 <= len(approx) <= 6:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if area < 0.6 * w * h:  # dikdörtgene benzemeyen şekiller
            continue
        candidate = Box(x / width, y / height, w / width, h / height)
        if not any(iou(candidate, kept) > 0.6 for kept in boxes):
            boxes.append(candidate)
        if len(boxes) >= max_boxes:
            break
    return sorted(boxes, key=lambda box: (round(box.y, 2), box.x))


def iou(a: Box, b: Box) -> float:
    """İki kutunun kesişim / birleşim oranı (0–1)."""
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def draw_boxes(rgb: np.ndarray, boxes: list[Box], color: tuple[int, int, int] = (15, 110, 140)) -> np.ndarray:
    """Kutuları yarı saydam dolgu ve ince çerçeveyle çizer."""
    height, width = rgb.shape[:2]
    overlay = rgb.copy()
    output = rgb.copy()
    thickness = max(2, min(height, width) // 300)
    for box in boxes:
        p1 = (int(box.x * width), int(box.y * height))
        p2 = (int((box.x + box.w) * width), int((box.y + box.h) * height))
        cv2.rectangle(overlay, p1, p2, color, -1)
        cv2.rectangle(output, p1, p2, color, thickness)
    return cv2.addWeighted(overlay, 0.18, output, 0.82, 0)


# --- Hat --------------------------------------------------------------------


def process(image: Image.Image, detect_paper: bool = True) -> ProcessResult:
    """Bütün hattı çalıştırır. Elle kırpılmış görüntüde `detect_paper=False` verilir."""
    original = np.asarray(image.convert("RGB"))
    corners = find_paper(original) if detect_paper else None
    flat = warp_paper(original, corners) if corners is not None else original
    corrected = remove_shadows(flat)
    if corners is None:
        corrected = rotate(corrected, skew_angle(binarize(corrected)))
    binary = binarize(corrected)
    return ProcessResult(
        original=original,
        corrected=corrected,
        binary=binary,
        boxes=detect_boxes(binary),
        paper_found=corners is not None,
        corners=corners,
    )
