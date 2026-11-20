import io

import numpy as np
import pytest
from PIL import Image

from sketch2code import samples, vision

# Beş sentetik "telefon fotoğrafı": farklı perspektif, konum ve gölge.
FIXTURES = {
    "duz": ([[300, 150], [1300, 150], [1300, 1350], [300, 1350]], 0.2),
    "sola_egik": ([[260, 140], [1300, 210], [1390, 1380], [170, 1300]], 0.45),
    "saga_egik": ([[380, 220], [1350, 120], [1420, 1290], [230, 1400]], 0.5),
    "yamuk": ([[450, 200], [1150, 200], [1450, 1400], [150, 1400]], 0.3),
    "kucuk_golgeli": ([[500, 350], [1150, 400], [1100, 1150], [450, 1100]], 0.6),
}


@pytest.fixture(scope="module")
def sketch() -> np.ndarray:
    return samples.make_sketch()


@pytest.mark.parametrize("name", list(FIXTURES))
def test_find_paper_corners_within_15px(sketch: np.ndarray, name: str) -> None:
    corners, shadow = FIXTURES[name]
    expected = np.array(corners, dtype=np.float32)
    photo = samples.photograph(sketch, expected, shadow=shadow)

    found = vision.find_paper(photo)

    assert found is not None, name
    assert np.abs(found - expected).max() <= 15


def test_process_corrects_perspective_and_finds_boxes() -> None:
    result = vision.process(samples.sample_photo())

    assert result.paper_found
    assert result.corrected.ndim == 3 and result.corrected.shape[2] == 3
    assert result.binary.ndim == 2
    assert len(result.boxes) >= 5
    assert all(0 <= b.x <= 1 and 0 <= b.y <= 1 and b.w > 0 and b.h > 0 for b in result.boxes)


def test_process_without_paper_keeps_image() -> None:
    blank = Image.new("RGB", (400, 300), (250, 250, 250))

    result = vision.process(blank)

    assert not result.paper_found
    assert result.corrected.shape[:2] == (300, 400)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "image",
    [
        Image.new("L", (64, 48), 128),
        Image.new("P", (64, 48), 3),
        Image.new("RGBA", (64, 48), (10, 20, 30, 0)),
        Image.new("LA", (64, 48), (90, 255)),
        Image.new("I;16", (64, 48), 4000),
    ],
    ids=["L", "P", "RGBA", "LA", "I16"],
)
def test_load_image_any_mode_becomes_rgb_and_processes(image: Image.Image) -> None:
    loaded = vision.load_image(_png(image))

    assert loaded.mode == "RGB"
    assert loaded.size == (64, 48)
    vision.process(loaded)  # çökmemeli


def test_transparent_pixels_become_white() -> None:
    loaded = vision.load_image(_png(Image.new("RGBA", (8, 8), (10, 20, 30, 0))))

    assert loaded.getpixel((0, 0)) == (255, 255, 255)


def test_load_image_applies_exif_orientation() -> None:
    image = Image.new("RGB", (40, 20), (200, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # 90° saat yönünde döndür
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    loaded = vision.load_image(buffer.getvalue())

    assert loaded.size == (20, 40)


def test_load_image_limits_longest_side() -> None:
    loaded = vision.load_image(_png(Image.new("RGB", (4000, 1000), (255, 255, 255))))

    assert max(loaded.size) == vision.MAX_SIDE


def test_load_image_rejects_non_image() -> None:
    with pytest.raises(vision.ImageLoadError):
        vision.load_image(b"bu bir resim degil")


def test_order_corners_any_input_order() -> None:
    shuffled = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)

    ordered = vision.order_corners(shuffled)

    assert ordered.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_crop_fraction_and_degenerate_crop() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)

    assert vision.crop_fraction(rgb, 0.25, 0.75, 0.1, 0.9).shape == (80, 100, 3)
    assert vision.crop_fraction(rgb, 0.5, 0.5, 0, 1).shape == rgb.shape  # boş kırpma yok sayılır


def test_skew_angle_is_bounded() -> None:
    binary = np.full((200, 200), 255, dtype=np.uint8)
    binary[90:110, 20:180] = 0

    assert abs(vision.skew_angle(binary)) <= vision.MAX_DESKEW_DEG
