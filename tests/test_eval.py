"""Değerlendirme hattı: ölçütler, örnek seçimi, akış (sahte üretici ve render; API çağrısı yok)."""

import io
from pathlib import Path

import numpy as np
import pytest

from eval import run
from eval.metrics import block_match, ssim
from sketch2code import llm, samples
from sketch2code.vision import Box
from tests.fakes import PAGE


def test_ssim_is_one_for_same_image_and_lower_for_different() -> None:
    rng = np.random.default_rng(0)
    page = rng.integers(0, 255, (400, 640, 3), dtype=np.uint8)
    blank = np.full_like(page, 255)

    assert ssim(page, page) == pytest.approx(1.0)
    assert ssim(blank, page) < 0.2


def test_block_match_penalises_missing_and_extra_blocks() -> None:
    a, b = Box(0, 0, 0.5, 0.5), Box(0.5, 0.5, 0.5, 0.5)

    assert block_match([a, b], [a, b]) == pytest.approx(1.0)
    assert block_match([a], [a, b]) == pytest.approx(0.5)
    assert block_match([a, b], [a]) == pytest.approx(0.5)
    assert block_match([], []) == 1.0
    assert block_match([], [a]) == 0.0


def _dataset(root: Path) -> Path:
    for page in ("a", "b", "c"):
        (root / f"{page}.html").write_text(PAGE, encoding="utf-8")
        for n in (1, 2):
            samples.sample_photo().save(root / f"{page}_{n}.png")
    samples.sample_photo().save(root / "yetim_1.png")  # hedefi yok: atlanır
    return root


def test_load_samples_pairs_sketches_and_is_reproducible(tmp_path: Path) -> None:
    data = _dataset(tmp_path)

    everything = run.load_samples(data, subset=30)
    first = run.load_samples(data, subset=3, seed=1)

    assert len(everything) == 6
    assert all(s.reference.name == f"{s.page}.html" for s in everything)
    assert first == run.load_samples(data, subset=3, seed=1)
    assert len(first) == 3


@pytest.mark.parametrize("mode", run.MODES)
def test_prepare_modes(mode: str) -> None:
    buffer = io.BytesIO()
    samples.sample_photo().save(buffer, format="PNG")

    image, boxes = run.prepare(buffer.getvalue(), mode)

    assert image.mode in ("RGB", "L")
    assert (len(boxes) > 0) == (mode == "pipeline")


def test_evaluate_scores_rows_and_records_errors(tmp_path: Path) -> None:
    data = _dataset(tmp_path)
    chosen = run.load_samples(data, subset=1)
    image = np.full((800, 1280, 3), 255, np.uint8)
    block = [Box(0.1, 0.1, 0.5, 0.2)]

    def generate(sample: run.Sample, image_in: object, boxes: list[Box], model: str) -> str:
        if model == "bozuk":
            raise llm.GenerationError(llm.ErrorKind.QUOTA)
        return PAGE

    rows = run.evaluate(chosen, ["raw"], ["iyi", "bozuk"], generate, lambda html: run.Render(image, block))

    good, bad = rows
    assert good.ssim == pytest.approx(1.0) and good.blocks == pytest.approx(1.0) and good.chars == len(PAGE)
    assert bad.error == "quota" and bad.ssim is None
    table = run.to_markdown(rows, "not")
    assert "| raw | iyi | 1 | 1 | 1.000 | 1.000 |" in table
    assert "| raw | bozuk | 1 | 0 | — |" in table
    assert "quota" in table


def test_cli_requires_key_without_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert run.main(["--data", str(_dataset(tmp_path)), "--subset", "1"]) == 2


@pytest.mark.browser
def test_dry_run_end_to_end_in_browser(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("playwright.sync_api")
    data = _dataset(tmp_path)

    try:
        code = run.main(["--data", str(data), "--subset", "1", "--modes", "pipeline", "--models", "x", "--dry-run"])
    except RuntimeError as exc:  # Chromium yok
        pytest.skip(str(exc))

    assert code == 0
    out = capsys.readouterr().out
    assert "KURU DENEME" in out
    assert "| pipeline | x | 1 | 1 | 1.000 | 1.000 |" in out
