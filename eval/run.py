"""Değerlendirme hattı: çizim → kod → render → hedef sayfayla karşılaştırma.

Kullanım (worktree kökünde):
    python -m eval.run --data <veri klasörü> --subset 30
    python -m eval.run --data <veri klasörü> --subset 5 --dry-run   # API çağrısı yok, hattı sınar

Veri klasörü düzeni (Sketch2Code veri setinin yerel kopyası; repoya girmez):
    <sayfa>.html            hedef sayfa
    <sayfa>_<n>.png|jpg     o sayfanın çizimleri

Ön işleme kipleri:
    raw       fotoğraf olduğu gibi
    binary    gri + uyarlanır eşik (eski yöntem)
    pipeline  sketch2code.vision hattı (perspektif, gölge, CLAHE) + kutu ipucu

Ölçütler: SSIM (gri, 640 px), blok konumu eşleşmesi (DOM blokları, IoU), üretim süresi, çıktı uzunluğu.
Gerçek çalıştırma GEMINI_API_KEY ister ve ücretli olabilir; CI'da koşmaz.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from eval.metrics import block_match, ssim
from sketch2code import llm, preview, prompts, vision

MODES = ("raw", "binary", "pipeline")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
VIEWPORT = {"width": 1280, "height": 800}
# Ağ isteği yok: render tekrarlanabilir olsun diye üretim "Saf CSS" ile istenir.
OPTIONS = prompts.GenerationOptions(framework="Saf CSS", color_scheme="Nötr ve sade")

BLOCKS_JS = """() => {
  const W = Math.max(document.documentElement.scrollWidth, 1), H = Math.max(document.documentElement.scrollHeight, 1);
  const out = [];
  for (const el of document.body.querySelectorAll('*')) {
    const own = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    const media = ['IMG', 'SVG', 'INPUT', 'BUTTON', 'TEXTAREA', 'SELECT', 'VIDEO'].includes(el.tagName.toUpperCase());
    if (!own && !media) continue;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    if (r.width < 2 || r.height < 2 || s.visibility === 'hidden' || s.display === 'none') continue;
    out.push([(r.left + scrollX) / W, (r.top + scrollY) / H, r.width / W, r.height / H]);
  }
  return out;
}"""


@dataclass(frozen=True)
class Sample:
    page: str
    sketch: Path
    reference: Path


@dataclass
class Row:
    sample: str
    mode: str
    model: str
    ssim: float | None = None
    blocks: float | None = None
    seconds: float | None = None
    chars: int | None = None
    error: str = ""


@dataclass
class Render:
    image: np.ndarray
    blocks: list[vision.Box]


def load_samples(data_dir: Path, subset: int, seed: int = 0) -> list[Sample]:
    """Çizim–hedef çiftlerini bulur; sabit tohumla `subset` tanesini seçer (tekrarlanabilir)."""
    samples = []
    for sketch in sorted(data_dir.iterdir()):
        if sketch.suffix.lower() not in IMAGE_SUFFIXES or "_" not in sketch.stem:
            continue
        reference = data_dir / f"{sketch.stem.rsplit('_', 1)[0]}.html"
        if reference.exists():
            samples.append(Sample(reference.stem, sketch, reference))
    if len(samples) <= subset:
        return samples
    return sorted(random.Random(seed).sample(samples, subset), key=lambda s: s.sketch.name)


def prepare(data: bytes, mode: str) -> tuple[Image.Image, list[vision.Box]]:
    """Kipe göre modele gidecek görüntü ve düzen ipucu."""
    image = vision.load_image(data)
    if mode == "raw":
        return image, []
    if mode == "binary":
        return Image.fromarray(vision.binarize(np.asarray(image))), []
    result = vision.process(image)
    return Image.fromarray(result.corrected), result.boxes


Generate = Callable[[Sample, Image.Image, list[vision.Box], str], str]
RenderFn = Callable[[str], Render]


def gemini_generator(client: llm.SupportsModels) -> Generate:
    def generate(sample: Sample, image: Image.Image, boxes: list[vision.Box], model: str) -> str:
        prompt = prompts.build_generation_prompt(OPTIONS, None, boxes)
        return llm.extract_html("".join(llm.stream_code(client, image, prompt, model=model)))

    return generate


def reference_generator(sample: Sample, image: Image.Image, boxes: list[vision.Box], model: str) -> str:
    """Kuru deneme: model yerine hedef HTML'i döner."""
    return sample.reference.read_text(encoding="utf-8")


@contextmanager
def browser_renderer() -> Iterator[RenderFn]:
    """Playwright Chromium ile render. Kendi Chromium'u yoksa kurulu Chrome ya da Edge denenir."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as p:
        browser = None
        for kwargs in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
            try:
                browser = p.chromium.launch(**kwargs)
                break
            except Error:
                continue
        if browser is None:
            raise RuntimeError("Chromium bulunamadı: playwright install chromium")
        page = browser.new_page(viewport=VIEWPORT)

        def render(html_code: str) -> Render:
            page.set_content(preview.with_csp(html_code), wait_until="load")
            shot = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
            boxes = [vision.Box(*b) for b in page.evaluate(BLOCKS_JS)]
            return Render(np.asarray(shot), boxes)

        try:
            yield render
        finally:
            browser.close()


def evaluate(
    samples: list[Sample], modes: list[str], models: list[str], generate: Generate, render: RenderFn
) -> list[Row]:
    rows = []
    for sample in samples:
        target = render(sample.reference.read_text(encoding="utf-8"))
        data = sample.sketch.read_bytes()
        for mode in modes:
            image, boxes = prepare(data, mode)
            for model in models:
                row = Row(sample.sketch.name, mode, model)
                started = time.perf_counter()
                try:
                    html_code = generate(sample, image, boxes, model)
                except llm.GenerationError as exc:
                    row.error = exc.kind.value
                    rows.append(row)
                    continue
                row.seconds = time.perf_counter() - started
                row.chars = len(html_code)
                got = render(html_code)
                row.ssim = ssim(got.image, target.image)
                row.blocks = block_match(got.blocks, target.blocks)
                rows.append(row)
                print(f"{row.sample} {mode} {model}: ssim={row.ssim:.3f} blok={row.blocks:.3f}", file=sys.stderr)
    return rows


def _mean(values: list[float]) -> str:
    return f"{statistics.fmean(values):.3f}" if values else "—"


def to_markdown(rows: list[Row], note: str) -> str:
    lines = [
        "# Değerlendirme sonuçları",
        "",
        note,
        "",
        "| Kip | Model | Örnek | Başarılı | SSIM | Blok eşleşmesi | Süre (sn) | Çıktı (karakter) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        groups.setdefault((row.mode, row.model), []).append(row)
    for (mode, model), group in groups.items():
        ok = [r for r in group if not r.error]
        lines.append(
            f"| {mode} | {model} | {len(group)} | {len(ok)} | {_mean([r.ssim for r in ok])} | "
            f"{_mean([r.blocks for r in ok])} | {_mean([r.seconds for r in ok])} | "
            f"{_mean([float(r.chars) for r in ok])} |"
        )
    errors = [r for r in rows if r.error]
    if errors:
        lines += ["", "Hatalar: " + ", ".join(f"{r.sample}/{r.mode}/{r.model}: {r.error}" for r in errors)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run", description="Sketch-to-Code değerlendirme hattı")
    parser.add_argument("--data", type=Path, required=True, help="Çizim ve hedef HTML klasörü")
    parser.add_argument("--subset", type=int, default=30, help="Örnek sayısı (varsayılan 30)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", default=",".join(MODES), help="raw,binary,pipeline")
    parser.add_argument("--models", default=f"{llm.model_code()},{llm.model_fast()}")
    parser.add_argument("--out", type=Path, default=Path("eval/results.md"))
    parser.add_argument("--dry-run", action="store_true", help="Model yerine hedef HTML'i kullanır; API çağrısı yok")
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := set(modes) - set(MODES):
        parser.error(f"bilinmeyen kip: {', '.join(sorted(unknown))}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    samples = load_samples(args.data, args.subset, args.seed)
    if not samples:
        parser.error(f"{args.data} içinde <sayfa>.html + <sayfa>_<n>.png çifti yok")

    if args.dry_run:
        generate = reference_generator  # hat uçtan uca sınanır; skor 1'e yakın çıkmalı
    else:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            print("GEMINI_API_KEY tanımlı değil. Kuru deneme için --dry-run verin.", file=sys.stderr)
            return 2
        generate = gemini_generator(llm.make_client(key))

    with browser_renderer() as render:
        rows = evaluate(samples, modes, models, generate, render)

    note = (
        f"Tarih: {date.today().isoformat()} · Örnek: {len(samples)} · Tohum: {args.seed} · "
        f"Veri: `{args.data.name}` · Çerçeve: Saf CSS, 1280 px"
    )
    if args.dry_run:
        print(to_markdown(rows, note + " · KURU DENEME (model yok)"))
        return 0
    args.out.write_text(to_markdown(rows, note), encoding="utf-8")
    print(f"Yazıldı: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
