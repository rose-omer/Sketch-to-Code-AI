"""Streamlit temasının ulaşamadığı küçük yüzeyler: seçim rengi, palet şeridi, hareket azaltma."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from sketch2code.palette import relative_luminance

ASSETS = Path(__file__).resolve().parents[1] / "assets"

CSS = """
<style>
::selection { background: color-mix(in srgb, var(--st-primary-color, #0F6E8C) 28%, transparent); }
.palette { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 !important; padding: 0 !important; list-style: none; }
.palette li { margin: 0 !important; flex: 1 1 72px; min-width: 72px; height: 56px; border-radius: 6px; display: flex;
  align-items: flex-end; padding: 6px 8px; font: 500 0.75rem/1 "JetBrains Mono", monospace;
  box-shadow: inset 0 0 0 1px rgba(127, 127, 127, 0.35); font-variant-numeric: tabular-nums; }
/* Yükleyicinin İngilizce iç metni: Streamlit yerelleştirme sunmuyor. Özgün metin görünmez ve erişilebilir
   addan çıkar (visibility), yerine Türkçe metin gelir. Ekran okuyucu yalnız Türkçe metni okur. */
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span { visibility: hidden; font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after {
  visibility: visible; content: "Dosya seç"; font-size: 0.875rem; line-height: 1.6; }
[data-testid="stFileUploaderDropzoneInstructions"] span::after {
  visibility: visible; content: "Bırakın ya da seçin · en çok 10 MB"; font-size: 0.875rem; line-height: 1.6; }
.st-key-sketch_files [data-testid="stFileUploaderDropzoneInstructions"] span::after {
  content: "JPG ya da PNG · en çok 10 MB"; }
.st-key-project_file [data-testid="stFileUploaderDropzoneInstructions"] span::after {
  content: "JSON · en çok 10 MB"; }
.st-key-device_grid_narrow { display: none; }
@media (max-width: 640px) {
  .st-key-device_grid { display: none; }
  .st-key-device_grid_narrow { display: block; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
}
</style>
"""


def inject() -> None:
    st.logo(str(ASSETS / "logo.svg"), size="large")
    st.html(CSS)


def palette_strip(colors: list[str]) -> None:
    """Hex etiketli renk şeridi; etiket rengi zemine göre seçilir (yalnız renge dayanmaz)."""
    items = []
    for color in colors:
        ink = "#111418" if relative_luminance(color) > 0.179 else "#FFFFFF"
        safe = html.escape(color)
        items.append(f'<li style="background:{safe};color:{ink}" title="{safe}">{safe}</li>')
    st.html(f'<ul class="palette" aria-label="Çizimden çıkarılan renkler">{"".join(items)}</ul>')
