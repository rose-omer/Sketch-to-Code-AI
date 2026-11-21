"""Tasarım dili "Önce ve Sonra": Streamlit temasının ulaşamadığı yüzeyler.

Renk, yazı tipi ve köşe yarıçapı `.streamlit/config.toml` içindedir. Bu dosya geri kalanı kurar:
geniş kesim başlıklar, mürekkep çerçeveli düğmeler (en az 44 px), düzeltme formu, kutu listesi,
üç sütunlu sürüm satırları, kod renkleri, odak halkası, geniş ekranda kutu listesinin sola alınması
ve hareket azaltma. Tokenler temaya göre `:root` üzerinde tanımlanır.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from sketch2code.palette import relative_luminance

ASSETS = Path(__file__).resolve().parents[1] / "assets"

ACCENT_RGB = (42, 70, 232)  # #2A46E8, taslak mavisi: kutu çerçeveleri ve numaralar

# Tema tokenleri. Kontrast (WCAG): mürekkep/kâğıt 16:1, silik/kâğıt 7.3:1; koyu temada 14:1 ve 7.4:1.
TOKENS = {
    "light": {
        "ink": "#18203A",
        "paper": "#FFFFFF",
        "muted": "#4D5670",
        "rule": "#D7DBE6",
        "accent": "#2A46E8",
        "code-tag": "#1F3BB8",  # kod zemininde (#F3F4F8) 8:1
        "code-attr": "#7A2E8C",  # 7,4:1
        "code-string": "#1D6440",  # 6,5:1
        "code-number": "#9A3412",  # 6,7:1
    },
    "dark": {
        "ink": "#E6E9F2",
        "paper": "#121831",
        "muted": "#A3AAC0",
        "rule": "#343C5C",
        "accent": "#6F86FF",  # odak halkası koyu zeminde 5.6:1
        "code-tag": "#A9B8FF",  # kod zemininde (#1B2242) 8:1
        "code-attr": "#E3A8F0",  # 8,2:1
        "code-string": "#8FD9AE",  # 9,4:1
        "code-number": "#FFB38A",  # 8,9:1
    },
}

CSS = """
<style>
::selection { background: rgba(42, 70, 232, 0.24); }

/* Başlıklar: Archivo geniş kesim (wdth 125). Tek aile; kişilik kesimden gelir. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-variation-settings: "wdth" 125; letter-spacing: -0.01em; }
.stApp [data-testid="stHeading"]:has(h1) { border-bottom: 1.5px solid var(--s2c-ink); padding-bottom: 0.35rem;
  margin-bottom: 0.25rem; }
.stApp [data-testid="stMain"] [data-testid="stHeading"]:has(h2) { border-top: 1px solid var(--s2c-rule);
  padding-top: 1.1rem; margin-top: 0.75rem; }
.stApp h3 { font-size: 0.95rem !important; padding-bottom: 0.25rem !important; }

/* Düğmeler: ikincil düğme mürekkep çerçeveli; birincil düğme taslak mavisi (temadan). Dokunma hedefi en az 44 px. */
.stApp :is([data-testid^="stBaseButton-primary"], [data-testid^="stBaseButton-secondary"],
  [data-testid^="stBaseButton-tertiary"]) { min-height: 44px; }
.stApp [data-testid="stBaseButton-secondary"] { border: 1.5px solid var(--s2c-ink); background: transparent; }
.stApp [data-testid="stBaseButton-secondary"]:hover:not(:disabled) { border-color: var(--s2c-accent);
  color: var(--s2c-accent); }
.stApp [data-testid="stBaseButton-primary"] p { font-weight: 650; }
.stApp button:disabled { border-style: dashed; }
/* Yeniden üret: dolu mürekkep (sahnenin yanında ikinci bir mavi odak yaratmaz). */
.st-key-generate_controls [data-testid="stBaseButton-secondary"]:not(:disabled) { background: var(--s2c-ink);
  color: var(--s2c-paper); border-color: var(--s2c-ink); }
.st-key-generate_controls [data-testid="stBaseButton-secondary"]:hover:not(:disabled) { background: var(--s2c-accent);
  border-color: var(--s2c-accent); color: #FFFFFF; }

/* Segmentli seçici: ayrık çerçeveli düğmeler, dar sütunda alt satıra geçer; seçili olan dolu mürekkep. */
.stApp [data-testid="stButtonGroup"] [role="radiogroup"] { flex-wrap: wrap; gap: 0.4rem; overflow: visible; }
.stApp button[data-variant="segmented_control"] { border: 1.5px solid var(--s2c-ink) !important;
  border-radius: 4px !important; margin: 0 !important; min-height: 44px; padding: 0 0.6rem; background: transparent; }
.stApp button[data-variant="segmented_control"][data-selected="true"] { background: var(--s2c-ink) !important;
  color: var(--s2c-paper) !important; }

/* Görünür odak: 3 px vurgu halkası, her etkileşimli öğede. */
.stApp :is(button, a, input, textarea, select, summary, [role="radio"], [role="tab"], [tabindex]):focus-visible {
  outline: 3px solid var(--s2c-accent) !important; outline-offset: 2px !important; }

/* Kapalı kenar çubuğu ekran dışında kalır ama öğeleri sekme sırasındaydı; gizlenince sırada yer almaz. */
[data-testid="stSidebar"][data-collapsed="true"] { visibility: hidden; }

/* Yükleme alanı: kesik çizgili kâğıt kenarı. */
.stApp [data-testid="stFileUploaderDropzone"] { border: 1.5px dashed var(--s2c-muted); background: transparent; }

/* Düzeltme isteği: görünür etiket, 52 px mürekkep çerçeveli giriş, metinli "Gönder", role=status satırı. */
.st-key-refine [data-testid="stWidgetLabel"] p { font-size: 1rem; font-weight: 700;
  font-variation-settings: "wdth" 125; }
.st-key-refine [data-testid="stTextInputRootElement"] { min-height: 52px; border: 1.5px solid var(--s2c-ink) !important;
  border-radius: 4px; background: var(--s2c-paper) !important; }
.st-key-refine [data-testid="stTextInputRootElement"]:has(input:disabled) { border-style: dashed !important;
  border-color: var(--s2c-muted) !important; }
.st-key-refine [data-testid="stTextInputRootElement"]:focus-within { outline: 3px solid var(--s2c-accent);
  outline-offset: 2px; }
.st-key-refine input[data-testid="stTextInputField"] { min-height: 49px; padding: 0 1rem; font-size: 1rem;
  background: transparent; color: var(--s2c-ink); outline: none !important; }
.st-key-refine [data-testid="stBaseButton-primaryFormSubmit"] { min-height: 52px; padding: 0 1.4rem; }
.st-key-refine [data-testid="stBaseButton-primaryFormSubmit"] p { font-weight: 700; }
.refine-status { margin: 0.1rem 0 0 !important; font-size: 0.88rem; color: var(--s2c-muted); }
/* Streamlit'in İngilizce "Press Enter to ..." ipucu yer tutucunun üstüne biner; görünür düğme işi anlatır. */
.stApp [data-testid="InputInstructions"] { display: none; }

/* Kutu ve karşılığı: numara rozeti sahnedeki etiketle aynı. */
.box-list { list-style: none; margin: 0 !important; padding: 0 !important; }
.box-list li { display: grid; grid-template-columns: 2.6rem minmax(0, 1fr); gap: 0.6rem; align-items: start;
  margin: 0 !important; padding: 0.55rem 0; border-top: 1px solid var(--s2c-rule); line-height: 1.35; }
.box-list .box-no { display: inline-flex; align-items: center; justify-content: center; height: 1.6rem;
  border-radius: 4px; background: #2A46E8; color: #FFFFFF; font-size: 0.8rem; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.box-list strong { font-weight: 600; }
.box-list small { color: var(--s2c-muted); font-size: 0.85rem; }

/* Sürüm satırları: numara | etiket | saat (silik). Radyo noktası gizli; girdi ve aria-checked yerinde kalır.
   Gösterilen sürümün yalnız etiketi fosforlu kalemle işaretli. */
.st-key-versions [data-testid="stRadioGroup"] { gap: 0 !important; font-variant-numeric: tabular-nums; }
.st-key-versions [data-testid="stRadioGroup"] > div { width: 100%; border-top: 1px solid var(--s2c-rule); }
.st-key-versions [data-testid="stRadioGroup"] > div:last-child { border-bottom: 1px solid var(--s2c-rule); }
.st-key-versions [data-testid="stRadioOption"] { width: 100%; min-height: 44px; margin: 0 !important;
  padding: 0.35rem 0.2rem; align-items: center; }
.st-key-versions [data-testid="stRadioOption"] > div { width: 100%; }
.st-key-versions [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
.st-key-versions [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] { width: 100%; }
.st-key-versions [data-testid="stRadioOption"] p { display: grid; grid-template-columns: 2rem minmax(0, 1fr) auto;
  gap: 0.5rem; align-items: center; margin: 0; font-size: 0.9rem; }
.st-key-versions [data-testid="stRadioOption"] p strong { font-weight: inherit; }
.st-key-versions [data-testid="stRadioOption"] p em { font-style: normal; font-weight: 400; font-size: 0.82rem;
  color: var(--s2c-muted); }
.st-key-versions [data-testid="stRadioOption"][data-selected="true"] p { font-weight: 700; }
.st-key-versions [data-testid="stRadioOption"][data-selected="true"] p strong { justify-self: start;
  background: #FFF176; color: #18203A; padding: 0 0.2rem; box-decoration-break: clone;
  -webkit-box-decoration-break: clone; }
.st-key-versions [data-testid="stRadioOption"]:has(input:focus-visible) { outline: 3px solid var(--s2c-accent);
  outline-offset: -3px; }
.st-key-versions [data-testid="stElementContainer"] { width: 100% !important; }

/* Sahne: çerçevenin dışına taşan gölge yok; iframe tam genişlik. */
.st-key-stage iframe { display: block; }

/* Kod bloğu: söz dizimi renkleri tema tokenlerinden; her biri kod zemininde en az 4,5:1. */
.stApp [data-testid="stCode"] .token { color: var(--s2c-ink) !important; }
.stApp [data-testid="stCode"] .token:is(.comment, .prolog, .doctype, .cdata, .punctuation, .operator) {
  color: var(--s2c-muted) !important; }
.stApp [data-testid="stCode"] .token:is(.tag, .selector, .keyword, .important) {
  color: var(--s2c-code-tag) !important; }
.stApp [data-testid="stCode"] .token:is(.attr-name, .property, .function, .class-name, .variable) {
  color: var(--s2c-code-attr) !important; }
.stApp [data-testid="stCode"] .token:is(.attr-value, .string, .char, .url, .inserted) {
  color: var(--s2c-code-string) !important; }
.stApp [data-testid="stCode"] .token:is(.number, .boolean, .constant, .symbol, .entity, .deleted) {
  color: var(--s2c-code-number) !important; }
.stApp [data-testid="stCode"] .linenumber { color: var(--s2c-muted) !important; }

.palette { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 !important; padding: 0 !important; list-style: none; }
.palette li { margin: 0 !important; flex: 1 1 72px; min-width: 72px; height: 56px; border-radius: 4px; display: flex;
  align-items: flex-end; padding: 6px 8px; font: 500 0.75rem/1 ui-monospace, Consolas, monospace;
  box-shadow: inset 0 0 0 1px rgba(127, 127, 127, 0.35); font-variant-numeric: tabular-nums; }

/* Yükleyicinin İngilizce iç metni: Streamlit yerelleştirme sunmuyor. Özgün metin görünmez ve erişilebilir
   addan çıkar (visibility), yerine Türkçe metin gelir. Ekran okuyucu yalnız Türkçe metni okur. */
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span { visibility: hidden; font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after {
  visibility: visible; content: "Dosya seç"; font-size: 0.875rem; line-height: 1.6; }
[data-testid="stFileUploaderDropzoneInstructions"] span::after {
  visibility: visible; content: "Bırakın ya da seçin, en çok 10 MB"; font-size: 0.875rem; line-height: 1.6; }
.st-key-sketch_files [data-testid="stFileUploaderDropzoneInstructions"] span::after {
  content: "JPG ya da PNG, en çok 10 MB"; }
.st-key-project_file [data-testid="stFileUploaderDropzoneInstructions"] span::after {
  content: "JSON, en çok 10 MB"; }

.st-key-device_grid_narrow { display: none; }
@media (max-width: 640px) {
  .st-key-device_grid { display: none; }
  .st-key-device_grid_narrow { display: block; }
}
/* DOM sırası: sahne, sağ sütun, kutu listesi (dar ekranda da bu sıra; odak sırası görsel sırayla aynı).
   Geniş ekranda kutu listesi sola alınır; listede odaklanan öğe yok, sekme sırası bozulmaz. */
@media (min-width: 641px) {
  .st-key-studio > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(3) { order: -1; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
}
</style>
"""


def is_dark() -> bool:
    """Tarayıcıda etkin Streamlit teması koyu mu? Bilinmiyorsa (ör. AppTest) açık kabul edilir."""
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) == "dark"


def _token_css() -> str:
    tokens = TOKENS["dark" if is_dark() else "light"]
    body = "".join(f"--s2c-{name}:{value};" for name, value in tokens.items())
    return f"<style>:root{{{body}}}</style>"


def inject() -> None:
    st.logo(str(ASSETS / "logo.svg"), size="large")
    st.html(_token_css() + CSS)


def palette_strip(colors: list[str]) -> None:
    """Hex etiketli renk şeridi; etiket rengi zemine göre seçilir (yalnız renge dayanmaz)."""
    items = []
    for color in colors:
        ink = "#111418" if relative_luminance(color) > 0.179 else "#FFFFFF"
        safe = html.escape(color)
        items.append(f'<li style="background:{safe};color:{ink}" title="{safe}">{safe}</li>')
    st.html(f'<ul class="palette" aria-label="Çizimden çıkarılan renkler">{"".join(items)}</ul>')
