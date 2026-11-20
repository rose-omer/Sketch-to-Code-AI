"""Üretilen HTML için güvenli, gerçek genişlikli önizleme.

Model çıktısı güvenilmezdir. Bu yüzden iki katman kullanılır:
- Dış sarmalayıcı (bizim yazdığımız, güvenilir) yalnız bir iç iframe ve ölçek betiği içerir.
- İç iframe `sandbox="allow-scripts"` ile açılır; `allow-same-origin` yoktur. Belge opak
  kökende çalışır: üst sayfaya, çerezlere ve Streamlit oturumuna erişemez; üst pencereyi
  yönlendiremez, form gönderemez, açılır pencere açamaz.
- İç belgenin başına CSP eklenir: ağ isteği yok, betik ve stil yalnız izinli CDN'lerden.

İç iframe cihazın gerçek genişliğinde (ör. 375 px) açılır ve CSS `transform: scale()` ile
sütuna sığdırılır; bu sayede medya sorguları gerçek kırılım noktasında tetiklenir.
"""

from __future__ import annotations

import base64
import html
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from sketch2code.vision import Box

SANDBOX = "allow-scripts"
PREVIEW_TITLE = "Üretilen sayfa önizlemesi"
COMPARE_TITLE = "Önce ve sonra karşılaştırması"
WRAPPER_TITLE = "Önizleme çerçevesi"

CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
    "style-src 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com https://cdn.jsdelivr.net data:; "
    "img-src data: blob:; media-src data:; connect-src 'none'; "
    "form-action 'none'; base-uri 'none'"
)


@dataclass(frozen=True)
class Device:
    key: str
    label: str
    width: int
    height: int


DEVICES: dict[str, Device] = {
    "desktop": Device("desktop", "Masaüstü", 1280, 800),
    "tablet": Device("tablet", "Tablet", 768, 1024),
    "mobile": Device("mobile", "Mobil", 375, 812),
}

# Sırayla denenir: <head>, <html>, <!DOCTYPE>. Meta bulunan ilk açılışın hemen ardına girer;
# böylece doctype ilk sırada kalır ve belge quirks moduna düşmez.
_ANCHORS = (
    re.compile(r"<head(\s[^>]*)?>", re.IGNORECASE),
    re.compile(r"<html(\s[^>]*)?>", re.IGNORECASE),
    re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE),
)


def with_csp(html_code: str) -> str:
    """CSP meta etiketini <head> başına ekler; <head> yoksa <html> ya da doctype ardına, o da yoksa başa."""
    meta = f'<meta http-equiv="Content-Security-Policy" content="{CSP}">'
    for anchor in _ANCHORS:
        match = anchor.search(html_code)
        if match:
            return html_code[: match.end()] + meta + html_code[match.end() :]
    return meta + html_code


def _inner_frame(html_code: str, width: int, height: int, scale: float) -> str:
    """Yalıtılmış iç iframe: gerçek cihaz genişliğinde, `transform: scale()` ile küçültülür."""
    srcdoc = html.escape(with_csp(html_code), quote=True)
    return (
        f'<iframe id="frame" title="{PREVIEW_TITLE}" sandbox="{SANDBOX}" referrerpolicy="no-referrer" '
        f'loading="lazy" width="{width}" height="{height}" '
        f'style="width:{width}px;height:{height}px;transform:scale({scale});transform-origin:0 0;'
        f'border:0;display:block;background:#fff" srcdoc="{srcdoc}"></iframe>'
    )


def sandboxed(html_code: str, width: int, height: int, scale: float | None = None) -> str:
    """Güvenli önizleme sarmalayıcısını döndürür.

    `scale=None` ise sarmalayıcı, iç iframe'i bulunduğu genişliğe sığacak şekilde ölçekler
    (en fazla 1). Sabit bir `scale` verilirse betik çalışmaz.
    """
    fixed = scale if scale is not None else 1.0
    inner = _inner_frame(html_code, width, height, fixed)
    fit_script = (
        "<script>(function(){var f=document.getElementById('frame'),b=document.getElementById('box');"
        f"var w={width},h={height};function fit(){{var s=Math.min(1,document.documentElement.clientWidth/w);"
        "f.style.transform='scale('+s+')';b.style.height=Math.ceil(h*s)+'px';b.style.width=Math.ceil(w*s)+'px';}"
        "fit();window.addEventListener('resize',fit);})();</script>"
        if scale is None
        else ""
    )
    box_size = f"width:{round(width * fixed)}px;height:{round(height * fixed)}px"
    return (
        f'<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8"><title>{WRAPPER_TITLE}</title>'
        "<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden}"
        "#box{overflow:hidden;border-radius:4px;box-shadow:0 0 0 1px rgba(127,127,127,.45)}</style>"
        f'</head><body><div id="box" style="{box_size}">{inner}</div>{fit_script}</body></html>'
    )


# --- Önce ve sonra -----------------------------------------------------------
# Sarmalayıcı bizim kodumuzdur: çizim görüntüsü (data: URL), numaralı kutular ve kaydırıcı onun
# içindedir. Üretilen HTML yine yalnız iç iframe'de, aynı sandbox ve CSP ile çalışır. Kaydırıcı
# yalnız çizim katmanının `clip-path` değerini değiştirir; iç belgeye dokunmaz.

COMPARE_RATIO = (0.45, 2.5)  # çizimin yükseklik/genişlik oranı bu aralığa kırpılır

_FONT_FACES = "".join(
    f'@font-face{{font-family:"Archivo Variable";src:url(app/static/fonts/archivo-{subset}-wdth-normal.woff2) '
    f'format("woff2");font-weight:100 900;font-stretch:62% 125%;font-display:swap;unicode-range:{ranges}}}'
    for subset, ranges in (
        ("latin", "U+0000-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,U+2000-206F,U+20AC,U+2122,U+2212"),
        ("latin-ext", "U+0100-0130,U+0132-0151,U+0154-02BA,U+1E00-1EFF,U+20A0-20AB,U+20AD-20C0"),
    )
)

_COMPARE_CSS = (
    _FONT_FACES + ":root{--ink:#18203a;--muted:#4d5670;--accent:#2a46e8;--paper:#fff}"
    ":root[data-theme=dark]{--ink:#e6e9f2;--muted:#a3aac0;--accent:#6f86ff;--paper:#121831}"
    "html,body{margin:0;padding:0;background:var(--paper);overflow:hidden}"
    'body{margin:0 auto;color:var(--ink);font:400 16px/1.5 "Archivo Variable",system-ui,sans-serif}'
    ".heads{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin:0 0 .55rem;"
    "font-size:1.05rem;font-weight:700;font-stretch:125%;letter-spacing:-.005em}"
    ".heads span:last-child{text-align:right}"
    "#box{position:relative;overflow:hidden;background:#fff;box-shadow:0 0 0 1.5px var(--ink)}"
    "#frame{position:absolute;top:0;left:0}"
    "#before{position:absolute;inset:0;background:#fff;clip-path:inset(0 50% 0 0)}"
    "#before img{display:block;width:100%;height:100%}"
    ".mark{position:absolute;box-sizing:border-box;border:2px solid #2a46e8;background:rgba(42,70,232,.06)}"
    ".mark b{position:absolute;top:0;left:0;min-width:1.35rem;padding:0 .3rem;box-sizing:border-box;"
    "background:#2a46e8;color:#fff;font-size:.72rem;line-height:1.25rem;font-weight:600;text-align:center;"
    "font-variant-numeric:tabular-nums;border-radius:0 0 3px 0}"
    "#line{position:absolute;top:0;bottom:0;left:50%;width:3px;margin-left:-1.5px;background:#2a46e8;"
    "pointer-events:none}"
    "#handle{position:absolute;top:50%;left:50%;width:52px;height:52px;margin:-26px 0 0 -26px;display:flex;"
    "align-items:center;justify-content:center;box-sizing:border-box;background:#2a46e8;border:3px solid #fff;"
    "border-radius:50%;box-shadow:0 1px 6px rgba(24,32,58,.4);cursor:ew-resize;touch-action:none}"
    ".range{display:flex;align-items:center;gap:1rem;margin:.6rem 0 0;font-size:.875rem;color:var(--muted)}"
    ".range input{flex:1;min-width:0;min-height:44px;margin:0;accent-color:var(--accent);cursor:pointer}"
    ".range input:focus-visible{outline:3px solid var(--accent);outline-offset:2px;border-radius:4px}"
)

# %(w)d ve %(h)d sahnenin gerçek boyutudur; '%%' yüzde işaretidir.
_COMPARE_SCRIPT = """<script>(function(){
var f=document.getElementById('frame'),box=document.getElementById('box'),before=document.getElementById('before'),
line=document.getElementById('line'),handle=document.getElementById('handle'),range=document.getElementById('split');
var w=%(w)d,h=%(h)d;
function fit(){var s=Math.min(1,document.documentElement.clientWidth/w);f.style.transform='scale('+s+')';
box.style.width=document.body.style.width=Math.floor(w*s)+'px';box.style.height=Math.floor(h*s)+'px';}
function set(p){p=Math.round(Math.max(0,Math.min(100,p)));before.style.clipPath='inset(0 '+(100-p)+'%% 0 0)';
line.style.left=handle.style.left=p+'%%';range.value=p;
range.setAttribute('aria-valuetext','Çizim yüzde '+p+', sayfa yüzde '+(100-p));}
function follow(e){var r=box.getBoundingClientRect();set((e.clientX-r.left)/r.width*100);}
range.addEventListener('input',function(){set(+range.value);});
handle.addEventListener('pointerdown',function(e){handle.setPointerCapture(e.pointerId);follow(e);e.preventDefault();});
handle.addEventListener('pointermove',function(e){if(handle.hasPointerCapture(e.pointerId))follow(e);});
fit();set(50);window.addEventListener('resize',fit);})();</script>"""

_HANDLE_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4v16"/>'
    '<path d="M8 8l-4 4 4 4"/><path d="M16 8l4 4-4 4"/></svg>'
)


def image_data_url(rgb: np.ndarray, max_width: int = 1000) -> str:
    """RGB diziyi küçültüp JPEG data: URL'ye çevirir. Sarmalayıcıya gömülür; ağ isteği yok."""
    image = Image.fromarray(rgb).convert("RGB")
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def compare(
    html_code: str,
    before_url: str,
    before_ratio: float,
    boxes: Sequence[Box],
    width: int,
    after_label: str,
    dark: bool = False,
) -> str:
    """Önce (düzeltilmiş çizim + numaralı kutular) ve sonra (sandbox önizleme) karşılaştırması.

    Sahne çizimin en-boy oranındadır. İç iframe cihazın gerçek genişliğinde açılır ve sahneye
    ölçeklenir; böylece çizimdeki kutu ile sayfadaki karşılığı aynı yerde üst üste gelir.
    """
    ratio = min(max(before_ratio, COMPARE_RATIO[0]), COMPARE_RATIO[1])
    height = round(width * ratio)
    marks = "".join(
        f'<div class="mark" style="left:{b.x * 100:.2f}%;top:{b.y * 100:.2f}%;'
        f'width:{b.w * 100:.2f}%;height:{b.h * 100:.2f}%"><b>{i}</b></div>'
        for i, b in enumerate(boxes, start=1)
    )
    stage = (
        f'<div id="box" style="width:{width}px;height:{height}px">'
        f"{_inner_frame(html_code, width, height, 1.0)}"
        f'<div id="before"><img src="{html.escape(before_url, quote=True)}" '
        f'alt="Önce: düzeltilmiş çizim ve {len(boxes)} kutu">{marks}</div>'
        f'<div id="line" aria-hidden="true"></div><div id="handle" aria-hidden="true">{_HANDLE_ICON}</div></div>'
    )
    controls = (
        '<label class="range"><span>Karşılaştırma çizgisi</span>'
        '<input id="split" type="range" min="0" max="100" step="1" value="50"></label>'
    )
    heads = f'<div class="heads"><span>Önce: çizim</span><span>Sonra: {html.escape(after_label)}</span></div>'
    theme = "dark" if dark else "light"
    script = _COMPARE_SCRIPT % {"w": width, "h": height}
    return (
        f'<!DOCTYPE html><html lang="tr" data-theme="{theme}"><head><meta charset="utf-8">'
        f"<title>{COMPARE_TITLE}</title><style>{_COMPARE_CSS}</style></head><body>{heads}{stage}{controls}{script}</body></html>"
    )
