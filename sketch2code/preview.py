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

import html
import re
from dataclasses import dataclass

SANDBOX = "allow-scripts"
PREVIEW_TITLE = "Üretilen sayfa önizlemesi"

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


def sandboxed(html_code: str, width: int, height: int, scale: float | None = None) -> str:
    """Güvenli önizleme sarmalayıcısını döndürür.

    `scale=None` ise sarmalayıcı, iç iframe'i bulunduğu genişliğe sığacak şekilde ölçekler
    (en fazla 1). Sabit bir `scale` verilirse betik çalışmaz.
    """
    srcdoc = html.escape(with_csp(html_code), quote=True)
    fixed = scale if scale is not None else 1.0
    inner = (
        f'<iframe id="frame" title="{PREVIEW_TITLE}" sandbox="{SANDBOX}" referrerpolicy="no-referrer" '
        f'loading="lazy" width="{width}" height="{height}" '
        f'style="width:{width}px;height:{height}px;transform:scale({fixed});transform-origin:0 0;'
        f'border:0;display:block;background:#fff" srcdoc="{srcdoc}"></iframe>'
    )
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
        '<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">'
        "<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden}"
        "#box{overflow:hidden;border-radius:6px;box-shadow:0 0 0 1px rgba(127,127,127,.35)}</style>"
        f'</head><body><div id="box" style="{box_size}">{inner}</div>{fit_script}</body></html>'
    )
