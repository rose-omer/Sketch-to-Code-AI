"""Sayfa başına sürüm geçmişi (saf veri; Streamlit oturum durumunda saklanır).

Sayfa sözlüğü: {"name": str, "versions": [{"html", "label", "created"}], "current": int}
"""

from __future__ import annotations

import difflib
import hashlib
from datetime import datetime
from typing import Any

MAX_VERSIONS = 30

Page = dict[str, Any]


def page_id(data: bytes) -> str:
    """Dosya baytlarından kısa, kararlı kimlik."""
    return hashlib.sha256(data).hexdigest()[:16]


def new_page(name: str) -> Page:
    return {"name": name, "versions": [], "current": -1}


def add_version(page: Page, html_code: str, label: str, now: datetime | None = None) -> Page:
    """Yeni sürümü sona ekler ve onu güncel yapar. Geri alınmış sürümlerin üstüne yazmaz."""
    version = {"html": html_code, "label": label, "created": f"{(now or datetime.now()):%H:%M:%S}"}
    versions = [*page["versions"], version][-MAX_VERSIONS:]
    return {**page, "versions": versions, "current": len(versions) - 1}


def current_html(page: Page) -> str | None:
    if page["current"] < 0 or not page["versions"]:
        return None
    return page["versions"][page["current"]]["html"]


def select(page: Page, index: int) -> Page:
    if not 0 <= index < len(page["versions"]):
        raise IndexError(f"Sürüm yok: {index}")
    return {**page, "current": index}


def undo(page: Page) -> Page:
    """Bir önceki sürüme döner; ilk sürümdeyse değişmez."""
    return {**page, "current": max(0, page["current"] - 1)} if page["versions"] else page


def diff(page: Page, a: int, b: int) -> str:
    """İki sürüm arasındaki birleşik fark (unified diff)."""
    old, new = page["versions"][a], page["versions"][b]
    lines = difflib.unified_diff(
        old["html"].splitlines(),
        new["html"].splitlines(),
        fromfile=f"v{a + 1} {old['label']}",
        tofile=f"v{b + 1} {new['label']}",
        lineterm="",
    )
    return "\n".join(lines)
