"""Dışa aktarma: tek HTML dosyası, ZIP paketi ve proje dosyası (.json)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any

ZIP_FILES = ("index.html", "README.md")
PROJECT_FORMAT = "sketch2code-project/1"


def zip_readme(name: str, created: datetime) -> str:
    return (
        f"# {name}\n\n"
        "Bu sayfa Sketch-to-Code ile bir kâğıt çiziminden üretildi.\n\n"
        "## İçerik\n"
        "- `index.html`: sayfanın tamamı. CSS ve JavaScript belgenin içindedir.\n\n"
        "## Kullanım\n"
        "1. `index.html` dosyasını tarayıcıda açın.\n"
        "2. Stil ve betikleri `index.html` içindeki `<style>` ve `<script>` bölümlerinde düzenleyin.\n\n"
        f"Oluşturma: {created:%Y-%m-%d %H:%M}\n"
    )


def build_zip(html_code: str, name: str = "sayfa", created: datetime | None = None) -> bytes:
    """`index.html` + `README.md` içeren ZIP baytları."""
    created = created or datetime.now()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/index.html", html_code)
        archive.writestr(f"{name}/README.md", zip_readme(name, created))
    return buffer.getvalue()


class ProjectFileError(ValueError):
    """Yüklenen proje dosyası okunamadı."""


def dump_project(pages: dict[str, dict[str, Any]]) -> bytes:
    """Sayfa sürümlerini indirilebilir JSON'a çevirir (görüntü baytları dahil değildir)."""
    payload = {
        "format": PROJECT_FORMAT,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "pages": [
            {"id": page_id, "name": page["name"], "versions": page["versions"], "current": page["current"]}
            for page_id, page in pages.items()
            if page.get("versions")
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_project(data: bytes) -> dict[str, dict[str, Any]]:
    """`dump_project` çıktısını geri okur; biçim yanlışsa `ProjectFileError`."""
    try:
        payload = json.loads(data.decode("utf-8"))
        if payload.get("format") != PROJECT_FORMAT:
            raise ProjectFileError("Bu dosya bir Sketch-to-Code proje dosyası değil.")
        pages: dict[str, dict[str, Any]] = {}
        for page in payload["pages"]:
            versions = [
                {"html": str(v["html"]), "label": str(v.get("label", "")), "created": str(v.get("created", ""))}
                for v in page["versions"]
            ]
            current = min(max(int(page.get("current", len(versions) - 1)), 0), len(versions) - 1)
            pages[str(page["id"])] = {"name": str(page["name"]), "versions": versions, "current": current}
        return pages
    except ProjectFileError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ProjectFileError("Proje dosyası bozuk ya da eksik.") from exc
