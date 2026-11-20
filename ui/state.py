"""Oturum durumu ve oturum başına Gemini istemcisi."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from sketch2code import llm, palette, prompts, quota, samples, vision

SAMPLE_ID = "ornek"


@dataclass
class Analysis:
    result: vision.ProcessResult
    colors: list[str]

    def image(self) -> Image.Image:
        """Modele giden görüntü: düzeltilmiş, renkli."""
        return Image.fromarray(self.result.corrected)


@dataclass
class Settings:
    api_key: str
    key_source: str
    options: prompts.GenerationOptions
    model_code: str
    model_fast: str


def init_state() -> None:
    defaults: dict[str, Any] = {
        "pages": {},  # page_id -> history.Page
        "sources": {},  # page_id -> (ad, bayt)
        "active_page": None,
        "crops": {},  # page_id -> (sol, sağ, üst, alt)
        "suggestions": {},
        "notices": {},  # page_id -> son üretim hatası; sonraki üretime kadar görünür
        "client": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def secrets_mapping() -> dict[str, Any]:
    """st.secrets'i güvenli okur; secrets.toml yoksa boş sözlük."""
    try:
        return {"GEMINI_API_KEY": st.secrets.get("GEMINI_API_KEY", "")}
    except Exception:  # noqa: BLE001 — dosya yok ya da bozuk: BYOK'a düş
        return {}


def env() -> dict[str, str]:
    return dict(os.environ)


def get_client(api_key: str) -> llm.SupportsModels:
    """Oturum başına istemci; anahtar değişince yenisi kurulur. Anahtar düz metin saklanmaz."""
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    cached = st.session_state.client
    if cached is None or cached[0] != fingerprint:
        st.session_state.client = (fingerprint, llm.make_client(api_key))
    return st.session_state.client[1]


@st.cache_resource
def _quota(limit: int) -> quota.DailyQuota:
    return quota.DailyQuota(limit)


def _visitor() -> str:
    """Ziyaretçi kimliği: IP özeti (IP saklanmaz). IP yoksa (yerel çalıştırma) tek ortak kimlik."""
    ip = getattr(st.context, "ip_address", None)
    if not isinstance(ip, str) or not ip:  # AppTest ve yerel çalıştırmada IP yok
        ip = "yerel"
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _owner_quota(settings: Settings) -> quota.DailyQuota | None:
    """Sahibin anahtarı kullanılıyor ve DAILY_GENERATION_LIMIT tanımlıysa sayaç; yoksa None (sınırsız)."""
    limit = quota.limit_from_env()
    if not limit or settings.key_source not in ("secrets", "env"):
        return None
    return _quota(limit)


def generations_left(settings: Settings) -> int | None:
    counter = _owner_quota(settings)
    return None if counter is None else counter.remaining(_visitor())


def take_generation(settings: Settings) -> bool:
    counter = _owner_quota(settings)
    return counter is None or counter.consume(_visitor())


@st.cache_data(show_spinner=False)
def sample_bytes() -> bytes:
    buffer = io.BytesIO()
    samples.sample_photo().save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(max_entries=12, show_spinner=False)
def analyse(data: bytes, crop: tuple[float, float, float, float] | None) -> Analysis:
    """Görüntü hattı + palet. Anahtar: dosya baytları ve kırpma; her etkileşimde yeniden hesaplanmaz."""
    image = vision.load_image(data)
    if crop is None:
        result = vision.process(image)
    else:
        cropped = vision.crop_fraction(np.asarray(image), *crop)
        result = vision.process(Image.fromarray(cropped), detect_paper=False)
    return Analysis(result=result, colors=palette.extract_palette(result.corrected))
