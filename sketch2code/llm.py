"""Gemini istemcisi (google-genai).

- Süreç geneli `genai.configure` yok: her oturum kendi `genai.Client` nesnesini taşır.
- Model adları ortam değişkeniyle değişir: `MODEL_CODE`, `MODEL_FAST`.
- Hatalar `GenerationError` olarak yükselir; `user_message` arayüzde gösterilecek Türkçe metindir.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from typing import Any, Protocol

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from sketch2code import prompts

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CODE = "gemini-4-hydrogen"
DEFAULT_MODEL_FAST = "gemini-3.5-flash-lite"
API_KEY_URL = "https://aistudio.google.com/apikey"
RETRY_DELAY_S = 1.5


def model_code() -> str:
    return os.environ.get("MODEL_CODE", "").strip() or DEFAULT_MODEL_CODE


def model_fast() -> str:
    return os.environ.get("MODEL_FAST", "").strip() or DEFAULT_MODEL_FAST


# --- Hatalar ----------------------------------------------------------------


class ErrorKind(StrEnum):
    INVALID_KEY = "invalid_key"
    QUOTA = "quota"
    BLOCKED = "blocked"
    NO_HTML = "no_html"
    SERVER = "server"
    NETWORK = "network"
    BAD_RESPONSE = "bad_response"


USER_MESSAGES: dict[ErrorKind, str] = {
    ErrorKind.INVALID_KEY: f"API anahtarı geçersiz ya da yetkisiz. Yeni anahtarı {API_KEY_URL} adresinden alın.",
    ErrorKind.QUOTA: "Kota doldu. Bir dakika bekleyip yeniden deneyin ya da AI Studio'da kota durumuna bakın.",
    ErrorKind.BLOCKED: "Model bu çizimi güvenlik filtresine takıldığı için işlemedi. Çizimi ya da isteği değiştirin.",
    ErrorKind.NO_HTML: "Model yanıtında HTML belgesi yok. Yeniden üretin ya da isteği sadeleştirin.",
    ErrorKind.SERVER: "Gemini sunucusu yanıt vermedi. Biraz sonra yeniden deneyin.",
    ErrorKind.NETWORK: "Gemini'ye bağlanılamadı. İnternet bağlantısını denetleyin.",
    ErrorKind.BAD_RESPONSE: "Model yanıtı beklenen biçimde değil. Yeniden deneyin.",
}


class GenerationError(Exception):
    """Kullanıcıya gösterilebilir üretim hatası."""

    def __init__(self, kind: ErrorKind, detail: str = "") -> None:
        super().__init__(f"{kind.value}: {detail}" if detail else kind.value)
        self.kind = kind
        self.detail = detail

    @property
    def user_message(self) -> str:
        return USER_MESSAGES[self.kind]

    @property
    def retriable(self) -> bool:
        return self.kind in (ErrorKind.QUOTA, ErrorKind.SERVER, ErrorKind.NETWORK)


def map_api_error(exc: Exception) -> GenerationError:
    """google.genai hatasını (ya da ağ hatasını) `GenerationError`'a çevirir."""
    code = getattr(exc, "code", None)
    text = f"{getattr(exc, 'status', '') or ''} {getattr(exc, 'message', '') or exc}"
    if code in (401, 403) or "API_KEY" in text.upper() or (code == 400 and "key" in text.lower()):
        return GenerationError(ErrorKind.INVALID_KEY, text)
    if code == 429:
        return GenerationError(ErrorKind.QUOTA, text)
    if isinstance(code, int) and code >= 500:
        return GenerationError(ErrorKind.SERVER, text)
    if code is None and isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return GenerationError(ErrorKind.NETWORK, str(exc))
    return GenerationError(ErrorKind.BAD_RESPONSE, text)


# --- İstemci ------------------------------------------------------------------


class SupportsModels(Protocol):
    models: Any


def resolve_api_key(secrets: Mapping[str, Any] | None, env: Mapping[str, str], user_input: str) -> tuple[str, str]:
    """Anahtar sırası: st.secrets → ortam değişkeni → kenar çubuğu (BYOK).

    Dönen değer (anahtar, kaynak). Anahtar yoksa ("", "").
    """
    if secrets:
        value = str(secrets.get("GEMINI_API_KEY", "") or "").strip()
        if value:
            return value, "secrets"
    value = env.get("GEMINI_API_KEY", "").strip()
    if value:
        return value, "env"
    value = user_input.strip()
    return (value, "input") if value else ("", "")


def make_client(api_key: str) -> SupportsModels:
    """Yeni bir `genai.Client`. Oturum başına bir kez çağrılır."""
    from google import genai  # ağır içe aktarma; testler sahte istemci kullanır

    return genai.Client(api_key=api_key)


# --- Yanıt işleme -----------------------------------------------------------------

_DOC_RE = re.compile(r"<!DOCTYPE\s+html.*?</html\s*>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<html[\s>].*?</html\s*>", re.IGNORECASE | re.DOTALL)


def extract_html(text: str) -> str:
    """Yanıttan tam HTML belgesini çıkarır; çit ve açıklama metnini atar."""
    match = _DOC_RE.search(text) or _HTML_RE.search(text)
    if not match:
        raise GenerationError(ErrorKind.NO_HTML, text[:200])
    return match.group(0).strip()


def _check_blocked(chunk: Any) -> None:
    feedback = getattr(chunk, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise GenerationError(ErrorKind.BLOCKED, str(feedback.block_reason))
    for candidate in getattr(chunk, "candidates", None) or []:
        reason = str(getattr(candidate, "finish_reason", "") or "")
        if reason.upper().endswith(("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST")):
            raise GenerationError(ErrorKind.BLOCKED, reason)


def _stream_with_retry(open_stream: Callable[[], Iterator[Any]], sleep: Callable[[float], None]) -> Iterator[str]:
    """Akışı açar; ilk parça gelmeden 429/5xx/ağ hatası olursa bir kez yeniden dener."""
    for attempt in (1, 2):
        started = False
        try:
            for chunk in open_stream():
                _check_blocked(chunk)
                text = getattr(chunk, "text", None)
                if text:
                    started = True
                    yield text
            return
        except GenerationError:
            raise
        except Exception as exc:
            error = map_api_error(exc)
            if started or attempt == 2 or not error.retriable:
                logger.warning("Gemini akışı başarısız: %s", error)
                raise error from exc
            logger.info("Gemini geçici hata, yeniden deneniyor: %s", error)
            sleep(RETRY_DELAY_S)


def _config(system_instruction: str, **extra: Any) -> Any:
    from google.genai import types

    return types.GenerateContentConfig(system_instruction=system_instruction, **extra)


def stream_code(
    client: SupportsModels,
    image: Image.Image,
    prompt: str,
    *,
    model: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Kod üretimini metin parçaları olarak akıtır."""
    config = _config(prompts.SYSTEM_INSTRUCTION, temperature=0.4, max_output_tokens=16384)

    def open_stream() -> Iterator[Any]:
        return client.models.generate_content_stream(
            model=model or model_code(), contents=[image, prompt], config=config
        )

    return _stream_with_retry(open_stream, sleep)


# --- Öneriler -------------------------------------------------------------------


class Suggestion(BaseModel):
    title: str = Field(max_length=80)
    detail: str = Field(max_length=300)


class Suggestions(BaseModel):
    items: list[Suggestion] = Field(max_length=4)


def suggest(client: SupportsModels, image: Image.Image, *, model: str | None = None) -> Suggestions:
    """Hızlı modelden yapılandırılmış (JSON) öneri listesi alır."""
    config = _config(
        "Kısa ve uygulanabilir tasarım önerileri veren bir arayüz uzmanısın.",
        temperature=0.3,
        response_mime_type="application/json",
        response_json_schema=Suggestions.model_json_schema(),
    )
    try:
        response = client.models.generate_content(
            model=model or model_fast(), contents=[image, prompts.SUGGESTION_PROMPT], config=config
        )
    except Exception as exc:
        raise map_api_error(exc) from exc
    _check_blocked(response)
    try:
        return Suggestions.model_validate_json(response.text or "")
    except ValidationError as exc:
        raise GenerationError(ErrorKind.BAD_RESPONSE, str(exc)) from exc
