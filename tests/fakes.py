"""Gemini istemcisi yerine geçen sahte nesneler. Testler hiçbir zaman gerçek API'ye gitmez."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

PAGE = (
    '<!DOCTYPE html><html lang="tr"><head><title>Deneme</title>'
    "<style>@media (max-width: 640px){body{background:rgb(1,2,3)}}</style></head>"
    "<body><h1>Merhaba</h1></body></html>"
)


def chunk(text: str | None = None, block_reason: str | None = None, finish_reason: str | None = None) -> Any:
    feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    candidates = [SimpleNamespace(finish_reason=finish_reason)] if finish_reason else None
    return SimpleNamespace(text=text, prompt_feedback=feedback, candidates=candidates)


class FakeModels:
    """`client.models` taklidi. `streams` her çağrıda sıradaki davranışı kullanır."""

    def __init__(self, streams: list[Any] | None = None, json_text: str = "", api_key: str = "") -> None:
        self.streams = list(streams or [])
        self.json_text = json_text
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, *, model: str, contents: list[Any], config: Any) -> Iterator[Any]:
        self.calls.append({"model": model, "contents": contents, "config": config, "key": self.api_key})
        behaviour = self.streams.pop(0) if self.streams else [chunk(PAGE)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return iter(behaviour)

    def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config, "key": self.api_key})
        return chunk(self.json_text)


class FakeClient:
    def __init__(self, streams: list[Any] | None = None, json_text: str = "", api_key: str = "") -> None:
        self.models = FakeModels(streams, json_text, api_key)


def split_stream(text: str, size: int = 37) -> list[Any]:
    return [chunk(text[i : i + size]) for i in range(0, len(text), size)]
