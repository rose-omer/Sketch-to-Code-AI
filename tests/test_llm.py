import threading

import pytest
from google.genai import errors
from PIL import Image

from sketch2code import llm
from tests.fakes import PAGE, FakeClient, chunk, split_stream

IMAGE = Image.new("RGB", (8, 8), (255, 255, 255))


def _collect(client: FakeClient, sleeps: list[float] | None = None) -> str:
    sleeps = sleeps if sleeps is not None else []
    return "".join(llm.stream_code(client, IMAGE, "istem", sleep=sleeps.append))


# --- akış ve HTML çıkarma ---------------------------------------------------------


def test_stream_joins_chunks_and_uses_code_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_CODE", raising=False)
    client = FakeClient([split_stream(PAGE)])

    assert _collect(client) == PAGE
    call = client.models.calls[0]
    assert call["model"] == llm.DEFAULT_MODEL_CODE == "gemini-4-hydrogen"
    assert call["contents"][0] is IMAGE
    assert call["config"].system_instruction == llm.prompts.SYSTEM_INSTRUCTION


def test_model_names_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_CODE", "ozel-kod")
    monkeypatch.setenv("MODEL_FAST", "ozel-hizli")

    assert (llm.model_code(), llm.model_fast()) == ("ozel-kod", "ozel-hizli")


@pytest.mark.parametrize(
    "text",
    [
        PAGE,
        f"```html\n{PAGE}\n```",
        f"İşte kodunuz:\n\n{PAGE}\n\nUmarım beğenirsiniz.",
        PAGE.replace("<!DOCTYPE html>", ""),
    ],
    ids=["temiz", "kod-citi", "aciklamali", "doctype-yok"],
)
def test_extract_html_strips_fences_and_chatter(text: str) -> None:
    extracted = llm.extract_html(text)

    assert extracted.endswith("</html>")
    assert "```" not in extracted and "İşte" not in extracted


def test_extract_html_keeps_fences_inside_document() -> None:
    doc = "<!DOCTYPE html><html><body><pre>```kod```</pre></body></html>"

    assert llm.extract_html(doc) == doc


def test_extract_html_without_document_raises() -> None:
    with pytest.raises(llm.GenerationError) as info:
        llm.extract_html("Üzgünüm, bunu yapamam.")

    assert info.value.kind is llm.ErrorKind.NO_HTML


# --- hata eşleme ve yeniden deneme -----------------------------------------------

INVALID_KEY = errors.ClientError(
    400, {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}
)
QUOTA = errors.ClientError(429, {"error": {"code": 429, "message": "exhausted", "status": "RESOURCE_EXHAUSTED"}})
SERVER = errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (INVALID_KEY, llm.ErrorKind.INVALID_KEY),
        (errors.ClientError(403, {"error": {"code": 403, "message": "denied"}}), llm.ErrorKind.INVALID_KEY),
        (QUOTA, llm.ErrorKind.QUOTA),
        (SERVER, llm.ErrorKind.SERVER),
        (ConnectionError("ağ yok"), llm.ErrorKind.NETWORK),
        (errors.ClientError(400, {"error": {"code": 400, "message": "bad image"}}), llm.ErrorKind.BAD_RESPONSE),
    ],
)
def test_map_api_error(exc: Exception, kind: llm.ErrorKind) -> None:
    error = llm.map_api_error(exc)

    assert error.kind is kind
    assert error.user_message  # her tür için Türkçe metin var


def test_invalid_key_message_points_to_ai_studio() -> None:
    assert llm.API_KEY_URL in llm.GenerationError(llm.ErrorKind.INVALID_KEY).user_message


def test_retries_once_on_server_error_before_first_chunk() -> None:
    client = FakeClient([SERVER, [chunk(PAGE)]])
    sleeps: list[float] = []

    assert _collect(client, sleeps) == PAGE
    assert len(client.models.calls) == 2
    assert sleeps == [llm.RETRY_DELAY_S]


def test_gives_up_after_second_failure() -> None:
    client = FakeClient([QUOTA, QUOTA])

    with pytest.raises(llm.GenerationError) as info:
        _collect(client)

    assert info.value.kind is llm.ErrorKind.QUOTA
    assert len(client.models.calls) == 2


def test_no_retry_for_invalid_key() -> None:
    client = FakeClient([INVALID_KEY, [chunk(PAGE)]])

    with pytest.raises(llm.GenerationError) as info:
        _collect(client)

    assert info.value.kind is llm.ErrorKind.INVALID_KEY
    assert len(client.models.calls) == 1


def test_no_retry_after_partial_output() -> None:
    def broken():
        yield chunk("<!DOCTYPE html><html>")
        raise SERVER

    client = FakeClient([broken()])

    with pytest.raises(llm.GenerationError):
        _collect(client)
    assert len(client.models.calls) == 1


@pytest.mark.parametrize(
    "blocked",
    [chunk(block_reason="SAFETY"), chunk(text=None, finish_reason="FinishReason.SAFETY")],
    ids=["istem", "yanit"],
)
def test_blocked_content_raises_blocked(blocked) -> None:
    with pytest.raises(llm.GenerationError) as info:
        _collect(FakeClient([[blocked]]))

    assert info.value.kind is llm.ErrorKind.BLOCKED


# --- öneriler ---------------------------------------------------------------------


def test_suggestions_are_validated_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_FAST", raising=False)
    body = '{"items": [{"title": "Alt bilgi", "detail": "İletişim bilgisi ekleyin."}]}'
    client = FakeClient(json_text=body)

    result = llm.suggest(client, IMAGE)

    assert result.items[0].title == "Alt bilgi"
    call = client.models.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == llm.Suggestions.model_json_schema()


def test_suggestions_with_bad_json_raise_bad_response() -> None:
    with pytest.raises(llm.GenerationError) as info:
        llm.suggest(FakeClient(json_text="liste: yok"), IMAGE)

    assert info.value.kind is llm.ErrorKind.BAD_RESPONSE


# --- anahtar ve istemci ---------------------------------------------------------------


def test_key_order_secrets_then_env_then_input() -> None:
    assert llm.resolve_api_key({"GEMINI_API_KEY": "s"}, {"GEMINI_API_KEY": "e"}, "i") == ("s", "secrets")
    assert llm.resolve_api_key({}, {"GEMINI_API_KEY": "e"}, "i") == ("e", "env")
    assert llm.resolve_api_key(None, {}, "  i  ") == ("i", "input")
    assert llm.resolve_api_key(None, {}, "") == ("", "")


def test_each_session_client_keeps_its_own_key() -> None:
    first, second = llm.make_client("test-key-a"), llm.make_client("test-key-b")

    assert first._api_client.api_key == "test-key-a"
    assert second._api_client.api_key == "test-key-b"


def test_concurrent_generations_use_their_own_client() -> None:
    clients = {name: FakeClient([split_stream(PAGE)] * 1, api_key=name) for name in ("sekme-a", "sekme-b")}
    barrier = threading.Barrier(2)
    outputs: dict[str, str] = {}

    def run(name: str) -> None:
        barrier.wait()
        outputs[name] = "".join(llm.stream_code(clients[name], IMAGE, name, sleep=lambda _: None))

    threads = [threading.Thread(target=run, args=(name,)) for name in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for name, client in clients.items():
        assert outputs[name] == PAGE
        assert [call["key"] for call in client.models.calls] == [name]
        assert client.models.calls[0]["contents"][1] == name
