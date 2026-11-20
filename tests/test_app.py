"""Arayüz akışı: örnek çizim → üret (sahte LLM) → sandbox önizleme → sohbetle düzelt → geri al."""

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.fakes import PAGE, FakeClient, split_stream

APP = Path(__file__).resolve().parents[1] / "app.py"
REFINED = PAGE.replace("Merhaba", "Ortalanmış başlık")


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return AppTest.from_file(str(APP), default_timeout=60)


def test_empty_state_without_key(app: AppTest) -> None:
    app.run()

    assert not app.exception
    assert app.title[0].value == "Sketch-to-Code"
    assert any("İyi sonuç için fotoğraf" in m.value for m in app.markdown)


def test_generate_is_disabled_without_key(app: AppTest) -> None:
    app.run()
    _button(app, "Örnekle dene").click().run()

    assert not app.exception
    assert _button(app, "Kodu üret").disabled
    assert any("API anahtarı girin" in c.value for c in app.caption)


def test_generate_preview_refine_and_undo(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = FakeClient([split_stream(PAGE), split_stream(f"Tamam:\n```html\n{REFINED}\n```")])

    with patch("sketch2code.llm.make_client", return_value=client) as make_client:
        app.run()
        _button(app, "Örnekle dene").click().run()
        _button(app, "Kodu üret").click().run()

        assert not app.exception
        page = app.session_state["pages"]["ornek"]
        assert [v["html"] for v in page["versions"]] == [PAGE]

        frames = app.get("iframe")
        assert len(frames) == 1
        proto = str(frames[0].proto)
        assert 'sandbox=\\"allow-scripts\\"' in proto
        assert "allow-same-origin" not in proto

        app.chat_input[0].set_value("başlığı ortala").run()
        page = app.session_state["pages"]["ornek"]
        assert page["current"] == 1
        assert page["versions"][1]["html"] == REFINED

        _button(app, "Geri al").click().run()
        assert app.session_state["pages"]["ornek"]["current"] == 0

    make_client.assert_called_once_with("test-key")  # oturum başına tek istemci
    first_call, refine_call = client.models.calls
    assert first_call["model"] == "gemini-4-hydrogen"
    assert PAGE in refine_call["contents"][1] and "başlığı ortala" in refine_call["contents"][1]


def test_generation_error_is_shown_and_no_version_is_saved(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    denied = errors.ClientError(403, {"error": {"code": 403, "message": "denied"}})

    with patch("sketch2code.llm.make_client", return_value=FakeClient([denied])):
        app.run()
        _button(app, "Örnekle dene").click().run()
        _button(app, "Kodu üret").click().run()

    assert not app.exception
    assert any("aistudio.google.com/apikey" in e.value for e in app.error)
    assert app.session_state["pages"]["ornek"]["versions"] == []


def _toggle(at: AppTest, label: str):
    return next(t for t in at.toggle if t.label == label)


def test_variant_failure_message_survives_rerun(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    denied = errors.ClientError(403, {"error": {"code": 403, "message": "denied"}})

    with patch("sketch2code.llm.make_client", return_value=FakeClient([split_stream(PAGE), denied])):
        app.run()
        _button(app, "Örnekle dene").click().run()
        _toggle(app, "3 varyant üret (sade, klasik, cesur)").set_value(True).run()
        _button(app, "Kodu üret").click().run()

        assert not app.exception
        assert len(app.session_state["pages"]["ornek"]["versions"]) == 1
        messages = [e.value for e in app.error]
        assert any("1/3 varyant üretildi" in m and "aistudio.google.com/apikey" in m for m in messages)

        app.run()  # sonraki etkileşim: bildirim bir sonraki üretime kadar kalır
        assert any("1/3 varyant üretildi" in e.value for e in app.error)


def test_daily_limit_applies_to_owner_key(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    from ui import state, workspace

    state._quota.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "sahibin-anahtari")
    monkeypatch.setenv("DAILY_GENERATION_LIMIT", "1")

    with patch("sketch2code.llm.make_client", return_value=FakeClient()):
        app.run()
        _button(app, "Örnekle dene").click().run()
        assert any("kalan üretim hakkı: 1" in c.value for c in app.caption)
        _button(app, "Kodu üret").click().run()

    assert not app.exception
    assert len(app.session_state["pages"]["ornek"]["versions"]) == 1
    assert _button(app, "Yeniden üret").disabled
    assert any(workspace.QUOTA_MESSAGE in c.value for c in app.caption)
    state._quota.clear()
