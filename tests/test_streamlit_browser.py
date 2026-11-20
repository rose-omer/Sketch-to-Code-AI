"""Uçtan uca: çalışan Streamlit sayfasında, st.iframe sarmalayıcısının içindeki önizleme yalıtılmış mı?

Sürüm, proje dosyası (.json) yüklenerek oluşturulur; Gemini çağrısı yok. Sunucu S2C_TEST_PORT
(varsayılan 8170) portunda açılır; port doluysa test atlanır.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from sketch2code import export, history, preview
from tests.fakes import PAGE

playwright_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("S2C_TEST_PORT", "8170"))
PROBE = "(() => { try { parent.document.title; return 'erisildi'; } catch (e) { return e.name; } })()"


def _port_busy(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def server_url():
    if _port_busy(PORT):
        pytest.skip(f"{PORT} portu dolu")
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    command = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT)]
    command += ["--server.address", "127.0.0.1", "--server.headless", "true"]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{PORT}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("Streamlit sunucusu açılmadı")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=20)


@pytest.fixture
def browser_page():
    with playwright_api.sync_playwright() as p:
        browser = None
        for kwargs in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
            try:
                browser = p.chromium.launch(**kwargs)
                break
            except playwright_api.Error:
                continue
        if browser is None:
            pytest.skip("Chromium bulunamadı")
        yield browser.new_page(viewport={"width": 1280, "height": 900})
        browser.close()


def _inner_preview(page, timeout_s: float = 30):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for frame in page.frames:
            if frame is not page.main_frame and frame.frame_element().get_attribute("id") == "frame":
                return frame
        page.wait_for_timeout(250)
    raise AssertionError("Önizleme iframe'i bulunamadı")


def test_preview_inside_streamlit_is_sandboxed(server_url: str, browser_page, tmp_path: Path) -> None:
    project = tmp_path / "proje.json"
    project.write_bytes(export.dump_project({"deneme": history.add_version(history.new_page("Deneme"), PAGE, "v")}))
    page = browser_page
    page.goto(server_url)
    page.locator(".st-key-project_file input[type=file]").set_input_files(project)

    inner = _inner_preview(page)
    element = inner.frame_element()
    wrapper = inner.parent_frame

    assert wrapper is not None and wrapper is not page.main_frame  # st.iframe sarmalayıcısı
    assert wrapper.parent_frame is page.main_frame
    assert element.get_attribute("sandbox") == preview.SANDBOX
    assert element.get_attribute("title") == preview.PREVIEW_TITLE
    inner.wait_for_function("document.body && document.body.textContent.includes('Merhaba')")
    assert inner.evaluate("String(window.origin)") == "null"  # opak köken
    assert inner.evaluate(PROBE) == "SecurityError"
