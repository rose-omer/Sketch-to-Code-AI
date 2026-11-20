"""Gerçek tarayıcıda sandbox doğrulaması (Playwright).

Tarayıcı yoksa test atlanır. Yerelde Playwright'ın kendi Chromium'u yoksa kurulu
Chrome ya da Edge kanalı denenir.
"""

import pytest

from sketch2code import preview
from tests.fakes import PAGE

playwright_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser

ATTACK = """<!DOCTYPE html><html><head></head><body><script>
var r;
try { r = parent.document.title; r = "erisildi"; } catch (e) { r = e.name; }
document.body.dataset.result = r;
document.body.dataset.origin = String(window.origin);
try { top.location.href = "https://example.com"; document.body.dataset.nav = "gitti"; }
catch (e) { document.body.dataset.nav = e.name; }
</script></body></html>"""


@pytest.fixture(scope="module")
def page():
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
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        yield page
        browser.close()


def _inner_frame(page):
    page.wait_for_function("document.getElementById('frame') !== null")
    frame = page.frame_locator("#frame").owner.element_handle().content_frame()
    frame.wait_for_load_state()
    return frame


def test_generated_script_cannot_reach_parent(page) -> None:
    page.set_content(preview.sandboxed(ATTACK, 800, 600))
    frame = _inner_frame(page)
    frame.wait_for_function("document.body && document.body.dataset.result !== undefined")

    assert frame.evaluate("document.body.dataset.result") == "SecurityError"
    assert frame.evaluate("document.body.dataset.origin") == "null"  # opak köken
    assert page.url == "about:blank"  # üst pencere yönlenmedi


@pytest.mark.parametrize(("device", "expected"), [("mobile", "rgb(1, 2, 3)"), ("desktop", "rgba(0, 0, 0, 0)")])
def test_media_query_uses_real_device_width(page, device: str, expected: str) -> None:
    spec = preview.DEVICES[device]
    page.set_content(preview.sandboxed(PAGE, spec.width, spec.height))
    frame = _inner_frame(page)

    assert frame.evaluate("window.innerWidth") == spec.width
    assert frame.evaluate("getComputedStyle(document.body).backgroundColor") == expected


def test_wrapper_scales_down_to_fit(page) -> None:
    page.set_viewport_size({"width": 640, "height": 900})
    spec = preview.DEVICES["desktop"]
    page.set_content(preview.sandboxed(PAGE, spec.width, spec.height))

    box_width = page.evaluate("document.getElementById('box').getBoundingClientRect().width")

    assert box_width == pytest.approx(640, abs=2)
