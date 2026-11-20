"""Adım 3: üret, sohbetle düzelt, sürümler, önizleme, kod ve indirme."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from sketch2code import export, history, llm, preview, prompts, vision
from ui import state

DEVICE_LABELS = {"desktop": "Masaüstü", "tablet": "Tablet", "mobile": "Mobil", "grid": "Yan yana"}
STREAM_TAIL = 8000  # akış sırasında gösterilen son karakter sayısı


QUOTA_MESSAGE = (
    "Bu demonun bugünkü üretim sınırı doldu. Yarın yeniden deneyin ya da projeyi kendi anahtarınızla "
    "yerelde çalıştırın."
)


def _generate(settings: state.Settings, image: Image.Image, prompt: str, pid: str, label: str) -> str | None:
    """Kodu akıtarak üretir; başarıda yeni sürüm ekler ve None döner. Hatada Türkçe mesajı döner."""
    if not state.take_generation(settings):
        return QUOTA_MESSAGE
    client = state.get_client(settings.api_key)
    with st.status(f"Kod yazılıyor: {label}", expanded=True) as status:
        code_slot = st.empty()
        text = ""
        try:
            for chunk in llm.stream_code(client, image, prompt, model=settings.model_code):
                text += chunk
                code_slot.code(text[-STREAM_TAIL:], language="html")
            html_code = llm.extract_html(text)
        except llm.GenerationError as exc:
            status.update(label=f"Üretim durdu: {label}", state="error", expanded=False)
            code_slot.empty()
            return exc.user_message
        status.update(label=f"Kod hazır: {label}", state="complete", expanded=False)
    pages = st.session_state.pages
    pages[pid] = history.add_version(pages[pid], html_code, label)
    return None


def _report_failure(pid: str, message: str, done: int, total: int) -> None:
    """Hatayı kalıcı bildirime yazar. Başarılı sürüm varsa sayfa yenilenir; bildirim yenilemeden sağ çıkar."""
    if total > 1:
        message = f"{done}/{total} varyant üretildi. Sonraki varyant durdu: {message}"
    st.session_state.notices[pid] = message


def _show_notice(pid: str, slot: DeltaGenerator) -> None:
    message = st.session_state.notices.get(pid)
    if message:
        slot.error(message, icon=":material/error:")
    else:
        slot.empty()


def _run_initial(
    pid: str, image: Image.Image, colors: list[str], boxes: list[vision.Box], settings: state.Settings, variants: bool
) -> None:
    styles = prompts.VARIANT_STYLES if variants else (settings.options.design_style,)
    done = 0
    for style in styles:
        options = replace(settings.options, design_style=style)
        prompt = prompts.build_generation_prompt(options, colors, boxes)
        label = f"Varyant {style.lower()}" if variants else f"Üretim, {style.lower()}"
        error = _generate(settings, image, prompt, pid, label)
        if error is not None:
            _report_failure(pid, error, done, len(styles))
            break
        done += 1
    if done:
        st.rerun()


def _versions_panel(pid: str, blocked: str | None) -> str | None:
    page = st.session_state.pages[pid]
    st.subheader("Sürümler")
    labels = {i: f"v{i + 1}  {v['label']}  ({v['created']})" for i, v in enumerate(page["versions"])}
    chosen = st.radio(
        "Gösterilen sürüm",
        list(labels),
        index=page["current"],
        format_func=labels.get,
        key=f"version_{pid}_{len(page['versions'])}_{page['current']}",
        label_visibility="collapsed",
    )
    if chosen is not None and chosen != page["current"]:
        st.session_state.pages[pid] = history.select(page, chosen)
        st.rerun()
    if st.button("Geri al", icon=":material/undo:", disabled=page["current"] <= 0, help="Bir önceki sürümü göster."):
        st.session_state.pages[pid] = history.undo(page)
        st.rerun()
    st.subheader("Düzelt")
    if blocked:
        st.caption(blocked)
    return st.chat_input("Ör. kartları üç sütun yap", key=f"refine_{pid}", disabled=blocked is not None)


def _preview_panel(html_code: str) -> None:
    device = st.segmented_control(
        "Önizleme genişliği",
        list(DEVICE_LABELS),
        format_func=DEVICE_LABELS.get,
        default="desktop",
        required=True,
        key="device",
    )
    if device == "grid":
        with st.container(key="device_grid"):
            for column, key in zip(st.columns([5, 3, 2]), ("desktop", "tablet", "mobile"), strict=True):
                with column:
                    _device_frame(html_code, preview.DEVICES[key], show_label=True)
        with st.container(key="device_grid_narrow"):
            st.caption("Yan yana görünüm geniş ekranda açılır. Dar ekranda mobil görünüm gösterilir.")
            _device_frame(html_code, preview.DEVICES["mobile"], show_label=False)
        return
    _device_frame(html_code, preview.DEVICES[device or "desktop"], show_label=False)


def _device_frame(html_code: str, device: preview.Device, show_label: bool) -> None:
    if show_label:
        st.caption(f"{device.label}, {device.width} × {device.height}")
    st.iframe(preview.sandboxed(html_code, device.width, device.height), height="content")


def _code_panel(pid: str, html_code: str) -> None:
    page = st.session_state.pages[pid]
    name = "ornek" if pid == state.SAMPLE_ID else "sayfa"
    with st.container(horizontal=True):
        st.download_button(
            "HTML indir", html_code, file_name="index.html", mime="text/html", icon=":material/download:"
        )
        st.download_button(
            "ZIP indir",
            export.build_zip(html_code, name),
            file_name=f"{name}.zip",
            mime="application/zip",
            icon=":material/folder_zip:",
        )
    code_tab, diff_tab = st.tabs(["Kod", "Sürüm farkı"])
    with code_tab:
        st.code(html_code, language="html", line_numbers=True, height=480)
    with diff_tab:
        count = len(page["versions"])
        if count < 2:
            st.caption("Fark görmek için en az iki sürüm gerekir.")
            return
        left, right = st.columns(2)
        fmt = lambda i: f"v{i + 1}  {page['versions'][i]['label']}"  # noqa: E731
        old = left.selectbox("Eski", range(count), index=max(0, page["current"] - 1), format_func=fmt)
        new = right.selectbox("Yeni", range(count), index=page["current"], format_func=fmt)
        text = history.diff(page, old, new)
        st.code(text or "İki sürüm aynı.", language="diff", height=420)


def render(pid: str, analysis: state.Analysis | None, settings: state.Settings) -> None:
    st.header("3. Üret ve düzenle")
    page = st.session_state.pages[pid]
    has_versions = bool(page["versions"])
    left = state.generations_left(settings)
    if not settings.api_key:
        blocked = "Sohbetle düzeltme için API anahtarı girin."
    elif analysis is None:
        blocked = "Sohbetle düzeltme için bu sayfanın çizimini yeniden yükleyin."
    elif left == 0:
        blocked = QUOTA_MESSAGE
    else:
        blocked = None
    can_generate = blocked is None

    button_col, toggle_col = st.columns([1, 3], vertical_alignment="center")
    clicked = button_col.button(
        "Yeniden üret" if has_versions else "Kodu üret",
        type="primary",
        icon=":material/code:",
        disabled=not can_generate,
        width="stretch",
    )
    variants = toggle_col.toggle("3 varyant üret (sade, klasik, cesur)", disabled=not can_generate)
    if not settings.api_key:
        st.caption(
            "Üretmek için kenar çubuğuna Gemini API anahtarı girin. Telefonda kenar çubuğu sol üstteki » "
            "düğmesiyle açılır. Anahtarsız da çizimi düzeltip inceleyebilirsiniz."
        )
    elif analysis is None:
        st.caption("Bu sayfanın çizimi yok. Sürümler görüntülenir; yeniden üretmek için çizimi yükleyin.")
    elif left == 0:
        st.caption(QUOTA_MESSAGE)
    elif left is not None:
        st.caption(f"Bu demoda bugün kalan üretim hakkı: {left}. Her varyant ve düzeltme bir hak kullanır.")

    notice_slot = st.empty()
    status_slot = st.container()
    if clicked and analysis is not None:
        st.session_state.notices.pop(pid, None)
        with status_slot:
            _run_initial(pid, analysis.image(), analysis.colors, analysis.result.boxes, settings, variants)
    _show_notice(pid, notice_slot)

    page = st.session_state.pages[pid]
    html_code = history.current_html(page)
    if html_code is None:
        if not clicked:
            st.caption(
                "Henüz sürüm yok. Kodu üret düğmesi düzeltilmiş çizimi modele gönderir; kod burada akarak görünür."
            )
        return

    side, main = st.columns([1, 2.4], gap="large")
    with side:
        request = _versions_panel(pid, blocked)
    with main:
        _preview_panel(html_code)
    if request and analysis is not None:
        st.session_state.notices.pop(pid, None)
        with status_slot:
            prompt = prompts.build_refine_prompt(html_code, request)
            label = request.strip()[:40]
            error = _generate(settings, analysis.image(), prompt, pid, label)
            if error is None:
                st.rerun()
            _report_failure(pid, error, 0, 1)
            _show_notice(pid, notice_slot)
    _code_panel(pid, html_code)
