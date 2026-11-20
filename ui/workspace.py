"""Adım 3: üret, önce ve sonra karşılaştırması, sohbetle düzelt, sürümler, kod ve indirme.

Yerleşim (geniş ekran): solda kutu listesi, ortada önce/sonra sahnesi ve düzeltme isteği,
sağda sürümler, önizleme genişliği ve çıktı. Sütunlar DOM'da sahne, sağ sütun, kutu listesi
sırasıyla çizilir; dar ekranda bu sıra korunur (odak sırası görsel sırayla aynı). Geniş ekranda
kutu listesi CSS `order` ile sola alınır (theme.CSS); kutu listesinde odaklanan öğe yoktur.
"""

from __future__ import annotations

import html
import re
from dataclasses import replace

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from sketch2code import export, history, llm, preview, prompts, vision
from ui import state, theme

DEVICE_LABELS = {"desktop": "Masaüstü", "tablet": "Tablet", "mobile": "Mobil", "grid": "Yan yana"}
DEVICE_ICONS = {
    "desktop": ":material/desktop_windows:",
    "tablet": ":material/tablet:",
    "mobile": ":material/smartphone:",
    "grid": ":material/view_column:",
}
STREAM_TAIL = 8000  # akış sırasında gösterilen son karakter sayısı
CODE_MAX_HEIGHT = 480  # kod bloğu bu yükseklikten uzunsa kaydırılır
CODE_SHORT_LINES = 18  # bu satır sayısına kadar blok içeriği kadar uzar
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>:$])")


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


def _blocked_reason(settings: state.Settings, analysis: state.Analysis | None, left: int | None) -> str | None:
    """Üretim ve düzeltme neden kapalı? Açıksa None."""
    if not settings.api_key:
        return "Sohbetle düzeltme için API anahtarı girin."
    if analysis is None:
        return "Sohbetle düzeltme için bu sayfanın çizimini yeniden yükleyin."
    if left == 0:
        return QUOTA_MESSAGE
    return None


def _generate_controls(
    has_versions: bool, can_generate: bool, analysis: state.Analysis | None, left: int | None, settings: state.Settings
) -> tuple[bool, bool]:
    """Üret düğmesi, varyant anahtarı ve durum açıklaması. Dönen: (tıklandı mı, varyant mı)."""
    _output_options()
    with st.container(key="generate_controls"):
        clicked = st.button(
            "Yeniden üret" if has_versions else "Kodu üret",
            type="secondary" if has_versions else "primary",
            icon=":material/code:",
            disabled=not can_generate,
            width="stretch" if has_versions else "content",
        )
        variants = st.toggle("3 varyant üret (sade, klasik, cesur)", disabled=not can_generate)
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
    return clicked, variants


def _output_options() -> None:
    """Çıktı seçimleri. Değerler oturum durumunda kalır (state.OUTPUT_DEFAULTS); kenar çubuğu okur."""
    st.selectbox("CSS altyapısı", list(prompts.FRAMEWORKS), key="out_framework")
    st.selectbox("Renkler", list(prompts.COLOR_SCHEMES), key="out_colors")
    st.selectbox("Stil", prompts.DESIGN_STYLES, key="out_style")


# --- Sol: kutu ve karşılığı ----------------------------------------------------


def describe_box(box: vision.Box) -> tuple[str, str]:
    """Kutunun yerini ve boyutunu sözle anlatır: (başlık, ayrıntı). Tür tahmini yapmaz."""
    cy = box.y + box.h / 2
    cx = box.x + box.w / 2
    row = "Üst" if cy < 1 / 3 else "Orta" if cy < 2 / 3 else "Alt"
    if box.w >= 0.8:
        title = f"{row} bant"
    else:
        column = "sol" if cx < 1 / 3 else "orta" if cx < 2 / 3 else "sağ"
        title = f"{row} {column}" if (row, column) != ("Orta", "orta") else "Merkez"
    return title, f"Genişlik %{round(box.w * 100)}, yükseklik %{round(box.h * 100)}"


def _boxes_panel(analysis: state.Analysis | None) -> None:
    st.subheader("Kutu ve karşılığı", anchor=False)
    if analysis is None:
        st.caption("Bu sayfanın çizimi yüklü değil. Kutular çizimle birlikte gelir.")
        return
    boxes = analysis.result.boxes
    if not boxes:
        st.caption("Çizimde kapalı kutu bulunamadı. Model yalnız görüntüye bakarak üretti.")
        return
    st.caption(f"{len(boxes)} kutu algılandı. Modele düzen ipucu olarak gitti. Numaralar sahnedeki etiketlerdir.")
    items = []
    for number, box in enumerate(boxes, start=1):
        title, detail = describe_box(box)
        items.append(
            f'<li><span class="box-no">{number}</span>'
            f"<span><strong>{html.escape(title)}</strong><br><small>{html.escape(detail)}</small></span></li>"
        )
    st.html(f'<ol class="box-list" aria-label="Algılanan kutular">{"".join(items)}</ol>')


# --- Sağ: sürümler, genişlik --------------------------------------------------


def _md_escape(text: str) -> str:
    """Markdown işaretlerini kaçırır: sürüm etiketi kullanıcının düzeltme isteğinden gelir."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def version_label(index: int, version: dict[str, str]) -> str:
    """Sürüm satırı: numara, kalın etiket, eğik saat. CSS bunları üç sütunlu ızgaraya dizer (theme.CSS)."""
    return f"v{index + 1} **{_md_escape(version['label'])}** *{_md_escape(version['created'])}*"


def _versions_panel(pid: str) -> None:
    page = st.session_state.pages[pid]
    st.subheader("Sürüm", anchor=False)
    labels = {i: version_label(i, v) for i, v in enumerate(page["versions"])}
    with st.container(key="versions"):
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


def _device_picker() -> str:
    device = st.segmented_control(
        "Sonra tarafının genişliği",
        list(DEVICE_LABELS),
        format_func=lambda key: f"{DEVICE_ICONS[key]} {DEVICE_LABELS[key]}",
        default="mobile",  # dar sahnede okunur ölçek; tasarımda "Sonra" mobil genişlikte
        required=True,
        key="device",
    )
    return device or "mobile"


# --- Orta: sahne ve düzeltme ---------------------------------------------------


def _version_name(page: history.Page) -> str:
    return f"v{page['current'] + 1}"


def _stage(html_code: str, analysis: state.Analysis | None, device: str, page: history.Page) -> None:
    if device == "grid":
        with st.container(key="device_grid"):
            for column, key in zip(st.columns([5, 3, 2]), ("desktop", "tablet", "mobile"), strict=True):
                with column:
                    _device_frame(html_code, preview.DEVICES[key], show_label=True)
        with st.container(key="device_grid_narrow"):
            st.caption("Yan yana görünüm geniş ekranda açılır. Dar ekranda mobil görünüm gösterilir.")
            _device_frame(html_code, preview.DEVICES["mobile"], show_label=False)
        return
    spec = preview.DEVICES[device]
    if analysis is None:
        st.caption(f"Sonra: {_version_name(page)}, {spec.label.lower()} {spec.width} px. Çizim yüklü değil.")
        _device_frame(html_code, spec, show_label=False)
        return
    corrected = analysis.result.corrected
    height, width = corrected.shape[:2]
    wrapper = preview.compare(
        html_code,
        before_url=_before_url(corrected.tobytes(), corrected.shape),
        before_ratio=height / width,
        boxes=analysis.result.boxes,
        width=spec.width,
        after_label=f"{_version_name(page)}, {spec.width} px",
        dark=theme.is_dark(),
    )
    with st.container(key="stage"):
        st.iframe(wrapper, height="content", alt=preview.COMPARE_TITLE)


@st.cache_data(max_entries=8, show_spinner=False)
def _before_url(pixels: bytes, shape: tuple[int, ...]) -> str:
    """Çizimin data: URL'si; her yeniden çalıştırmada JPEG'e yeniden kodlanmaz."""
    return preview.image_data_url(np.frombuffer(pixels, dtype=np.uint8).reshape(shape))


def _device_frame(html_code: str, device: preview.Device, show_label: bool) -> None:
    if show_label:
        st.caption(f"{device.label}, {device.width} × {device.height}")
    st.iframe(
        preview.sandboxed(html_code, device.width, device.height),
        height="content",
        alt=f"{device.label} önizleme çerçevesi",
    )


def _refine_panel(pid: str, blocked: str | None, page: history.Page) -> str | None:
    """Görünür etiketli düzeltme formu ve role=status durum satırı. Gönderilen isteği döner."""
    with (
        st.container(key="refine"),
        st.form(f"refine_form_{pid}", clear_on_submit=True, border=False),
        st.container(horizontal=True, vertical_alignment="bottom", gap="small"),
    ):
        request = st.text_input(
            "Düzeltme isteği",
            placeholder="Ör. kayıt düğmesini çizimdeki maviye boya",
            key=f"refine_{pid}",
            disabled=blocked is not None,
        )
        sent = st.form_submit_button("Gönder", type="primary", icon=":material/send:", disabled=blocked is not None)
    if blocked:
        status = blocked
    else:
        current = page["versions"][page["current"]]
        status = (
            f"Gösterilen: {_version_name(page)}, {current['label']}. Sonraki istek v{len(page['versions']) + 1} olur."
        )
    st.html(f'<p role="status" class="refine-status">{html.escape(status)}</p>')
    return request.strip() if sent and request and request.strip() else None


# --- Üst çubuk ve kod ------------------------------------------------------------


def _toolbar(pid: str, page: history.Page, html_code: str) -> None:
    name = "ornek" if pid == state.SAMPLE_ID else "sayfa"
    with st.container(horizontal=True, vertical_alignment="center", key="toolbar"):
        st.markdown(f"**{page['name']}**, {_version_name(page)} gösteriliyor", width="stretch")
        st.download_button(
            "ZIP indir",
            export.build_zip(html_code, name),
            file_name=f"{name}.zip",
            mime="application/zip",
            icon=":material/folder_zip:",
        )
        st.download_button(
            "HTML indir",
            html_code,
            file_name="index.html",
            mime="text/html",
            icon=":material/download:",
            type="primary",
        )


def code_height(text: str) -> int | str:
    """Kısa kod içeriği kadar uzar; uzun kod sabit yükseklikte kaydırılır (altı boş kalmaz)."""
    return "content" if text.count("\n") < CODE_SHORT_LINES else CODE_MAX_HEIGHT


def _code_panel(page: history.Page, html_code: str) -> None:
    code_tab, diff_tab = st.tabs(["Kod", "Sürüm farkı"])
    with code_tab:
        st.code(html_code, language="html", line_numbers=True, height=code_height(html_code))
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
        text = text or "İki sürüm aynı."
        st.code(text, language="diff", height=code_height(text))


def render(pid: str, analysis: state.Analysis | None, settings: state.Settings) -> None:
    st.header("3. Üret ve düzenle")
    page = st.session_state.pages[pid]
    left = state.generations_left(settings)
    blocked = _blocked_reason(settings, analysis, left)
    can_generate = blocked is None

    notice_slot = st.empty()
    status_slot = st.container()
    html_code = history.current_html(page)

    if html_code is None:
        clicked, variants = _generate_controls(False, can_generate, analysis, left, settings)
        if clicked and analysis is not None:
            st.session_state.notices.pop(pid, None)
            with status_slot:
                _run_initial(pid, analysis.image(), analysis.colors, analysis.result.boxes, settings, variants)
        _show_notice(pid, notice_slot)
        if not clicked:
            st.caption(
                "Henüz sürüm yok. Kodu üret düğmesi düzeltilmiş çizimi modele gönderir; kod burada akarak görünür. "
                "Sonra çizim ile sayfayı kaydırıcıyla üst üste karşılaştırırsınız."
            )
        return

    _toolbar(pid, page, html_code)
    with st.container(key="studio"):
        # DOM sırası = dar ekrandaki görsel sıra = odak sırası. Geniş ekranda kutu listesi sola alınır.
        stage_col, side_col, boxes_col = st.columns([2.9, 1.35, 1.15], gap="large")
    with side_col:
        _versions_panel(pid)
        device = _device_picker()
        st.subheader("Çıktı", anchor=False)
        clicked, variants = _generate_controls(True, can_generate, analysis, left, settings)
    with stage_col:
        _stage(html_code, analysis, device, page)
        request = _refine_panel(pid, blocked, page)
    with boxes_col:
        _boxes_panel(analysis)

    if clicked and analysis is not None:
        st.session_state.notices.pop(pid, None)
        with status_slot:
            _run_initial(pid, analysis.image(), analysis.colors, analysis.result.boxes, settings, variants)
    if request and analysis is not None:
        st.session_state.notices.pop(pid, None)
        with status_slot:
            prompt = prompts.build_refine_prompt(html_code, request)
            error = _generate(settings, analysis.image(), prompt, pid, request.strip()[:40])
            if error is None:
                st.rerun()
            _report_failure(pid, error, 0, 1)
    _show_notice(pid, notice_slot)
    _code_panel(page, html_code)
