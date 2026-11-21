"""Adım 1 (yükle) ve adım 2 (düzelt ve incele)."""

from __future__ import annotations

import streamlit as st

from sketch2code import history, llm, vision
from ui import state, theme

Source = tuple[str, bytes]  # (ad, bayt)


def render_upload() -> dict[str, Source]:
    """Yükleme alanları. Dönen sözlük: sayfa kimliği → (ad, bayt)."""
    st.header("1. Çizimi yükle")
    upload_col, side_col = st.columns([3, 2], gap="large")
    with upload_col:
        files = st.file_uploader(
            "Kâğıt çiziminin fotoğrafı",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            max_upload_size=10,
            key="sketch_files",
            help="JPG ya da PNG, en çok 10 MB. Her dosya ayrı bir sayfa olur.",
        )
    with side_col:
        st.write("Çiziminiz yanınızda değil mi?")
        if st.button("Örnekle dene", icon=":material/draw:"):
            st.session_state.use_sample = True
        use_camera = st.toggle("Kamerayla çek", help="Telefonda çizimi doğrudan fotoğraflayın.")
    photo = st.camera_input("Çizimi çerçeveye sığdırın", key="camera") if use_camera else None

    sources: dict[str, Source] = {}
    for file in files or []:
        data = file.getvalue()
        sources[history.page_id(data)] = (file.name, data)
    if photo is not None:
        data = photo.getvalue()
        sources[history.page_id(data)] = ("Kamera fotoğrafı", data)
    if st.session_state.get("use_sample"):
        sources[state.SAMPLE_ID] = ("Örnek çizim", state.sample_bytes())
    for pid, (name, _) in sources.items():
        st.session_state.pages.setdefault(pid, history.new_page(name))
    return sources


def choose_page(sources: dict[str, Source]) -> str | None:
    """Birden çok sayfa varsa seçici gösterir; etkin sayfa kimliğini döner."""
    pages = st.session_state.pages
    options = list(sources) + [pid for pid, page in pages.items() if pid not in sources and page["versions"]]
    if not options:
        return None
    if len(options) == 1:
        return options[0]
    active = st.session_state.active_page if st.session_state.active_page in options else options[0]
    chooser = st.segmented_control if len(options) <= 4 else st.selectbox
    kwargs = (
        {"default": active, "required": True} if chooser is st.segmented_control else {"index": options.index(active)}
    )
    chosen = chooser("Üzerinde çalışılan sayfa", options, format_func=lambda pid: pages[pid]["name"], **kwargs)
    st.session_state.active_page = chosen or active
    return st.session_state.active_page


def render_empty() -> None:
    st.image(
        str(theme.ASSETS / "pipeline.jpg"),
        caption="Örnek: masadaki fotoğraf, düzeltilmiş çizim ve modele düzen ipucu olarak giden kutular.",
        width="stretch",
    )
    with st.container(border=True):
        st.markdown(
            "**İyi sonuç için fotoğraf**\n"
            "- Kâğıdın dört köşesi kadrajda olsun; arka plan kâğıttan koyu olsun.\n"
            "- Kutuları kapalı çizin: menü, görsel, kart ve düğme ayrı dikdörtgen olsun.\n"
            "- Gölge sorun değil; uygulama aydınlatmayı ve perspektifi düzeltir."
        )
        st.caption("Çiziminiz yoksa Örnekle dene düğmesi hazır bir fotoğrafı işler.")


def _crop_form(pid: str, paper_found: bool) -> None:
    crop = st.session_state.crops.get(pid)
    with st.expander("Elle kırp", expanded=not paper_found and crop is None), st.form(f"crop_{pid}", border=False):
        horizontal = st.slider("Yatay aralık (%)", 0, 100, (0, 100) if crop is None else _pct(crop[:2]))
        vertical = st.slider("Dikey aralık (%)", 0, 100, (0, 100) if crop is None else _pct(crop[2:]))
        apply_col, reset_col = st.columns(2)
        if apply_col.form_submit_button("Kırpmayı uygula", type="primary"):
            st.session_state.crops[pid] = (*_frac(horizontal), *_frac(vertical))
            st.rerun()
        if reset_col.form_submit_button("Otomatik algılamaya dön", disabled=crop is None):
            st.session_state.crops.pop(pid, None)
            st.rerun()


def _pct(pair: tuple[float, ...]) -> tuple[int, int]:
    return int(pair[0] * 100), int(pair[1] * 100)


def _frac(pair: tuple[int, int]) -> tuple[float, float]:
    return pair[0] / 100, pair[1] / 100


def render_review(pid: str, source: Source, settings: state.Settings) -> state.Analysis | None:
    st.header("2. Düzelt ve incele")
    crop = st.session_state.crops.get(pid)
    try:
        with st.spinner("Kâğıt bulunuyor, gölge ve perspektif düzeltiliyor…"):
            analysis = state.analyse(source[1], crop)
    except vision.ImageLoadError as exc:
        st.error(f"{exc} Dosyayı yeniden kaydedip yükleyin.")
        return None
    result = analysis.result
    if crop is None and not result.paper_found:
        st.warning("Kâğıdın kenarları bulunamadı. Görüntü olduğu gibi kullanıldı; gerekirse elle kırpın.")

    columns = st.columns(3)
    columns[0].image(result.original, caption="Orijinal fotoğraf", width="stretch")
    columns[1].image(result.corrected, caption="Düzeltilmiş: perspektif, gölge, kontrast", width="stretch")
    columns[2].image(
        vision.draw_boxes(result.corrected, result.boxes, color=theme.ACCENT_RGB),
        caption=f"Algılanan kutular: {len(result.boxes)}. Modele düzen ipucu olarak gider.",
        width="stretch",
    )
    _crop_form(pid, result.paper_found)

    palette_col, tips_col = st.columns([3, 2], gap="large")
    with palette_col:
        st.subheader("Renk paleti")
        theme.palette_strip(analysis.colors)
        st.code(" ".join(analysis.colors), language=None)
    with tips_col:
        _suggestions(pid, analysis, settings)
    return analysis


def _suggestions(pid: str, analysis: state.Analysis, settings: state.Settings) -> None:
    st.subheader("Model önerileri")
    clicked = st.button(
        "Öneri al", disabled=not settings.api_key, help="Hızlı model çizime bakıp en çok 4 öneri verir."
    )
    if clicked:
        try:
            with st.spinner("Öneriler hazırlanıyor…"):
                st.session_state.suggestions[pid] = llm.suggest(
                    state.get_client(settings.api_key), analysis.image(), model=settings.model_fast
                )
        except llm.GenerationError as exc:
            st.error(exc.user_message)
    found = st.session_state.suggestions.get(pid)
    if found:
        st.markdown("\n".join(f"- **{item.title}**: {item.detail}" for item in found.items))
    elif not settings.api_key:
        st.caption("Öneri için kenar çubuğuna API anahtarı girin.")
    else:
        st.caption("Eksik bölüm ve düzen önerileri burada listelenir.")
