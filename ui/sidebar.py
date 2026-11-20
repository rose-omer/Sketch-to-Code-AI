"""Kenar çubuğu: anahtar, kod seçenekleri, gelişmiş ayarlar ve proje dosyası.

CSS altyapısı, renkler ve stil adım 3'ün sağ sütunundadır (workspace); değerleri oturum durumundan okunur.
"""

from __future__ import annotations

import streamlit as st

from sketch2code import export, llm, prompts
from ui.state import Settings, env, secrets_mapping


def _key_section() -> tuple[str, str]:
    st.subheader("Gemini anahtarı")
    secret_key, source = llm.resolve_api_key(secrets_mapping(), env(), "")
    if secret_key:
        label = "sunucu gizli ayarı" if source == "secrets" else "ortam değişkeni"
        st.caption(f"Anahtar {label} üzerinden tanımlı.")
        return secret_key, source
    typed = st.text_input(
        "API anahtarı",
        type="password",
        placeholder="AIza…",
        help="Anahtar yalnız bu tarayıcı oturumunda kalır; diske yazılmaz.",
    )
    st.caption(
        f"[Anahtarım nerede?]({llm.API_KEY_URL}) Ücretsiz katmanda Google, gönderilen içeriği "
        "ürün geliştirmede kullanabilir. Gizli çizim yüklemeyin."
    )
    return llm.resolve_api_key(None, {}, typed)


def _options_section() -> prompts.GenerationOptions:
    st.subheader("Kod seçenekleri")
    responsive = st.toggle("Mobil uyumlu", value=True)
    add_accessibility = st.toggle("Erişilebilirlik kuralları", value=True)
    add_seo = st.toggle("SEO etiketleri", value=False)
    animations = st.toggle("Geçiş animasyonları", value=False)
    custom_request = st.text_area(
        "Ek istek",
        placeholder="Ör. iletişim formu ekle, menü sabit kalsın",
        max_chars=prompts.MAX_USER_TEXT,
        height=96,
    )
    return prompts.GenerationOptions(
        framework=st.session_state.out_framework,
        color_scheme=st.session_state.out_colors,
        design_style=st.session_state.out_style,
        responsive=responsive,
        animations=animations,
        add_seo=add_seo,
        add_accessibility=add_accessibility,
        custom_request=custom_request,
    )


def _project_section() -> None:
    st.subheader("Proje dosyası")
    st.caption("Sürümler sayfa yenilenince silinir. Saklamak için indirin.")
    pages = st.session_state.pages
    has_versions = any(page["versions"] for page in pages.values())
    st.download_button(
        "Projeyi indir (.json)",
        export.dump_project(pages) if has_versions else b"",
        file_name="sketch2code-proje.json",
        mime="application/json",
        disabled=not has_versions,
        width="stretch",
    )
    uploaded = st.file_uploader("Proje dosyası yükle", type=["json"], key="project_file")
    if uploaded is not None and st.session_state.get("loaded_project") != uploaded.file_id:
        try:
            loaded = export.load_project(uploaded.getvalue())
        except export.ProjectFileError as exc:
            st.error(str(exc))
        else:
            st.session_state.pages.update(loaded)
            st.session_state.loaded_project = uploaded.file_id
            st.success(f"{len(loaded)} sayfanın sürümleri yüklendi.")


def render() -> Settings:
    with st.sidebar:
        api_key, source = _key_section()
        st.divider()
        options = _options_section()
        with st.expander("Gelişmiş"):
            code_model = st.text_input("Kod modeli", value=llm.model_code(), help="Ortam değişkeni: MODEL_CODE")
            fast_model = st.text_input("Öneri modeli", value=llm.model_fast(), help="Ortam değişkeni: MODEL_FAST")
        st.divider()
        _project_section()
    return Settings(
        api_key=api_key,
        key_source=source,
        options=options,
        model_code=code_model.strip() or llm.model_code(),
        model_fast=fast_model.strip() or llm.model_fast(),
    )
