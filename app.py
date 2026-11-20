"""Sketch-to-Code: kâğıt çizimi fotoğrafından çalışan HTML sayfası.

Çalıştırma: streamlit run app.py --server.port 8101
Bu dosya yalnız sayfa akışını kurar; mantık `sketch2code/`, arayüz parçaları `ui/` içindedir.
"""

from __future__ import annotations

import logging

import streamlit as st

from ui import inputs, sidebar, state, theme, workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    st.set_page_config(page_title="Sketch-to-Code", page_icon=":material/draw:", layout="wide")
    theme.inject()
    state.init_state()
    settings = sidebar.render()

    st.title("Sketch-to-Code")
    st.write(
        "Kâğıda çizdiğiniz sayfa taslağının fotoğrafını düzeltir, kutuları algılar ve "
        "Gemini ile çalışan bir HTML sayfasına çevirir. Önizleme yalıtılmış bir çerçevede açılır."
    )

    sources = inputs.render_upload()
    page_id = inputs.choose_page(sources)
    if page_id is None:
        inputs.render_empty()
        return

    analysis = inputs.render_review(page_id, sources[page_id], settings) if page_id in sources else None
    workspace.render(page_id, analysis, settings)


main()
