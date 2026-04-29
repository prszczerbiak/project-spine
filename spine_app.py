"""
PROJECT: SPINE — spine System
==================================
Główny punkt wejścia aplikacji webowej opartej na Streamlit.
Odpowiada wyłącznie za warstwę prezentacji i orkiestrację wywołań pipeline'u.
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from spine.pipeline import SpineAnalyzer, AnalysisResult
from spine.config import SpineConfig

# ---------------------------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PROJECT: SPINE — spine",
    page_icon="🦴",
    layout="centered",
)

st.title("PROJECT: SPINE")
st.caption("System spine — Analiza RTG kręgosłupa")
st.divider()


# ---------------------------------------------------------------------------
# Singleton pipeline (cache Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Ładowanie modeli spine…")
def load_pipeline() -> SpineAnalyzer:
    """Tworzy i cachuje instancję SpineAnalyzer na czas sesji."""
    cfg = SpineConfig.from_env()
    return SpineAnalyzer(cfg)


# ---------------------------------------------------------------------------
# Główny UI
# ---------------------------------------------------------------------------
def main() -> None:
    analyzer: SpineAnalyzer = load_pipeline()

    uploaded = st.file_uploader(
        label="Wgraj zdjęcie RTG kręgosłupa (JPG / PNG / DICOM .dcm)",
        type=["jpg", "jpeg", "png", "dcm"],
        help="Obsługiwane formaty: JPEG, PNG oraz DICOM.",
    )

    if uploaded is None:
        st.info("Oczekiwanie na plik…")
        return

    # Konwersja wgranego pliku na PIL.Image
    try:
        image: Image.Image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as exc:
        st.error(f"Nie można odczytać pliku: {exc}")
        return

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Wgrane zdjęcie", use_container_width=True)

    with col_result:
        with st.spinner("Analiza w toku…"):
            result: AnalysisResult = analyzer.analyze(image)

        _render_result(result)


def _render_result(result: AnalysisResult) -> None:
    """Renderuje wyniki analizy w panelu bocznym."""

    # --- Krok 1: Weryfikacja obrazu ---
    st.subheader("1 · Weryfikacja obrazu")
    if not result.is_spine_xray:
        st.error(
            f"❌ Obraz nie został rozpoznany jako RTG kręgosłupa "
            f"(pewność: {result.classification_confidence:.1%})"
        )
        return  # Dalsze kroki bezsensowne

    st.success(
        f"✅ RTG kręgosłupa rozpoznane "
        f"(pewność: {result.classification_confidence:.1%})"
    )

    # --- Krok 2: Punkty kluczowe ---
    st.subheader("2 · Detekcja punktów kluczowych")
    if result.landmarks is None or len(result.landmarks) == 0:
        st.warning("⚠️ Nie wykryto punktów kluczowych.")
    else:
        st.write(f"Wykryto **{len(result.landmarks)}** punktów.")

    # --- Krok 3: Kąt Cobba ---
    st.subheader("3 · Kąt Cobba")
    if result.cobb_angle_deg is None:
        st.warning("Brak danych do obliczenia kąta.")
    else:
        st.metric(label="Kąt Cobba", value=f"{result.cobb_angle_deg:.1f}°")

    # --- Krok 4: Klasyfikacja medyczna ---
    st.subheader("4 · Klasyfikacja medyczna")
    if result.scoliosis_grade is not None:
        severity_color = {
            "Brak": "green",
            "Łagodna": "blue",
            "Umiarkowana": "orange",
            "Ciężka": "red",
        }.get(result.scoliosis_grade, "gray")

        st.markdown(
            f"**Ocena:** :{severity_color}[{result.scoliosis_grade}]"
        )

    if result.notes:
        st.caption(result.notes)

    st.divider()
    st.caption(
        "⚕️ Wyniki mają charakter pomocniczy. "
        "Diagnoza musi być zatwierdzona przez lekarza specjalistę."
    )


if __name__ == "__main__":
    main()
