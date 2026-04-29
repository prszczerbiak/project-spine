"""
spine.pipeline
==================
Główny orkiestrator pipeline'u ML systemu spine.

Łączy trzy niezależne etapy w jeden spójny przepływ:
    Weryfikator → Detektor → Analizator

Każdy etap można testować i podmieniać niezależnie.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from spine.config import SpineConfig
from spine.stages.verifier import ImageVerifier
from spine.stages.detector import LandmarkDetector
from spine.stages.analyzer import SpinalAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typ wynikowy
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    Kompletny wynik analizy jednego zdjęcia RTG.

    Attributes:
        is_spine_xray: True jeżeli obraz przeszedł weryfikację.
        classification_confidence: Pewność klasyfikatora (0–1).
        landmarks: Tablica kształtu (N, 2) z wykrytymi punktami [x, y].
        landmark_scores: Pewności poszczególnych punktów kluczowych.
        cobb_angle_deg: Obliczony kąt Cobba w stopniach lub None.
        scoliosis_grade: Tekstowa ocena stopnia skoliozy lub None.
        notes: Dodatkowe uwagi / ostrzeżenia.
    """

    is_spine_xray: bool = False
    classification_confidence: float = 0.0
    landmarks: Optional[np.ndarray] = None          # shape (N, 2)
    landmark_scores: Optional[np.ndarray] = None    # shape (N,)
    cobb_angle_deg: Optional[float] = None
    scoliosis_grade: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SpineAnalyzer:
    """
    Główna klasa pipeline'u spine.

    Opakowuje trzy etapy przetwarzania i zarządza ich cyklem życia.
    Etapy są inicjalizowane leniwie (lazy) przy pierwszym wywołaniu.

    Args:
        config: Konfiguracja systemu spine.

    Example::

        cfg = SpineConfig.from_env()
        analyzer = SpineAnalyzer(cfg)
        result = analyzer.analyze(pil_image)
    """

    def __init__(self, config: SpineConfig) -> None:
        self._cfg = config
        self._verifier: Optional[ImageVerifier] = None
        self._detector: Optional[LandmarkDetector] = None
        self._analyzer: Optional[SpinalAnalyzer] = None

    # ------------------------------------------------------------------
    # Leniwa inicjalizacja etapów
    # ------------------------------------------------------------------

    @property
    def verifier(self) -> ImageVerifier:
        if self._verifier is None:
            logger.info("Inicjalizacja ImageVerifier…")
            self._verifier = ImageVerifier(self._cfg)
        return self._verifier

    @property
    def detector(self) -> LandmarkDetector:
        if self._detector is None:
            logger.info("Inicjalizacja LandmarkDetector…")
            self._detector = LandmarkDetector(self._cfg)
        return self._detector

    @property
    def analyzer(self) -> SpinalAnalyzer:
        if self._analyzer is None:
            logger.info("Inicjalizacja SpinalAnalyzer…")
            self._analyzer = SpinalAnalyzer(self._cfg)
        return self._analyzer

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def analyze(self, image: Image.Image) -> AnalysisResult:
        """
        Przeprowadza pełną analizę wgranego zdjęcia RTG.

        Args:
            image: Wejściowy obraz PIL (RGB).

        Returns:
            AnalysisResult z wynikami wszystkich etapów.
        """
        result = AnalysisResult()

        # --- Etap 1: Weryfikacja ---
        logger.info("Etap 1: weryfikacja obrazu")
        is_spine, confidence = self.verifier.verify(image)
        result.is_spine_xray = is_spine
        result.classification_confidence = confidence

        if not is_spine:
            result.notes = (
                f"Obraz odrzucony przez weryfikator "
                f"(pewność: {confidence:.1%} < próg: "
                f"{self._cfg.classifier_threshold:.1%})."
            )
            logger.warning(result.notes)
            return result

        # --- Etap 2: Detekcja punktów ---
        logger.info("Etap 2: detekcja punktów kluczowych")
        landmarks, scores = self.detector.detect(image)
        result.landmarks = landmarks
        result.landmark_scores = scores

        if landmarks is None or len(landmarks) < 4:
            result.notes = "Zbyt mało punktów kluczowych do analizy geometrycznej."
            logger.warning(result.notes)
            return result

        # --- Etap 3: Analiza geometryczna i klasyfikacja ---
        logger.info("Etap 3: analiza Cobba i klasyfikacja")
        cobb, grade, notes = self.analyzer.analyze(landmarks, scores)
        result.cobb_angle_deg = cobb
        result.scoliosis_grade = grade
        result.notes = notes

        logger.info(
            "Analiza zakończona: kąt Cobba=%.1f°, ocena=%s", cobb or -1, grade
        )
        return result
