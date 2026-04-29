"""
spine.config
================
Centralna konfiguracja systemu spine.
Wszystkie ścieżki, progi i hiperparametry trafiają tutaj —
nigdy bezpośrednio do kodu logiki.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpineConfig:
    """
    Konfiguracja systemu spine.

    Attributes:
        model_dir: Katalog z plikami wag modeli (.pth).
        classifier_weights: Ścieżka do wag modelu weryfikującego (krok 1).
        landmark_weights: Ścieżka do wag detektora punktów (krok 2).
        classifier_threshold: Minimalny próg pewności dla klasyfikatora.
        landmark_score_threshold: Minimalny próg pewności dla keypointa.
        num_landmarks: Liczba oczekiwanych punktów kluczowych (4 per krąg × N kręgów).
        device: Urządzenie PyTorch ('cuda' | 'cpu' | 'mps').
        image_size: Rozmiar wejściowy modeli (szerokość, wysokość).
        enable_lateral_view: Flaga aktywująca (przyszłą) analizę widoku bocznego.
    """

    model_dir: Path = field(default_factory=lambda: Path("models"))
    classifier_weights: Path = field(
        default_factory=lambda: Path("models/classifier_resnet18.pth")
    )
    landmark_weights: Path = field(
        default_factory=lambda: Path("models/landmark_hrnet.pth")
    )
    classifier_threshold: float = 0.75
    landmark_score_threshold: float = 0.5
    num_landmarks: int = 68          # 17 kręgów × 4 punkty narożnikowe
    device: str = "cpu"
    image_size: tuple[int, int] = (512, 512)
    enable_lateral_view: bool = False   # future-proofing: lordoza / kifoza

    @classmethod
    def from_env(cls) -> "SpineConfig":
        """
        Tworzy konfigurację na podstawie zmiennych środowiskowych.
        Każda zmienna ma prefiks ``spine_``.

        Returns:
            SpineConfig z wartościami nadpisanymi przez ENV.
        """
        return cls(
            model_dir=Path(os.getenv("spine_MODEL_DIR", "models")),
            classifier_weights=Path(
                os.getenv(
                    "spine_CLASSIFIER_WEIGHTS",
                    "models/classifier_resnet18.pth",
                )
            ),
            landmark_weights=Path(
                os.getenv(
                    "spine_LANDMARK_WEIGHTS",
                    "models/landmark_hrnet.pth",
                )
            ),
            classifier_threshold=float(
                os.getenv("spine_CLASSIFIER_THRESHOLD", "0.75")
            ),
            landmark_score_threshold=float(
                os.getenv("spine_LANDMARK_SCORE_THRESHOLD", "0.5")
            ),
            num_landmarks=int(os.getenv("spine_NUM_LANDMARKS", "68")),
            device=os.getenv("spine_DEVICE", "cpu"),
            enable_lateral_view=os.getenv(
                "spine_LATERAL_VIEW", "false"
            ).lower() == "true",
        )
