"""
spine.stages.verifier
=========================
Etap 1: Weryfikacja obrazu wejściowego.

Cel: Odróżnić RTG kręgosłupa od obrazów niezwiązanych.
Model: Fine-tuned ResNet-18 (torchvision) — lekki, skuteczny klasyfikator binarny.

Uzasadnienie wyboru architektury:
    ResNet-18 to kompromis między wydajnością a rozmiarem modelu.
    Przy transferze wiedzy z ImageNet wystarczy kilka tysięcy przykładów
    RTG do osiągnięcia dobrej dokładności. Alternatywy:
    - EfficientNet-B0: lepszy stosunek accuracy/FLOPs, ale trudniejszy deploy.
    - MobileNetV3: optymalny dla CPU/edge, akceptowalny accuracy.
    Dla środowisk z GPU warto rozważyć upgrade do ResNet-50.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from spine.config import SpineConfig

logger = logging.getLogger(__name__)

# Normalizacja zgodna z pre-treningiem ImageNet
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier(num_classes: int = 2) -> nn.Module:
    """
    Buduje model klasyfikatora oparty na ResNet-18 z wymienioną głowicą.

    Args:
        num_classes: Liczba klas wyjściowych (domyślnie 2: spine_xray / other).

    Returns:
        nn.Module gotowy do treningu lub inferencji.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features: int = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


class ImageVerifier:
    """
    Weryfikator obrazu RTG kręgosłupa (Etap 1 pipeline'u spine).

    Używa fine-tunowanego ResNet-18 do binarnej klasyfikacji:
    - klasa 0: `other`      — obraz niezwiązany z RTG kręgosłupa
    - klasa 1: `spine_xray` — poprawne RTG kręgosłupa

    Args:
        config: Konfiguracja systemu spine.
    """

    SPINE_CLASS_IDX: int = 1

    def __init__(self, config: SpineConfig) -> None:
        self._cfg = config
        self._device = torch.device(config.device)
        self._model: Optional[nn.Module] = None
        self._transform = T.Compose([
            T.Resize(config.image_size),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        self._load_model()

    # ------------------------------------------------------------------
    # Inicjalizacja modelu
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Ładuje wagi modelu z dysku lub inicjalizuje architekturę (bez wag)."""
        self._model = build_classifier(num_classes=2)
        weights_path: Path = self._cfg.classifier_weights

        if weights_path.exists():
            logger.info("Ładowanie wag klasyfikatora: %s", weights_path)
            state = torch.load(weights_path, map_location=self._device)
            self._model.load_state_dict(state)
        else:
            logger.warning(
                "Brak pliku wag '%s' — model działa bez pretrenowanych wag. "
                "Wyniki będą losowe do momentu treningu.",
                weights_path,
            )

        self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def verify(self, image: Image.Image) -> tuple[bool, float]:
        """
        Ocenia, czy wejściowy obraz to RTG kręgosłupa.

        Args:
            image: Obraz PIL w trybie RGB.

        Returns:
            Krotka (is_spine_xray, confidence):
                - is_spine_xray: True jeżeli pewność >= próg konfiguracyjny.
                - confidence: Prawdopodobieństwo klasy spine_xray (0–1).
        """
        assert self._model is not None, "Model niezainicjalizowany."

        tensor: torch.Tensor = self._transform(image).unsqueeze(0).to(self._device)
        logits: torch.Tensor = self._model(tensor)                   # (1, 2)
        probs: torch.Tensor = torch.softmax(logits, dim=1)           # (1, 2)
        confidence: float = probs[0, self.SPINE_CLASS_IDX].item()
        is_spine = confidence >= self._cfg.classifier_threshold

        return is_spine, confidence

    # ------------------------------------------------------------------
    # Helpers dla treningu (używane w training scripts)
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        """Zwraca model do użycia w skryptach treningowych."""
        assert self._model is not None
        return self._model

    def get_transform_train(self) -> T.Compose:
        """
        Transformacje z augmentacją dla zbioru treningowego.

        Returns:
            Kompozycja transformacji z losowymi augmentacjami typowymi dla RTG.
        """
        return T.Compose([
            T.Resize((int(self._cfg.image_size[0] * 1.1), int(self._cfg.image_size[1] * 1.1))),
            T.RandomCrop(self._cfg.image_size),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
