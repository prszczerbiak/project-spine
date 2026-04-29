"""
spine.stages.detector
=========================
Etap 2: Detekcja punktów kluczowych na kręgosłupie.

Cel: Wylokalizować specyficzne punkty anatomiczne na kręgach
     (4 narożniki każdego kręgu + punkt centralny = 5 × 17 = 85 pkt dla
     pełnego kręgosłupa piersiowo-lędźwiowego; AASCE dataset używa 68 pkt).

Uzasadnienie wyboru architektury:
    HRNet-W32 (High-Resolution Network) — state-of-art w pose estimation
    medycznym. W odróżnieniu od sieci hourglass, HRNet utrzymuje wysoką
    rozdzielczość przez cały forward pass, co jest kluczowe dla precyzji
    punktów (błąd < 2mm ma znaczenie kliniczne).

    Alternatywy:
    - Keypoint R-CNN (torchvision): gotowy out-of-box, łatwy start,
      ale mniejsza precyzja przy małych, nakładających się strukturach.
    - StackedHourglass: solidny, ale gorsza skalowanie do multi-scale.
    - UNet z heatmap regression: dobra alternatywa dla custom datasets.

    Dla produkcji: fine-tune HRNet-W32 na AASCE Challenge dataset.
    Wagi startowe: pretrenowane na COCO Keypoints.
"""

from __future__ import annotations

import logging 
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from spine.config import SpineConfig

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Minimalna implementacja heatmap-based keypoint head
# (zastąpić pełnym HRNet po zaimportowaniu biblioteki hrnet lub mmpose)
# ---------------------------------------------------------------------------

class _HeatmapKeypointHead(nn.Module):
    """
    Uproszczona głowica regresji heatmap.

    W środowisku produkcyjnym należy zastąpić tę klasę modelem HRNet-W32
    z biblioteki `mmpose` lub oficjalnego repozytorium HRNet:
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

    Architektura zastępcza (do testów bez wag):
        ResNet-50 backbone → 1×1 conv → upsample → sigmoid heatmaps
    """

    def __init__(self, num_keypoints: int = 68, pretrained_backbone: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        backbone = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        # Usuwamy avgpool i fc — zachowujemy feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # (B, 2048, H/32, W/32)

        self.keypoint_head = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),  # ×2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # ×4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, num_keypoints, kernel_size=4, stride=2, padding=1),  # ×8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, 3, H, W)
        Returns:
            Heatmapy (B, num_keypoints, H/4, W/4)
        """
        features = self.backbone(x)
        heatmaps = self.keypoint_head(features)
        return heatmaps


def _heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    original_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Konwertuje tensory heatmap na współrzędne 2D i wartości pewności.

    Args:
        heatmaps: Tensor (1, K, H', W') z modelu.
        original_size: Rozmiar oryginalnego obrazu (W, H) dla skalowania.

    Returns:
        Krotka (keypoints, scores):
            - keypoints: ndarray (K, 2) współrzędne [x, y] w px orig.
            - scores: ndarray (K,) maks. wartość heatmapy jako miara pewności.
    """
    heatmaps_np: np.ndarray = heatmaps.squeeze(0).cpu().numpy()  # (K, H', W')
    K, H, W = heatmaps_np.shape
    orig_w, orig_h = original_size

    keypoints = np.zeros((K, 2), dtype=np.float32)
    scores = np.zeros(K, dtype=np.float32)

    for k in range(K):
        hm = heatmaps_np[k]
        flat_idx = int(np.argmax(hm))
        y_hm, x_hm = divmod(flat_idx, W)

        # Skalowanie do przestrzeni oryginalnego obrazu
        keypoints[k, 0] = x_hm / W * orig_w
        keypoints[k, 1] = y_hm / H * orig_h
        scores[k] = float(hm[y_hm, x_hm])

    # Normalizacja scores do [0, 1] przez sigmoid
    scores = 1.0 / (1.0 + np.exp(-scores))
    return keypoints, scores


class LandmarkDetector:
    """
    Detektor punktów kluczowych kręgosłupa (Etap 2 pipeline'u spine).

    Używa sieci heatmapowej (docelowo HRNet-W32) do lokalizacji N punktów
    kluczowych na kręgach kręgosłupa.

    Args:
        config: Konfiguracja systemu spine.
    """

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

    def _load_model(self) -> None:
        """Ładuje model detektora lub inicjalizuje architekturę bez wag."""
        self._model = _HeatmapKeypointHead(num_keypoints=self._cfg.num_landmarks)
        weights_path: Path = self._cfg.landmark_weights

        if weights_path.exists():
            logger.info("Ładowanie wag detektora: %s", weights_path)
            state = torch.load(weights_path, map_location=self._device)
            self._model.load_state_dict(state)
        else:
            logger.warning(
                "Brak pliku wag '%s' — detektor bez wag (wyniki losowe).",
                weights_path,
            )

        self._model.to(self._device)
        self._model.eval()

    @torch.inference_mode()
    def detect(
        self, image: Image.Image
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Wykrywa punkty kluczowe kręgosłupa na obrazie.

        Args:
            image: Obraz PIL (RGB).

        Returns:
            Krotka (keypoints, scores):
                - keypoints: ndarray (K, 2) lub None przy błędzie.
                - scores: ndarray (K,) lub None przy błędzie.
        """
        assert self._model is not None

        orig_size: tuple[int, int] = image.size  # (W, H)
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        heatmaps: torch.Tensor = self._model(tensor)  # (1, K, H', W')
        keypoints, scores = _heatmaps_to_keypoints(heatmaps, orig_size)

        # Filtruj punkty poniżej progu pewności
        valid_mask = scores >= self._cfg.landmark_score_threshold
        logger.debug(
            "Wykryto %d/%d punktów powyżej progu %.2f",
            valid_mask.sum(), len(scores), self._cfg.landmark_score_threshold,
        )

        return keypoints, scores

    def visualize_landmarks(
        self, image: Image.Image, keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """
        Rysuje wykryte punkty na kopii obrazu (do debugowania / UI).

        Args:
            image: Oryginalny obraz PIL.
            keypoints: ndarray (K, 2).
            scores: ndarray (K,).

        Returns:
            ndarray BGR (OpenCV) z nałożonymi punktami.
        """
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        for (x, y), score in zip(keypoints, scores):
            color = (0, 255, 0) if score >= self._cfg.landmark_score_threshold else (0, 0, 255)
            cv2.circle(img_bgr, (int(x), int(y)), radius=4, color=color, thickness=-1)

        return img_bgr
