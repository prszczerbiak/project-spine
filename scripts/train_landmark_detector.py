"""
scripts/train_landmark_detector.py
=====================================
Skrypt treningowy dla Etapu 2 — detektora punktów kluczowych kręgosłupa.

Uruchomienie:
    python scripts/train_landmark_detector.py \
        --images_dir data/landmarks/images \
        --landmarks_csv data/landmarks/landmarks.csv \
        --filenames_csv data/landmarks/filenames.csv \
        --angles_csv data/landmarks/angles.csv \
        --output_dir models \
        --epochs 50 \
        --batch_size 8 \
        --device cpu

Format danych wejściowych (AASCE):
    - filenames.csv: jedna nazwa pliku per linia, bez nagłówka
    - landmarks.csv: 136 wartości per linia (68 punktów × x,y), znormalizowane 0-1
    - angles.csv: 3 kąty Cobba per linia (górny piersiowy, główny, lędźwiowy)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image

from spine.config import SpineConfig
from spine.stages.analyzer import compute_cobb_angle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Normalizacja ImageNet
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

NUM_LANDMARKS = 68
IMAGE_SIZE = (1024, 512)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpineLandmarkDataset(Dataset):
    """
    Dataset punktów kluczowych kręgosłupa w formacie AASCE.

    Args:
        images_dir: Katalog ze zdjęciami RTG.
        filenames_csv: Ścieżka do pliku z nazwami plików.
        landmarks_csv: Ścieżka do pliku z punktami kluczowymi (136 wartości per linia).
        image_size: Rozmiar obrazu wejściowego (szerokość, wysokość).
        augment: Czy stosować augmentację (tylko dla zbioru treningowego).
    """

    def __init__(
        self,
        images_dir: Path,
        filenames_csv: Path,
        landmarks_csv: Path,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.image_size = image_size

        # Wczytaj nazwy plików
        self.filenames = pd.read_csv(filenames_csv, header=None)[0].tolist()

        # Wczytaj punkty kluczowe (136 kolumn, znormalizowane 0-1)
        self.landmarks = pd.read_csv(landmarks_csv, header=None).values.astype(np.float32)

        assert len(self.filenames) == len(self.landmarks), (
            f"Niezgodna liczba plików ({len(self.filenames)}) "
            f"i punktów ({len(self.landmarks)})"
        )

        # Transformacje
        if augment:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ])

        logger.info("Dataset załadowany: %d próbek", len(self.filenames))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca krotkę (obraz, punkty_kluczowe).

        Returns:
            image: Tensor (3, H, W)
            landmarks: Tensor (136,) — współrzędne x,y znormalizowane 0-1
                        kolejność: x0,y0, x1,y1, ..., x67,y67
        """
        img_path = self.images_dir / self.filenames[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.warning("Brak pliku: %s — zastępuję czarnym obrazem", img_path)
            image = Image.new("RGB", self.image_size, color=0)

        image_tensor = self.transform(image)
        landmarks_tensor = torch.tensor(self.landmarks[idx], dtype=torch.float32)

        return image_tensor, landmarks_tensor


# ---------------------------------------------------------------------------
# Model — ResNet-50 z głowicą regresji punktów
# ---------------------------------------------------------------------------

def build_landmark_model(num_landmarks: int = NUM_LANDMARKS) -> nn.Module:
    """
    Buduje model regresji punktów kluczowych.

    Architektura: ResNet-50 backbone (pretrenowany ImageNet)
    + głowica regresji → 2 × num_landmarks wartości (x,y znormalizowane 0-1).

    Sigmoid na wyjściu gwarantuje wartości w zakresie [0, 1].

    Args:
        num_landmarks: Liczba punktów kluczowych (domyślnie 68).

    Returns:
        nn.Module gotowy do treningu.
    """
    import torchvision.models as tvm

    backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    in_features = backbone.fc.in_features

    backbone.fc = nn.Sequential(
        # Warstwa 1: 2048 -> 1024
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        
        # Warstwa 2: 1024 -> 512
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        
        # Warstwa 3: 512 -> 256
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        
        # Wyjście: 256 -> 136 (68 punktów x 2)
        nn.Linear(256, num_landmarks * 2),
        nn.Sigmoid()
    )

    return backbone


# ---------------------------------------------------------------------------
# Metryki
# ---------------------------------------------------------------------------

def mean_distance_error(
    preds: torch.Tensor,
    targets: torch.Tensor,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> float:
    """
    Oblicza średni błąd odległości euklidesowej w pikselach.

    Args:
        preds: Tensor (B, 136) — przewidywane współrzędne znormalizowane.
        targets: Tensor (B, 136) — prawdziwe współrzędne znormalizowane.
        image_size: Rozmiar obrazu do denormalizacji.

    Returns:
        Średni błąd w pikselach.
    """
    w, h = image_size
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Reshape do (B, 68, 2)
    preds_pts = preds_np.reshape(-1, NUM_LANDMARKS, 2)
    targets_pts = targets_np.reshape(-1, NUM_LANDMARKS, 2)

    # Denormalizacja
    preds_pts[:, :, 0] *= w
    preds_pts[:, :, 1] *= h
    targets_pts[:, :, 0] *= w
    targets_pts[:, :, 1] *= h

    # Odległość euklidesowa per punkt
    distances = np.sqrt(np.sum((preds_pts - targets_pts) ** 2, axis=2))
    return float(distances.mean())


def cobb_angle_error(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Oblicza średni błąd kąta Cobba między przewidywaniami a ground truth.

    Używa uproszczonej metody: kąt między pierwszym a ostatnim kręgiem.

    Args:
        preds: Tensor (B, 136)
        targets: Tensor (B, 136)

    Returns:
        Średni błąd kąta Cobba w stopniach.
    """
    errors = []
    preds_np = preds.detach().cpu().numpy().reshape(-1, NUM_LANDMARKS, 2)
    targets_np = targets.detach().cpu().numpy().reshape(-1, NUM_LANDMARKS, 2)

    for pred_pts, true_pts in zip(preds_np, targets_np):
        try:
            pred_angle = compute_cobb_angle(
                pred_pts[0], pred_pts[1],   # górna płytka pierwszego kręgu
                pred_pts[-4], pred_pts[-3], # dolna płytka ostatniego kręgu
            )
            true_angle = compute_cobb_angle(
                true_pts[0], true_pts[1],
                true_pts[-4], true_pts[-3],
            )
            errors.append(abs(pred_angle - true_angle))
        except ValueError:
            continue

    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Trening / ewaluacja
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Trenuje model przez jedną epokę, zwraca średnią stratę MSE."""
    model.train()
    total_loss = 0.0
    total_batches = len(loader)
    log_every = max(1, total_batches // 5)

    for batch_idx, (imgs, landmarks) in enumerate(loader):
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, landmarks)
        loss.backward()

        # Gradient clipping — stabilizuje trening regresji
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(imgs)

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
            logger.info(
                "  Epoka %d/%d | Batch %d/%d (%.0f%%) | MSE: %.6f",
                epoch, total_epochs,
                batch_idx + 1, total_batches,
                100.0 * (batch_idx + 1) / total_batches,
                loss.item(),
            )

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: tuple[int, int],
) -> tuple[float, float, float]:
    """
    Ewaluuje model na zbiorze walidacyjnym.

    Returns:
        Krotka (mse, mean_dist_px, cobb_error_deg).
    """
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    all_preds = []
    all_targets = []

    for imgs, landmarks in loader:
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)
        preds = model(imgs)
        loss = criterion(preds, landmarks)
        total_loss += loss.item() * len(imgs)
        all_preds.append(preds.cpu())
        all_targets.append(landmarks.cpu())

    all_preds_t = torch.cat(all_preds)
    all_targets_t = torch.cat(all_targets)

    mse = total_loss / len(loader.dataset)
    dist_px = mean_distance_error(all_preds_t, all_targets_t, image_size)
    cobb_err = cobb_angle_error(all_preds_t, all_targets_t)

    return mse, dist_px, cobb_err


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening detektora punktów kluczowych PROJECT: SPINE (Etap 2)"
    )
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--landmarks_csv", type=Path, required=True)
    parser.add_argument("--filenames_csv", type=Path, required=True)
    parser.add_argument("--angles_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info("=== PROJECT: SPINE — Trening detektora punktów (Etap 2) ===")
    logger.info("Urządzenie: %s", args.device)
    logger.info("Epoki: %d | Batch size: %d | LR: %.0e", args.epochs, args.batch_size, args.lr)

    # Dataset
    full_dataset = SpineLandmarkDataset(
        images_dir=args.images_dir,
        filenames_csv=args.filenames_csv,
        landmarks_csv=args.landmarks_csv,
        image_size=IMAGE_SIZE,
        augment=False,  # augmentacja tylko dla train split
    )

    # Podział train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Włącz augmentację dla zbioru treningowego
    train_ds.dataset.augment = True

    logger.info("Train: %d zdjęć | Val: %d zdjęć", train_size, val_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"), drop_last=True
    )

    # Model
    logger.info("Budowanie modelu ResNet-50 + głowica regresji...")
    model = build_landmark_model(num_landmarks=NUM_LANDMARKS).to(device)

    # Sprawdź czy istnieją wagi do wczytania
    weights_path = args.output_dir / "landmark_hrnet.pth"
    if weights_path.exists():
        logger.info("Wczytywanie istniejących wag: %s", weights_path)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dist = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rozpoczynam trening...")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        logger.info("--- Epoka %d/%d START ---", epoch, args.epochs)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        logger.info("Ewaluacja...")
        mse, dist_px, cobb_err = evaluate(model, val_loader, device, IMAGE_SIZE)
        scheduler.step()

        elapsed = time.time() - t_start
        logger.info(
            "Epoka %3d/%d | MSE train: %.6f | MSE val: %.6f | "
            "Błąd punktów: %.1f px | Błąd Cobba: %.2f° | Czas: %.0fs",
            epoch, args.epochs, train_loss, mse, dist_px, cobb_err, elapsed,
        )

        # Zapisz jeśli błąd odległości punktów się poprawił
        if dist_px < best_dist:
            best_dist = dist_px
            torch.save(model.state_dict(), weights_path)
            logger.info(
                "Zapisano model → %s (błąd punktów: %.1f px, Cobb: %.2f°)",
                weights_path, dist_px, cobb_err,
            )

    logger.info("Trening zakończony. Najlepszy błąd punktów: %.1f px", best_dist)


if __name__ == "__main__":
    main()