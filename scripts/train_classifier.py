"""
scripts/train_classifier.py
==============================
Skrypt treningowy dla Etapu 1 — klasyfikatora RTG kręgosłupa.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from spine.config import SpineConfig
from spine.stages.verifier import build_classifier, ImageVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening klasyfikatora spine (Etap 1)")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Trenuje model przez jedną epokę, zwraca średnią stratę."""
    model.train()
    total_loss = 0.0
    total_batches = len(loader)
    log_every = max(1, total_batches // 5)  # loguj 5 razy per epoka

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
            logger.info(
                "  Epoka %d/%d | Batch %d/%d (%.0f%%) | Strata bieżąca: %.4f",
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
) -> float:
    """Ewaluuje dokładność (accuracy) na zbiorze walidacyjnym."""
    model.eval()
    correct = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()

    return correct / len(loader.dataset)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info("=== PROJECT: SPINE — Trening klasyfikatora ===")
    logger.info("Urządzenie: %s", args.device)
    logger.info("Epoki: %d | Batch size: %d | LR: %s", args.epochs, args.batch_size, args.lr)

    cfg = SpineConfig(device=args.device, model_dir=args.output_dir)
    verifier = ImageVerifier(cfg)
    model = verifier.get_model().to(device)

    logger.info("Wczytywanie datasetu z: %s", args.data_dir)
    train_ds = ImageFolder(args.data_dir / "train", transform=verifier.get_transform_train())
    val_ds = ImageFolder(args.data_dir / "val", transform=verifier._transform)
    logger.info("Dataset treningowy: %d zdjęć", len(train_ds))
    logger.info("Dataset walidacyjny: %d zdjęć", len(val_ds))
    logger.info("Klasy: %s", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    logger.info("Liczba batchy per epoka: %d", len(train_loader))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rozpoczynam trening...")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        logger.info("--- Epoka %d/%d START ---", epoch, args.epochs)

        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        logger.info("Ewaluacja na zbiorze walidacyjnym...")
        acc = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t_start
        logger.info(
            "Epoka %3d/%d | Strata: %.4f | Acc val: %.4f | Czas: %.0fs",
            epoch, args.epochs, loss, acc, elapsed,
        )

        if acc > best_acc:
            best_acc = acc
            out_path = args.output_dir / "classifier_resnet18.pth"
            torch.save(model.state_dict(), out_path)
            logger.info("Zapisano model → %s (acc=%.4f)", out_path, acc)

    logger.info("Trening zakończony. Najlepsza dokładność: %.4f", best_acc)


if __name__ == "__main__":
    main()