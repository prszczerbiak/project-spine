"""
scripts/sample_fracatlas.py
============================
Losuje 98 zdjęć z datasetu FracAtlas do folderu data/raw/other.

Uruchomienie:
    python scripts/sample_fracatlas.py \
        --fracatlas_dir C:/ścieżka/do/FracAtlas \
        --output_dir data/raw/other \
        --count 98

Następnie uruchom augmentację:
    python scripts/augment_dataset.py \
        --input_dir data/raw/other \
        --output_dir data/classifier \
        --class_name other \
        --augmentations 10
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losowe próbkowanie zdjęć z FracAtlas do folderu other/"
    )
    parser.add_argument(
        "--fracatlas_dir", type=Path, required=True,
        help="Główny katalog FracAtlas (zawierający podfoldery z RTG)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/raw/other"),
        help="Folder wyjściowy (domyślnie: data/raw/other)"
    )
    parser.add_argument(
        "--count", type=int, default=98,
        help="Liczba zdjęć do wylosowania (domyślnie: 98)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Ziarno losowości (domyślnie: 42)"
    )
    return parser.parse_args()


def collect_all_images(root: Path) -> list[Path]:
    """
    Rekurencyjnie zbiera wszystkie obsługiwane pliki obrazów z katalogu.

    Args:
        root: Główny katalog do przeszukania.

    Returns:
        Lista ścieżek do plików obrazów.
    """
    all_files = [
        f for f in root.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    ]
    return all_files


def main() -> None:
    args = parse_args()

    if not args.fracatlas_dir.exists():
        logger.error("Katalog FracAtlas nie istnieje: %s", args.fracatlas_dir)
        return

    # Zbierz wszystkie zdjęcia z FracAtlas
    logger.info("Przeszukiwanie katalogu FracAtlas: %s", args.fracatlas_dir)
    all_images = collect_all_images(args.fracatlas_dir)
    logger.info("Znaleziono łącznie %d zdjęć", len(all_images))

    if len(all_images) < args.count:
        logger.warning(
            "Za mało zdjęć (%d) — wylosowano wszystkie zamiast %d",
            len(all_images), args.count
        )
        args.count = len(all_images)

    # Losowe próbkowanie
    random.seed(args.seed)
    sampled = random.sample(all_images, args.count)

    # Pokaż z jakich podkatalogów pochodzą zdjęcia
    categories: dict[str, int] = {}
    for f in sampled:
        cat = f.parent.name
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("Rozkład po kategoriach:")
    for cat, cnt in sorted(categories.items()):
        logger.info("  %-30s %d zdjęć", cat, cnt)

    # Kopiuj do folderu wyjściowego
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in sampled:
        # Zachowaj unikalną nazwę przez dodanie nazwy kategorii jako prefiks
        out_name = f"{f.parent.name}_{f.name}"
        out_path = args.output_dir / out_name

        # Obsłuż kolizje nazw
        counter = 1
        while out_path.exists():
            out_path = args.output_dir / f"{f.parent.name}_{f.stem}_{counter}{f.suffix}"
            counter += 1

        shutil.copy2(f, out_path)
        copied += 1

    logger.info("=" * 50)
    logger.info("Skopiowano %d zdjęć do: %s", copied, args.output_dir)
    logger.info("=" * 50)
    logger.info("Następny krok — augmentacja:")
    logger.info(
        "  python scripts/augment_dataset.py "
        "--input_dir %s --output_dir data/classifier "
        "--class_name other --augmentations 10",
        args.output_dir
    )


if __name__ == "__main__":
    main()
