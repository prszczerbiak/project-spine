"""
scripts/augment_dataset.py
===========================
Skrypt augmentacji datasetu treningowego dla klasyfikatora (Etap 1).

Dla każdego oryginalnego zdjęcia generuje N augmentowanych kopii,
co pozwala rozmnożyć mały dataset (np. 90 zdjęć) do użytecznego rozmiaru.

Uruchomienie:
    python scripts/augment_dataset.py \
        --input_dir data/raw/spine_xray \
        --output_dir data/classifier/train/spine_xray \
        --augmentations 10

    python scripts/augment_dataset.py \
        --input_dir data/raw/other \
        --output_dir data/classifier/train/other \
        --augmentations 10

Struktura wyjściowa:
    data/classifier/
        train/
            spine_xray/   ← oryginały + augmentacje (80% datasetu)
            other/
        val/
            spine_xray/   ← tylko oryginały (20% datasetu)
            other/
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}


# ---------------------------------------------------------------------------
# Augmentacje
# ---------------------------------------------------------------------------

def augment_image(image: Image.Image, seed: int) -> Image.Image:
    """
    Stosuje losowy zestaw augmentacji na obrazie.

    Augmentacje dobrane pod kątem RTG kręgosłupa:
    - Obrót: klinicznie RTG bywają lekko przekrzywione
    - Odbicie poziome: lewa/prawa strona kręgosłupa
    - Jasność/kontrast: różne ustawienia aparatu RTG
    - Szum: symulacja ziarna RTG
    - Przycinanie: różne kadrowania
    - Rozmycie: różna ostrość zdjęć

    Args:
        image: Oryginalny obraz PIL.
        seed: Ziarno losowości dla reprodukowalności.

    Returns:
        Augmentowany obraz PIL.
    """
    random.seed(seed)
    np.random.seed(seed)

    img = image.copy().convert("RGB")
    w, h = img.size

    # 1. Losowy obrót ±15°
    if random.random() > 0.3:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    # 2. Odbicie poziome (50% szans)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 3. Losowe przycinanie (zoom 85–100%)
    if random.random() > 0.4:
        crop_factor = random.uniform(0.85, 1.0)
        new_w = int(w * crop_factor)
        new_h = int(h * crop_factor)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    # 4. Zmiana jasności (typowa dla różnych aparatów RTG)
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.4)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # 5. Zmiana kontrastu
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.5)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # 6. Losowy szum gaussowski (symulacja ziarna RTG)
    if random.random() > 0.5:
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(5, 20), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    # 7. Lekkie rozmycie (różna ostrość zdjęć)
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # 8. Zmiana ostrości
    if random.random() > 0.5:
        factor = random.uniform(0.8, 2.0)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    return img


# ---------------------------------------------------------------------------
# Wczytywanie obrazów (z obsługą DICOM)
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image:
    """
    Wczytuje obraz z dysku. Obsługuje JPEG, PNG oraz DICOM.

    Args:
        path: Ścieżka do pliku obrazu.

    Returns:
        Obraz PIL w trybie RGB.

    Raises:
        ValueError: Przy nieobsługiwanym formacie.
    """
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
            dcm = pydicom.dcmread(str(path))
            pixel_array = dcm.pixel_array.astype(np.float32)
            # Normalizacja do 0–255
            pixel_array -= pixel_array.min()
            if pixel_array.max() > 0:
                pixel_array /= pixel_array.max()
            pixel_array = (pixel_array * 255).astype(np.uint8)
            img = Image.fromarray(pixel_array)
            return img.convert("RGB")
        except ImportError:
            raise ImportError(
                "Zainstaluj pydicom aby obsługiwać pliki DICOM: "
                "pip install pydicom"
            )
    else:
        return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Główna logika
# ---------------------------------------------------------------------------

def process_class(
    input_dir: Path,
    train_dir: Path,
    val_dir: Path,
    augmentations_per_image: int,
    val_split: float,
    seed: int,
) -> dict[str, int]:
    """
    Przetwarza jeden folder klasy: dzieli na train/val i augmentuje train.

    Args:
        input_dir: Folder z oryginalnymi zdjęciami.
        train_dir: Folder wyjściowy dla zbioru treningowego.
        val_dir: Folder wyjściowy dla zbioru walidacyjnego.
        augmentations_per_image: Liczba augmentacji per zdjęcie (tylko train).
        val_split: Odsetek zdjęć trafiających do val (np. 0.2 = 20%).
        seed: Ziarno losowości.

    Returns:
        Słownik ze statystykami: {'original', 'train', 'val', 'augmented'}.
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Zbierz wszystkie obsługiwane pliki
    all_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        logger.warning("Brak obsługiwanych plików w: %s", input_dir)
        return {"original": 0, "train": 0, "val": 0, "augmented": 0}

    # Podział train/val
    random.seed(seed)
    random.shuffle(all_files)
    val_count = max(1, int(len(all_files) * val_split))
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    logger.info(
        "%s → %d train, %d val (łącznie %d oryginałów)",
        input_dir.name, len(train_files), len(val_files), len(all_files)
    )

    # --- Kopiuj do val (bez augmentacji) ---
    for f in val_files:
        try:
            img = load_image(f)
            out_path = val_dir / f"{f.stem}.jpg"
            img.save(out_path, "JPEG", quality=95)
        except Exception as exc:
            logger.error("Błąd przy %s: %s", f.name, exc)

    # --- Kopiuj oryginały do train + augmentuj ---
    augmented_count = 0
    for idx, f in enumerate(train_files):
        try:
            img = load_image(f)
        except Exception as exc:
            logger.error("Błąd wczytywania %s: %s", f.name, exc)
            continue

        # Zapisz oryginał
        orig_out = train_dir / f"{f.stem}_orig.jpg"
        img.save(orig_out, "JPEG", quality=95)

        # Generuj augmentacje
        for aug_idx in range(augmentations_per_image):
            aug_seed = seed + idx * 1000 + aug_idx
            try:
                aug_img = augment_image(img, seed=aug_seed)
                aug_out = train_dir / f"{f.stem}_aug{aug_idx:03d}.jpg"
                aug_img.save(aug_out, "JPEG", quality=90)
                augmented_count += 1
            except Exception as exc:
                logger.error("Błąd augmentacji %s (aug %d): %s", f.name, aug_idx, exc)

        if (idx + 1) % 10 == 0:
            logger.info("  Przetworzono %d/%d zdjęć...", idx + 1, len(train_files))

    total_train = len(train_files) + augmented_count
    return {
        "original": len(all_files),
        "train": total_train,
        "val": len(val_files),
        "augmented": augmented_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmentacja datasetu treningowego PROJECT: SPINE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

  # Augmentuj RTG kręgosłupa
  python scripts/augment_dataset.py \\
      --input_dir data/raw/spine_xray \\
      --output_dir data/classifier \\
      --class_name spine_xray \\
      --augmentations 10

  # Augmentuj negatywy
  python scripts/augment_dataset.py \\
      --input_dir data/raw/other \\
      --output_dir data/classifier \\
      --class_name other \\
      --augmentations 10
        """,
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Folder z oryginalnymi zdjęciami jednej klasy"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/classifier"),
        help="Główny folder wyjściowy datasetu (domyślnie: data/classifier)"
    )
    parser.add_argument(
        "--class_name", type=str, required=True,
        choices=["spine_xray", "other"],
        help="Nazwa klasy: 'spine_xray' lub 'other'"
    )
    parser.add_argument(
        "--augmentations", type=int, default=10,
        help="Liczba augmentacji per zdjęcie (domyślnie: 10)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Odsetek zdjęć do walidacji (domyślnie: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Ziarno losowości dla reprodukowalności (domyślnie: 42)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        logger.error("Folder wejściowy nie istnieje: %s", args.input_dir)
        return

    train_dir = args.output_dir / "train" / args.class_name
    val_dir = args.output_dir / "val" / args.class_name

    logger.info("=" * 60)
    logger.info("PROJECT: SPINE — Augmentacja datasetu")
    logger.info("=" * 60)
    logger.info("Klasa:         %s", args.class_name)
    logger.info("Wejście:       %s", args.input_dir)
    logger.info("Train wyjście: %s", train_dir)
    logger.info("Val wyjście:   %s", val_dir)
    logger.info("Augmentacje:   %d per zdjęcie", args.augmentations)
    logger.info("Val split:     %.0f%%", args.val_split * 100)
    logger.info("=" * 60)

    stats = process_class(
        input_dir=args.input_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        augmentations_per_image=args.augmentations,
        val_split=args.val_split,
        seed=args.seed,
    )

    logger.info("=" * 60)
    logger.info("PODSUMOWANIE:")
    logger.info("  Oryginałów wejściowych: %d", stats["original"])
    logger.info("  Zdjęć treningowych:     %d (oryginały + augmentacje)", stats["train"])
    logger.info("  Zdjęć walidacyjnych:    %d", stats["val"])
    logger.info("  Wygenerowanych augm.:   %d", stats["augmented"])
    logger.info("=" * 60)
    logger.info("Gotowe! Możesz teraz uruchomić trening:")
    logger.info(
        "  python scripts/train_classifier.py "
        "--data_dir %s --output_dir models --epochs 30",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
