# PROJECT: SPINE — System spine

> Automatyczna analiza RTG kręgosłupa z detekcją punktów kluczowych i obliczeniem kąta Cobba.

---

## Struktura projektu

```
project_spine/
│
├── spine_app.py                  # Punkt wejścia — Streamlit UI
│
├── spine/                        # Główny pakiet systemu spine
│   ├── __init__.py
│   ├── config.py                    # spineConfig — centralna konfiguracja
│   ├── pipeline.py                  # SpineAnalyzer + AnalysisResult (orkiestrator)
│   │
│   └── stages/                      # Niezależne etapy pipeline'u
│       ├── __init__.py
│       ├── verifier.py              # Etap 1: ImageVerifier (ResNet-18)
│       ├── detector.py              # Etap 2: LandmarkDetector (HRNet-W32)
│       └── analyzer.py              # Etap 3: SpinalAnalyzer (Cobb + sklearn)
│
├── scripts/
│   └── train_classifier.py          # Skrypt treningu klasyfikatora (Etap 1)
│
├── tests/
│   └── test_cobb_angle.py           # Testy jednostkowe geometrii
│
├── models/                          # Wagi modeli (.pth) — NIE commitować do git
│   ├── classifier_resnet18.pth
│   └── landmark_hrnet.pth
│
├── data/                            # Dane treningowe — NIE commitować do git
│   ├── classifier/
│   │   ├── train/
│   │   │   ├── spine_xray/
│   │   │   └── other/
│   │   └── val/
│   │       ├── spine_xray/
│   │       └── other/
│   └── landmarks/                   # Dataset AASCE lub własny
│       ├── images/
│       └── annotations/
│
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Architektura ML

### Etap 1: Weryfikacja obrazu — ResNet-18

**Cel:** Klasyfikacja binarna `spine_xray` / `other`.

**Uzasadnienie:** ResNet-18 (pre-trenowany ImageNet) to optymalny wybór:
- Wymagana tylko wymiana ostatniej warstwy FC (2 klasy zamiast 1000).
- Fine-tuning na ~2000–5000 zdjęciach RTG daje >95% accuracy.
- Lekki (~45MB), działa na CPU w <100ms.

**Alternatywy:**
| Model | Accuracy | Rozmiar | Latency CPU |
|---|---|---|---|
| ResNet-18 ✅ | ~95% | 45MB | ~80ms |
| EfficientNet-B0 | ~97% | 20MB | ~60ms |
| MobileNetV3-Small | ~92% | 10MB | ~30ms |

Dla środowisk z GPU → upgrade do ResNet-50 lub EfficientNet-B3.

---

### Etap 2: Detekcja punktów kluczowych — HRNet-W32

**Cel:** Lokalizacja 4 narożników każdego z 17 kręgów = 68 punktów.

**Uzasadnienie:** HRNet (High-Resolution Network) utrzymuje wysoką rozdzielczość
przestrzenną przez cały forward pass (w przeciwieństwie do sieci encoder-decoder),
co jest kluczowe dla precyzji na poziomie 1–2mm.

**Wagi startowe:** COCO Keypoints → fine-tune na AASCE Challenge Dataset.

**Alternatywy:**
| Model | SMAPE ↓ | Złożoność |
|---|---|---|
| HRNet-W32 ✅ | ~6.1% | Wysoka |
| Stacked Hourglass | ~7.8% | Średnia |
| Keypoint R-CNN | ~9.2% | Niska (out-of-box) |
| UNet + heatmap | ~7.0% | Niska–Średnia |

Na start (bez GPU/dużego datasetu): **Keypoint R-CNN** z torchvision —
gotowy do użycia bez konfiguracji.

---

## Datasety

### Etap 1 — Klasyfikator
- **SpineWeb** (spineimage.cs.uchicago.edu): RTG + CT kręgosłupa.
- **NIH Chest X-ray Dataset**: negatywy (RTG klatki piersiowej ≠ kręgosłup).
- **RSNA datasets**: kaggle.com/competitions/rsna-* — mix RTG.
- **Własne negatywy**: ImageNet sample, MRI, RTG kończyn.

### Etap 2 — Detekcja punktów
- **AASCE Challenge Dataset** (Accurate Automated Spinal Curvature Estimation):
  - 609 RTG kręgosłupa AP z annotacjami 68 punktów kluczowych.
  - https://aasce19.github.io
  - Benchmark: SMAPE kąta Cobba ~6%.
- **BoostNet Dataset**: 481 RTG z punktami + kąty Cobba.
  - https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe

---

## Uruchomienie

```bash
# Instalacja zależności
pip install -e ".[dev]"

# Kopiuj i uzupełnij konfigurację
cp .env.example .env

# Uruchom aplikację
streamlit run spine_app.py

# Testy
pytest tests/ -v

# Trening klasyfikatora (po przygotowaniu danych)
python scripts/train_classifier.py \
    --data_dir data/classifier \
    --output_dir models \
    --epochs 30 \
    --device cuda
```

---

## Zmienne środowiskowe (.env)

```env
spine_DEVICE=cpu                              # cpu | cuda | mps
spine_MODEL_DIR=models
spine_CLASSIFIER_WEIGHTS=models/classifier_resnet18.pth
spine_LANDMARK_WEIGHTS=models/landmark_hrnet.pth
spine_CLASSIFIER_THRESHOLD=0.75
spine_LANDMARK_SCORE_THRESHOLD=0.5
spine_NUM_LANDMARKS=68
spine_LATERAL_VIEW=false                      # przyszła analiza boczna
```

---

## Rozszerzenie o widok boczny (lordoza/kifoza)

Architektura jest przygotowana na rozbudowę. Kroki:

1. **Nowe dane:** Zebrać RTG boczne z annotacjami punktów.
2. **Nowy detektor:** `LandmarkDetectorLateral` w `stages/detector_lateral.py`
   (analogiczny do `LandmarkDetector`, inne `num_landmarks`).
3. **Nowa analiza:** Odblokować `SpinalAnalyzer.analyze_lateral()` i zaimplementować
   obliczenia wg `LATERAL_THRESHOLDS` zdefiniowanych w `analyzer.py`.
4. **UI:** Dodać zakładkę "Widok boczny" w `spine_app.py`.
5. **Config:** `spine_LATERAL_VIEW=true`.

---

## ⚠️ Zastrzeżenie medyczne

System spine jest narzędziem wspomagającym. Wyniki analizy
mają charakter pomocniczy i muszą być zweryfikowane przez
uprawnionego lekarza specjalistę (ortopeda, radiolog).
System nie zastępuje diagnozy medycznej.
