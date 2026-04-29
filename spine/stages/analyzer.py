"""
spine.stages.analyzer
=========================
Etap 3: Analiza geometryczna i klasyfikacja medyczna.

Odpowiada za:
    1. Wyliczenie kąta Cobba (Cobb angle) na podstawie punktów kluczowych.
    2. Klasyfikację stopnia skoliozy przy użyciu reguł klinicznych.

Terminologia medyczna:
    - Krąg krańcowy górny (UIV): najbardziej pochylony krąg na szczycie łuku.
    - Krąg krańcowy dolny (LIV): najbardziej pochylony krąg u podstawy łuku.
    - Kąt Cobba: kąt między górną płytką UIV a dolną płytką LIV.

Format punktów kluczowych (AASCE convention):
    68 punktów = 17 kręgów × 4 narożniki
    Dla kręgu k: indeksy 4k, 4k+1, 4k+2, 4k+3
        - 4k   = top-left  (TL) — lewy górny
        - 4k+1 = top-right (TR) — prawy górny
        - 4k+2 = bottom-left (BL) — lewy dolny
        - 4k+3 = bottom-right (BR) — prawy dolny

Kąt Cobba — poprawna metoda:
    1. Dla każdego kręgu oblicz kąt nachylenia górnej płytki (TL→TR).
    2. Znajdź punkt infleksji krzywizny — miejsce gdzie nachylenie
       zmienia kierunek (z lewostronnego na prawostronny lub odwrotnie).
    3. UIV = krąg o maksymalnym nachyleniu powyżej infleksji.
    4. LIV = krąg o maksymalnym nachyleniu poniżej infleksji.
    5. Kąt Cobba = kąt między górną płytką UIV a dolną płytką LIV.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from spine.config import SpineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometria — kąt Cobba
# ---------------------------------------------------------------------------

def compute_cobb_angle(
    upper_endplate_left: np.ndarray,
    upper_endplate_right: np.ndarray,
    lower_endplate_left: np.ndarray,
    lower_endplate_right: np.ndarray,
) -> float:
    """
    Wylicza kąt Cobba między dwiema płytkami granicznymi kręgów.

    Args:
        upper_endplate_left:  Lewy punkt górnej płytki UIV  [x, y].
        upper_endplate_right: Prawy punkt górnej płytki UIV [x, y].
        lower_endplate_left:  Lewy punkt dolnej płytki LIV  [x, y].
        lower_endplate_right: Prawy punkt dolnej płytki LIV [x, y].

    Returns:
        Kąt Cobba w stopniach (0–90°).

    Raises:
        ValueError: Jeśli wektory są zdegenerowane (zerowej długości).
    """
    vec_upper: np.ndarray = upper_endplate_right - upper_endplate_left
    vec_lower: np.ndarray = lower_endplate_right - lower_endplate_left

    len_upper = float(np.linalg.norm(vec_upper))
    len_lower = float(np.linalg.norm(vec_lower))

    if len_upper < 1e-6 or len_lower < 1e-6:
        raise ValueError(
            "Zdegenerowany wektor płytki — punkty zbieżne lub identyczne."
        )

    vec_upper_n = vec_upper / len_upper
    vec_lower_n = vec_lower / len_lower

    cos_angle = np.clip(np.dot(vec_upper_n, vec_lower_n), -1.0, 1.0)
    angle_between_deg = float(np.degrees(np.arccos(cos_angle)))

    cobb = 180.0 - angle_between_deg if angle_between_deg > 90.0 else angle_between_deg

    logger.debug("compute_cobb_angle → %.2f°", cobb)
    return cobb


def compute_cobb_from_landmarks(
    landmarks: np.ndarray,
    upper_vert_idx: int,
    lower_vert_idx: int,
) -> float:
    """
    Wylicza kąt Cobba dla pary kręgów krańcowych.

    Używa górnej płytki UIV (TL, TR) i dolnej płytki LIV (BL, BR).

    Args:
        landmarks: ndarray (68, 2) — punkty kluczowe [x, y].
        upper_vert_idx: Indeks kręgu UIV (0-based, 0=Th1, 16=L5).
        lower_vert_idx: Indeks kręgu LIV (0-based).

    Returns:
        Kąt Cobba w stopniach.
    """
    ui = upper_vert_idx
    li = lower_vert_idx

    # Górna płytka UIV
    uiv_tl = landmarks[4 * ui]        # top-left
    uiv_tr = landmarks[4 * ui + 1]    # top-right

    # Dolna płytka LIV
    liv_bl = landmarks[4 * li + 2]    # bottom-left
    liv_br = landmarks[4 * li + 3]    # bottom-right

    return compute_cobb_angle(uiv_tl, uiv_tr, liv_bl, liv_br)


def _get_endplate_tilt(landmarks: np.ndarray, vert_idx: int) -> float:
    """
    Oblicza kąt nachylenia górnej płytki kręgu względem poziomej.

    Kąt dodatni = płytka pochylona w prawo (góra-prawo)
    Kąt ujemny  = płytka pochylona w lewo  (góra-lewo)

    Args:
        landmarks: ndarray (68, 2).
        vert_idx: Indeks kręgu (0-based).

    Returns:
        Kąt nachylenia w stopniach (-90 do +90).
    """
    tl = landmarks[4 * vert_idx]
    tr = landmarks[4 * vert_idx + 1]
    vec = tr - tl
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def find_endplate_vertebrae(
    landmarks: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Wyznacza pary kręgów krańcowych (UIV, LIV) metodą infleksji krzywizny.

    Algorytm:
        1. Oblicz kąt nachylenia górnej płytki dla każdego kręgu.
        2. Znajdź punkty infleksji — miejsca gdzie nachylenie zmienia znak
           (krzywa przechodzi z lewostronnej na prawostronnną lub odwrotnie).
        3. Każdy łuk skoliotyczny to segment między dwoma infleksjami.
        4. UIV = krąg o max nachyleniu w górnej części łuku.
        5. LIV = krąg o max nachyleniu w dolnej części łuku.

    Args:
        landmarks: ndarray (68, 2) — punkty kluczowe w pikselach.

    Returns:
        Lista krotek [(uiv_idx, liv_idx), ...] — jedna krotka per łuk.
        Zazwyczaj 1–3 łuki dla skoliozy piersiowo-lędźwiowej.
    """
    num_vertebrae = len(landmarks) // 4

    # Kąty nachylenia górnych płytek (ze znakiem)
    tilts = np.array([
        _get_endplate_tilt(landmarks, k)
        for k in range(num_vertebrae)
    ])

    logger.debug("Nachylenia kręgów: %s", np.round(tilts, 2))

    # Znajdź punkty infleksji (zmiana znaku nachylenia)
    # Używamy znormalizowanego sygnału nachylenia
    tilt_signs = np.sign(tilts)
    inflection_points = [0]  # zawsze zaczynamy od pierwszego kręgu

    for i in range(1, num_vertebrae):
        if tilt_signs[i] != tilt_signs[i - 1] and tilt_signs[i] != 0:
            inflection_points.append(i)

    inflection_points.append(num_vertebrae)  # zawsze kończymy na ostatnim

    logger.debug("Punkty infleksji: %s", inflection_points)

    # Wyznacz pary UIV/LIV dla każdego segmentu między infleksjami
    pairs: list[tuple[int, int]] = []

    for seg_start, seg_end in zip(inflection_points[:-1], inflection_points[1:]):
        segment_tilts = np.abs(tilts[seg_start:seg_end])

        if len(segment_tilts) < 2:
            continue

        # UIV = krąg o max nachyleniu w górnej połowie segmentu
        half = len(segment_tilts) // 2
        upper_half = segment_tilts[:max(half, 1)]
        lower_half = segment_tilts[max(half, 1):]

        if len(upper_half) == 0 or len(lower_half) == 0:
            continue

        uiv_local = int(np.argmax(upper_half))
        liv_local = int(np.argmax(lower_half)) + max(half, 1)

        uiv_global = seg_start + uiv_local
        liv_global = seg_start + liv_local

        if uiv_global != liv_global:
            pairs.append((uiv_global, liv_global))

    # Fallback: jeśli nie znaleziono par, użyj pierwszego i ostatniego kręgu
    if not pairs:
        logger.warning(
            "Nie znaleziono punktów infleksji — używam kręgów skrajnych."
        )
        pairs = [(0, num_vertebrae - 1)]

    logger.debug("Pary UIV/LIV: %s", pairs)
    return pairs


# ---------------------------------------------------------------------------
# Klasyfikacja medyczna
# ---------------------------------------------------------------------------

def classify_scoliosis(cobb_angle_deg: float) -> tuple[str, str]:
    """
    Klasyfikuje stopień skoliozy według kąta Cobba (wg SRS).

    Progi kliniczne:
        < 10°  → Brak
        10–25° → Łagodna
        25–40° → Umiarkowana
        > 40°  → Ciężka

    Args:
        cobb_angle_deg: Kąt Cobba w stopniach.

    Returns:
        Krotka (grade_pl, recommendation_pl).
    """
    if cobb_angle_deg < 10.0:
        return "Brak", "Obserwacja — kontrola raz na rok."
    elif cobb_angle_deg < 25.0:
        return "Łagodna", "Fizjoterapia; kontrola co 6 miesięcy."
    elif cobb_angle_deg < 40.0:
        return "Umiarkowana", "Rozważyć leczenie gorsetem; konsultacja ortopedyczna."
    else:
        return "Ciężka", "Pilna konsultacja ortopedyczna; rozważyć korekcję chirurgiczną."


# ---------------------------------------------------------------------------
# Future-proofing: progi widoku bocznego
# ---------------------------------------------------------------------------

LATERAL_THRESHOLDS: dict[str, dict[str, float]] = {
    "kyphosis": {
        "normal_min": 20.0,
        "normal_max": 40.0,
        "hyperkyphosis_threshold": 40.0,
        "hypokyphosis_threshold": 20.0,
    },
    "lordosis_lumbar": {
        "normal_min": 30.0,
        "normal_max": 50.0,
        "hyperlordosis_threshold": 50.0,
        "hypolordosis_threshold": 30.0,
    },
    "lordosis_cervical": {
        "normal_min": 20.0,
        "normal_max": 40.0,
    },
}


# ---------------------------------------------------------------------------
# Klasa główna etapu 3
# ---------------------------------------------------------------------------

class SpinalAnalyzer:
    """
    Analizator geometryczny i klasyfikator medyczny (Etap 3 pipeline'u).

    Args:
        config: Konfiguracja systemu PROJECT: SPINE.
    """

    def __init__(self, config: SpineConfig) -> None:
        self._cfg = config
        self._clf: Optional[DecisionTreeClassifier] = None

    def analyze(
        self,
        landmarks: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[Optional[float], Optional[str], str]:
        """
        Przeprowadza pełną analizę geometryczną i klasyfikację.

        Dla skoliozy wielołukowej zwraca wynik dla łuku o największym kącie
        (tzw. łuk główny — primary curve).

        Args:
            landmarks: ndarray (68, 2) z wykrytymi punktami [x, y] w pikselach.
            scores: ndarray (68,) z pewnościami punktów.

        Returns:
            Krotka (cobb_angle_deg, grade, notes).
        """
        n_pts = len(landmarks)
        if n_pts < 8:
            return None, None, f"Za mało punktów ({n_pts}); potrzeba co najmniej 8."

        # Wyznacz pary kręgów krańcowych
        pairs = find_endplate_vertebrae(landmarks)

        # Oblicz kąt Cobba dla każdego łuku
        cobb_angles: list[tuple[float, int, int]] = []  # (kąt, uiv, liv)

        for uiv, liv in pairs:
            try:
                angle = compute_cobb_from_landmarks(landmarks, uiv, liv)
                cobb_angles.append((angle, uiv, liv))
                logger.info(
                    "Łuk Th%d–L%d: kąt Cobba = %.1f°",
                    uiv + 1, liv + 1, angle
                )
            except (ValueError, IndexError) as exc:
                logger.warning("Błąd obliczenia kąta dla pary (%d, %d): %s", uiv, liv, exc)

        if not cobb_angles:
            return None, None, "Nie udało się obliczyć kąta Cobba."

        # Łuk główny = największy kąt
        max_cobb, main_uiv, main_liv = max(cobb_angles, key=lambda x: x[0])

        grade, recommendation = classify_scoliosis(max_cobb)

        # Zbuduj szczegółowe notatki
        notes_parts = [recommendation]
        if len(cobb_angles) > 1:
            arcs = ", ".join(
                f"Th{u+1}–L{l+1}: {a:.1f}°"
                for a, u, l in sorted(cobb_angles, key=lambda x: x[1])
            )
            notes_parts.append(f"Wykryto {len(cobb_angles)} łuki: {arcs}")

        notes = " | ".join(notes_parts)

        # DataFrame dla debugowania
        df = self._landmarks_to_dataframe(landmarks, scores)
        logger.debug("Ramka punktów:\n%s", df.head())

        return max_cobb, grade, notes

    def analyze_lateral(
        self,
        landmarks: np.ndarray,
        scores: np.ndarray,
        curve_type: str = "kyphosis",
    ) -> tuple[Optional[float], Optional[str], str]:
        """
        ZAREZERWOWANE — Analiza widoku bocznego (lordoza / kifoza).

        Raises:
            NotImplementedError: Zawsze — metoda czeka na implementację.
        """
        raise NotImplementedError(
            "Analiza widoku bocznego nie jest jeszcze zaimplementowana. "
            "Aby dodać: (1) LandmarkDetectorLateral, "
            "(2) obliczenia wg LATERAL_THRESHOLDS, "
            "(3) podmień NotImplementedError."
        )

    @staticmethod
    def _landmarks_to_dataframe(
        landmarks: np.ndarray,
        scores: np.ndarray,
    ) -> pd.DataFrame:
        """Konwertuje tablice NumPy do Pandas DataFrame."""
        n = len(landmarks)
        corners = ["top_left", "top_right", "bottom_left", "bottom_right"]

        return pd.DataFrame({
            "point_id": range(n),
            "x": landmarks[:, 0],
            "y": landmarks[:, 1],
            "score": scores,
            "vertebra_idx": [i // 4 for i in range(n)],
            "corner": [corners[i % 4] for i in range(n)],
            "tilt_deg": [
                _get_endplate_tilt(landmarks, i // 4) if i % 4 == 0 else None
                for i in range(n)
            ],
        })