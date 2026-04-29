"""
tests/test_cobb_angle.py
=========================
Testy jednostkowe dla funkcji compute_cobb_angle.

Uruchomienie:
    pytest tests/test_cobb_angle.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from spine.stages.analyzer import compute_cobb_angle, classify_scoliosis


class TestCobbAngle:
    """Testy geometryczne funkcji compute_cobb_angle."""

    def test_parallel_plates_zero_angle(self) -> None:
        """Dwie równoległe płytki → kąt = 0°."""
        angle = compute_cobb_angle(
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
            np.array([0.0, 100.0]),
            np.array([100.0, 100.0]),
        )
        assert abs(angle) < 0.01, f"Oczekiwano ≈0°, otrzymano {angle:.4f}°"

    def test_perpendicular_plates_90_degrees(self) -> None:
        """Płytki prostopadłe do siebie → kąt = 90°."""
        angle = compute_cobb_angle(
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),   # pozioma
            np.array([0.0, 0.0]),
            np.array([0.0, 100.0]),   # pionowa
        )
        assert abs(angle - 90.0) < 0.01, f"Oczekiwano ≈90°, otrzymano {angle:.4f}°"

    def test_known_angle_30_degrees(self) -> None:
        """Płytka dolna nachylona o 30° → kąt Cobba ≈ 30°."""
        angle_rad = np.radians(30.0)
        angle = compute_cobb_angle(
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([100.0 * np.cos(angle_rad), 100.0 * np.sin(angle_rad)]),
        )
        assert abs(angle - 30.0) < 0.5, f"Oczekiwano ≈30°, otrzymano {angle:.4f}°"

    def test_degenerate_point_raises(self) -> None:
        """Zdegenerowany wektor (identyczne punkty) → ValueError."""
        with pytest.raises(ValueError, match="Zdegenerowany"):
            compute_cobb_angle(
                np.array([50.0, 50.0]),
                np.array([50.0, 50.0]),  # identyczny punkt!
                np.array([0.0, 200.0]),
                np.array([100.0, 200.0]),
            )

    def test_symmetry(self) -> None:
        """Kąt powinien być symetryczny (zamiana UIV↔LIV nie zmienia wyniku)."""
        pts = [
            np.array([0.0, 0.0]),
            np.array([100.0, 5.0]),
            np.array([0.0, 200.0]),
            np.array([100.0, 215.0]),
        ]
        angle_ab = compute_cobb_angle(pts[0], pts[1], pts[2], pts[3])
        angle_ba = compute_cobb_angle(pts[2], pts[3], pts[0], pts[1])
        assert abs(angle_ab - angle_ba) < 0.01


class TestClassifyScoliosis:
    """Testy klasyfikacji stopnia skoliozy."""

    @pytest.mark.parametrize("angle,expected_grade", [
        (5.0,  "Brak"),
        (9.9,  "Brak"),
        (10.0, "Łagodna"),
        (24.9, "Łagodna"),
        (25.0, "Umiarkowana"),
        (39.9, "Umiarkowana"),
        (40.0, "Ciężka"),
        (65.0, "Ciężka"),
    ])
    def test_grade_boundaries(self, angle: float, expected_grade: str) -> None:
        grade, _ = classify_scoliosis(angle)
        assert grade == expected_grade, (
            f"Kąt {angle}°: oczekiwano '{expected_grade}', otrzymano '{grade}'"
        )

    def test_returns_recommendation(self) -> None:
        """Klasyfikacja powinna zawsze zwracać niepustą rekomendację."""
        for angle in [5.0, 15.0, 30.0, 50.0]:
            _, recommendation = classify_scoliosis(angle)
            assert recommendation, f"Pusta rekomendacja dla kąta {angle}°"
