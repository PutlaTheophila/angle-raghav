"""Unit tests for the pure geometry that must match the paper exactly.

Run:  python -m pytest test_tracker.py -q
  or: python test_tracker.py
"""
import math

from tracker import chord_sagitta_radius, compute_strain


def test_semicircle_recovers_radius():
    # A semicircle: chord L = 2R, sagitta h = R  ->  R recovered exactly.
    R = 1.0675                       # mm  (L ~ 2.135 mm, the paper's scale)
    assert math.isclose(chord_sagitta_radius(2 * R, R), R, rel_tol=1e-9)


def test_shallow_arc_recovers_radius():
    # Known circle: h = R - sqrt(R^2 - (L/2)^2) must invert back to R.
    R, L = 5.0, 2.135
    h = R - math.sqrt(R ** 2 - (L / 2) ** 2)
    assert math.isclose(chord_sagitta_radius(L, h), R, rel_tol=1e-6)


def test_degenerate_is_infinite():
    assert chord_sagitta_radius(2.0, 0.0) == float("inf")


def test_strain_formula():
    # epsilon = t / (2R) * 100
    assert math.isclose(compute_strain(0.010, 1.0675),
                        0.010 / (2 * 1.0675) * 100, rel_tol=1e-9)


def test_strain_zero_radius():
    assert compute_strain(0.01, 0.0) == 0.0


if __name__ == "__main__":
    test_semicircle_recovers_radius()
    test_shallow_arc_recovers_radius()
    test_degenerate_is_infinite()
    test_strain_formula()
    test_strain_zero_radius()
    print("all geometry tests passed")
