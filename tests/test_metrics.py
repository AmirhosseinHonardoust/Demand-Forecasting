"""Unit tests for error metrics."""

import numpy as np

from metrics import mape, rmse


def test_rmse_zero_on_perfect_prediction() -> None:
    y = [1.0, 2.0, 3.0, 4.0]
    assert rmse(y, y) == 0.0


def test_rmse_known_value() -> None:
    # Errors are all 2.0 -> RMSE == 2.0
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 22.0, 32.0]
    assert rmse(y_true, y_pred) == 2.0


def test_mape_zero_on_perfect_prediction() -> None:
    y = [5.0, 10.0, 20.0]
    assert mape(y, y) == 0.0


def test_mape_known_value() -> None:
    # 10% error on every element -> MAPE == 10.0
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = y_true * 1.1
    assert abs(mape(y_true, y_pred) - 10.0) < 1e-6


def test_mape_zero_denominator_is_finite() -> None:
    # Guard (1e-9 floor) keeps the result finite when a true value is 0.
    assert np.isfinite(mape([0.0, 10.0], [1.0, 10.0]))
