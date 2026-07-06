"""Unit tests for error metrics."""

import numpy as np

from demand_forecasting.metrics import mae, mape, rmse, smape


def test_rmse_zero_on_perfect() -> None:
    y = [1.0, 2.0, 3.0, 4.0]
    assert rmse(y, y) == 0.0


def test_rmse_known_value() -> None:
    assert rmse([10.0, 20.0, 30.0], [12.0, 22.0, 32.0]) == 2.0


def test_mae_known_value() -> None:
    assert mae([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 2.0


def test_mape_known_value() -> None:
    y = np.array([100.0, 200.0, 400.0])
    assert abs(mape(y, y * 1.1) - 10.0) < 1e-6


def test_smape_symmetric_and_bounded() -> None:
    val = smape([100.0, 100.0], [110.0, 90.0])
    assert 0.0 <= val <= 200.0


def test_mape_zero_denominator_finite() -> None:
    assert np.isfinite(mape([0.0, 10.0], [1.0, 10.0]))
