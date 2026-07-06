"""Tests for model search and baselines."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.models import default_pdq, grid_search_aic, seasonal_naive_forecast


def _series(n=60):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(50 + np.arange(n) * 0.1, index=idx)


def test_default_pdq_size() -> None:
    assert len(default_pdq()) == 18


def test_seasonal_naive_repeats_last_season() -> None:
    s = pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float)
    out = seasonal_naive_forecast(s, horizon=10, season_length=7)
    assert len(out) == 10
    assert list(out[:7]) == [1, 2, 3, 4, 5, 6, 7]
    assert out[7] == 1


def test_seasonal_naive_short_series() -> None:
    s = pd.Series([9.0, 8.0])
    out = seasonal_naive_forecast(s, horizon=4, season_length=7)
    assert len(out) == 4


def test_grid_search_returns_valid_order() -> None:
    pytest.importorskip("statsmodels")
    best = grid_search_aic(_series(), pdq_list=[(1, 1, 1)], seasonal_period=None)
    assert best["order"] == (1, 1, 1)
    assert np.isfinite(best["aic"])
