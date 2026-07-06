"""Tests for the rolling-origin backtest."""

import numpy as np
import pandas as pd

from demand_forecasting.backtest import rolling_backtest, summarize
from demand_forecasting.models import seasonal_naive_forecast


def _series(n=120):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(50 + np.arange(n) * 0.1, index=idx)


def test_backtest_produces_one_row_per_fold() -> None:
    y = _series()
    bt = rolling_backtest(y, lambda tr, h: seasonal_naive_forecast(tr, h, 7), horizon=10, n_folds=3)
    assert len(bt) == 3
    assert {"rmse", "mae", "mape", "smape", "fold", "cutoff"} <= set(bt.columns)


def test_summarize_returns_metric_means() -> None:
    y = _series()
    bt = rolling_backtest(y, lambda tr, h: seasonal_naive_forecast(tr, h, 7), horizon=10, n_folds=2)
    s = summarize(bt)
    assert set(s) == {"rmse", "mae", "mape", "smape"}
    assert all(np.isfinite(v) for v in s.values())
