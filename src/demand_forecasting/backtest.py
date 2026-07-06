"""Rolling-origin (expanding-window) backtesting."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from demand_forecasting.metrics import mae, mape, rmse, smape

Forecaster = Callable[[pd.Series, int], np.ndarray]


def rolling_backtest(
    y: pd.Series,
    forecaster: Forecaster,
    horizon: int,
    n_folds: int = 3,
    step: int | None = None,
) -> pd.DataFrame:
    """Expanding-window backtest of ``forecaster`` over ``n_folds`` folds.

    For each fold the model trains on all data up to the cutoff and forecasts
    the next ``horizon`` points, which are scored against the held-out actuals.
    Returns one row per fold with rmse/mae/mape/smape and the cutoff index.
    """
    step = step or horizon
    n = len(y)
    rows = []
    for k in range(n_folds, 0, -1):
        cutoff = n - k * step
        if cutoff <= 0 or cutoff + horizon > n:
            continue
        train = y.iloc[:cutoff]
        actual = y.iloc[cutoff : cutoff + horizon]
        pred = np.asarray(forecaster(train, horizon), dtype=float)[: len(actual)]
        a = actual.to_numpy(dtype=float)
        rows.append(
            {
                "fold": n_folds - k + 1,
                "cutoff": int(cutoff),
                "rmse": rmse(a, pred),
                "mae": mae(a, pred),
                "mape": mape(a, pred),
                "smape": smape(a, pred),
            }
        )
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict[str, float]:
    """Mean of each metric column across folds."""
    metric_cols = ["rmse", "mae", "mape", "smape"]
    return {c: float(bt[c].mean()) for c in metric_cols if c in bt}
