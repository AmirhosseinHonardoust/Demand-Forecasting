"""Model search, SARIMA fitting, and baseline forecasters."""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

PDQ = tuple[int, int, int]
SeasonalOrder = tuple[int, int, int, int] | None


def default_pdq() -> list[PDQ]:
    """Default (p, d, q) grid: p, q in {0, 1, 2}, d in {0, 1}."""
    return [(p, d, q) for p in [0, 1, 2] for d in [0, 1] for q in [0, 1, 2]]


def grid_search_aic(
    y: pd.Series,
    pdq_list: list[PDQ],
    seasonal_period: int | None = None,
) -> dict[str, object]:
    """Return the lowest-AIC SARIMAX configuration over ``pdq_list``.

    When ``seasonal_period`` is set, each (p, d, q) is combined with seasonal
    (P, D, Q) in {0, 1}^3 at that period. Fits that fail to converge are logged
    and skipped rather than aborting the search.
    """
    best: dict[str, object] = {"aic": np.inf, "order": None, "seasonal_order": None}
    for p, d, q in pdq_list:
        seasonal_candidates: list[SeasonalOrder]
        if seasonal_period:
            seasonal_candidates = [
                (P, D, Q, seasonal_period) for P in [0, 1] for D in [0, 1] for Q in [0, 1]
            ]
        else:
            seasonal_candidates = [None]
        for order_s in seasonal_candidates:
            try:
                model = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=order_s,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=500)
            except Exception as exc:  # noqa: BLE001 - fit failures are expected
                logger.debug("SARIMAX fit failed order=%s seasonal=%s: %s", (p, d, q), order_s, exc)
                continue
            if res.aic < best["aic"]:
                best = {"aic": float(res.aic), "order": (p, d, q), "seasonal_order": order_s}
    return best


def fit_sarima(y: pd.Series, order: PDQ, seasonal_order: SeasonalOrder):
    """Fit a SARIMAX model with the standard (non-enforcing) settings."""
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=500)


def sarima_forecast(train: pd.Series, horizon: int, order: PDQ, seasonal_order: SeasonalOrder):
    """Fit SARIMA on ``train`` and return an ``horizon``-step point forecast."""
    res = fit_sarima(train, order, seasonal_order)
    return np.asarray(res.get_forecast(steps=horizon).predicted_mean, dtype=float)


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    """Baseline: repeat the last ``season_length`` observations across the horizon."""
    values = np.asarray(train, dtype=float)
    if values.size < season_length:
        season_length = values.size
    last_season = values[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]
