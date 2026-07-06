"""ARIMA/SARIMA demand forecasting pipeline.

Fits a grid of SARIMAX models selected by AIC, evaluates on a hold-out
validation window (RMSE/MAPE), writes a horizon forecast with confidence
intervals, and renders history/forecast and residual diagnostic plots.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ensure sibling modules are importable regardless of how this script is
# invoked (direct run, `python -m`, or imported by the test suite).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import mape, rmse  # noqa: E402

logger = logging.getLogger(__name__)

PDQ = tuple[int, int, int]
SeasonalOrder = tuple[int, int, int, int] | None


def ensure_outdir(path: str) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_series(path: str) -> pd.DataFrame:
    """Load a daily ``date,sales`` CSV into a gap-filled, daily-indexed frame."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date").asfreq("D")
    df["sales"] = df["sales"].interpolate("linear").round().astype(int)
    return df


def default_pdq() -> list[PDQ]:
    """Default (p, d, q) grid: p,q in {0,1,2}, d in {0,1}."""
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
                logger.debug(
                    "SARIMAX fit failed for order=%s seasonal=%s: %s", (p, d, q), order_s, exc
                )
                continue
            if res.aic < best["aic"]:
                best = {"aic": float(res.aic), "order": (p, d, q), "seasonal_order": order_s}
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to daily_sales.csv")
    ap.add_argument("--horizon", type=int, default=90, help="forecast horizon in days")
    ap.add_argument("--val_days", type=int, default=60, help="validation window size")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument(
        "--seasonal_period",
        type=int,
        default=7,
        help="seasonal period for the SARIMA search (0 disables seasonality)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ensure_outdir(args.outdir)
    df = load_series(args.input)
    y = df["sales"]
    train = y.iloc[: -args.val_days]
    val = y.iloc[-args.val_days :]

    seasonal_period = args.seasonal_period or None
    best = grid_search_aic(train, pdq_list=default_pdq(), seasonal_period=seasonal_period)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = SARIMAX(
            y,
            order=best["order"],
            seasonal_order=best["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=500)

    res_val = res.get_prediction(start=val.index[0], end=val.index[-1], dynamic=False)
    val_pred = res_val.predicted_mean.reindex(val.index)
    val_rmse = rmse(val.to_numpy(), val_pred.to_numpy())
    val_mape = mape(val.to_numpy(), val_pred.to_numpy())

    fut = res.get_forecast(steps=args.horizon)
    f_mean = fut.predicted_mean
    f_ci = fut.conf_int(alpha=0.05)
    f = pd.DataFrame(
        {
            "date": f_mean.index,
            "forecast": f_mean.values,
            "lower": f_ci.iloc[:, 0].values,
            "upper": f_ci.iloc[:, 1].values,
        }
    )

    f.to_csv(os.path.join(args.outdir, "forecast.csv"), index=False)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fp:
        json.dump(
            {
                "rmse": val_rmse,
                "mape": val_mape,
                "arima_order": best["order"],
                "seasonal_order": best["seasonal_order"],
                "aic": best["aic"],
            },
            fp,
            indent=2,
        )

    # Plot history + forecast
    fig, ax = plt.subplots(figsize=(12, 4))
    y.plot(ax=ax, label="history")
    f_mean.plot(ax=ax, label="forecast")

    # Convert datetime to numeric for fill_between
    x = mdates.date2num(pd.to_datetime(f["date"]).dt.to_pydatetime())
    lower = f["lower"].astype(float).to_numpy()
    upper = f["upper"].astype(float).to_numpy()
    ax.fill_between(x, lower, upper, alpha=0.2, label="95% CI")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    ax.set_title("Daily Sales: History & Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_history_forecast.png"), dpi=160)
    plt.close(fig)

    # Residual diagnostics
    resid = res.resid
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(resid.index, resid.values)
    ax.set_title("Residuals (model fit on full data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_residuals.png"), dpi=160)
    plt.close(fig)

    logger.info("[OK] Forecasting complete.")
    logger.info("Best order: %s", best)
    logger.info("Validation RMSE=%.3f, MAPE=%.2f%%", val_rmse, val_mape)
    logger.info("Outputs saved to: %s", args.outdir)


if __name__ == "__main__":
    main()
