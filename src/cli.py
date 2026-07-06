"""Command-line interface: generate, forecast, and backtest."""

import argparse
import json
import logging
import os
import warnings
from typing import cast

from demand_forecasting.backtest import rolling_backtest, summarize
from demand_forecasting.data import load_series, make_series
from demand_forecasting.metrics import mape, rmse
from demand_forecasting.models import (
    PDQ,
    SeasonalOrder,
    default_pdq,
    fit_sarima,
    grid_search_aic,
    sarima_forecast,
    seasonal_naive_forecast,
)
from demand_forecasting.plots import plot_history_forecast, plot_residuals

logger = logging.getLogger(__name__)


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cmd_generate(args: argparse.Namespace) -> None:
    df = make_series(args.start, args.end, args.seed)
    df.to_csv(args.out, index=False)
    logger.info("[OK] wrote %s with %d rows", args.out, len(df))


def cmd_forecast(args: argparse.Namespace) -> None:
    import pandas as pd

    _ensure_outdir(args.outdir)
    df = load_series(args.input)
    y = df["sales"]
    train = y.iloc[: -args.val_days]
    val = y.iloc[-args.val_days :]

    seasonal_period = args.seasonal_period or None
    best = grid_search_aic(train, pdq_list=default_pdq(), seasonal_period=seasonal_period)
    best_order = cast(PDQ, best["order"])
    best_seasonal = cast(SeasonalOrder, best["seasonal_order"])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        res = fit_sarima(y, best_order, best_seasonal)

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

    plot_history_forecast(y, f_mean, f, os.path.join(args.outdir, "fig_history_forecast.png"))
    plot_residuals(res.resid, os.path.join(args.outdir, "fig_residuals.png"))

    logger.info("[OK] Forecasting complete.")
    logger.info("Best order: %s", best)
    logger.info("Validation RMSE=%.3f, MAPE=%.2f%%", val_rmse, val_mape)
    logger.info("Outputs saved to: %s", args.outdir)


def cmd_backtest(args: argparse.Namespace) -> None:
    _ensure_outdir(args.outdir)
    df = load_series(args.input)
    y = df["sales"]
    period = args.seasonal_period or 7

    def naive(train, horizon):
        return seasonal_naive_forecast(train, horizon, season_length=period)

    def sarima(train, horizon):
        return sarima_forecast(train, horizon, order=(1, 1, 1), seasonal_order=(0, 1, 1, period))

    results = {}
    for name, fn in [("seasonal_naive", naive), ("sarima_111", sarima)]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bt = rolling_backtest(y, fn, horizon=args.horizon, n_folds=args.folds)
        results[name] = summarize(bt)
        logger.info("[%s] %s", name, results[name])

    with open(os.path.join(args.outdir, "backtest.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("[OK] Backtest complete. Saved to %s", args.outdir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="demand-forecast", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic daily sales")
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/daily_sales.csv")
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("forecast", help="fit SARIMA and forecast")
    f.add_argument("--input", required=True, help="path to daily_sales.csv")
    f.add_argument("--horizon", type=int, default=90, help="forecast horizon in days")
    f.add_argument("--val_days", type=int, default=60, help="validation window size")
    f.add_argument("--outdir", default="outputs")
    f.add_argument("--seasonal_period", type=int, default=7, help="0 disables seasonality")
    f.set_defaults(func=cmd_forecast)

    b = sub.add_parser("backtest", help="rolling-origin backtest vs baseline")
    b.add_argument("--input", required=True, help="path to daily_sales.csv")
    b.add_argument("--horizon", type=int, default=30)
    b.add_argument("--folds", type=int, default=3)
    b.add_argument("--outdir", default="outputs")
    b.add_argument("--seasonal_period", type=int, default=7)
    b.set_defaults(func=cmd_backtest)
    return ap


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
