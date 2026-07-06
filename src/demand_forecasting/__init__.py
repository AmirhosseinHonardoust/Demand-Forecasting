"""Demand forecasting: ARIMA/SARIMA + baselines with backtesting."""

from demand_forecasting.metrics import mae, mape, rmse, smape

__version__ = "0.2.0"
__all__ = ["rmse", "mape", "mae", "smape", "__version__"]
