# Changelog

## 0.2.0
### Added
- Installable `src/` package (`demand_forecasting`) with a `demand-forecast`
  console entry point exposing `generate`, `forecast`, and `backtest`.
- Rolling-origin (expanding-window) backtest comparing SARIMA against a
  seasonal-naive baseline.
- Additional metrics: MAE and sMAPE (alongside RMSE and MAPE).
- Unit and smoke tests; GitHub Actions CI (ruff, black, mypy, pytest) on
  Python 3.10–3.12.
- Tooling config in `pyproject.toml`; `.gitignore`; dev requirements.

### Changed
- Split the original two scripts into cohesive modules (`data`, `models`,
  `metrics`, `backtest`, `plots`, `cli`). The SARIMA forecast path is unchanged
  and produces numerically identical output.
- Grid-search fit failures are logged instead of silently swallowed; seasonal
  period is configurable via `--seasonal_period`.
- Replaced `print` with logging; added type hints and docstrings throughout.

### Removed
- Committed generated artifacts under `outputs/` (now produced at runtime and
  git-ignored).
- Flat scripts `src/forecast_arima.py`, `src/metrics.py`,
  `data/generate_timeseries.py` (superseded by the package + CLI).
