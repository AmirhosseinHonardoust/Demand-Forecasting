# Demand Forecasting (Time-Series)
End-to-end demand forecasting with Python using synthetic time-series sales data. Includes data generation, cleaning, ARIMA/SARIMA model selection by AIC, evaluation with RMSE and MAPE, a rolling-origin backtest against a seasonal-naive baseline, and 90-day forecasts with confidence intervals. Installable package, tests, and CI for a reproducible portfolio showcase.

---

## Features
- Synthetic daily sales generator (trend + weekly + annual seasonality + noise)
- Train/validation split
- ARIMA/SARIMA model search by AIC (weekly seasonality)
- Metrics: RMSE, MAE, MAPE, sMAPE
- **Rolling-origin backtest** comparing SARIMA vs a seasonal-naive baseline
- 90-day forecast with confidence intervals
- Plots: history vs. forecast, residual diagnostics
- Deterministic seeding for reproducibility
- Installable package with a `demand-forecast` CLI
- Unit tests + CI quality gate (ruff, black, mypy, pytest) on Python 3.10–3.12

---

## Project Structure
```
demand-forecasting/
├─ README.md
├─ CHANGELOG.md
├─ LICENSE
├─ pyproject.toml               # build + project + tool config + entry point
├─ requirements.txt             # pinned runtime deps (reproducible env)
├─ requirements-dev.txt         # dev/tooling deps
├─ .github/workflows/ci.yml     # lint, type-check, test (py3.10–3.12)
├─ data/
│  └─ daily_sales.csv           # sample data (regenerate via the CLI)
├─ src/demand_forecasting/
│  ├─ __init__.py
│  ├─ metrics.py                # rmse, mae, mape, smape
│  ├─ data.py                   # make_series, load_series
│  ├─ models.py                 # grid search, SARIMA fit, seasonal-naive baseline
│  ├─ backtest.py               # rolling-origin backtest
│  ├─ plots.py                  # history/forecast + residual plots
│  └─ cli.py                    # generate / forecast / backtest
├─ tests/
└─ outputs/                     # generated at runtime (git-ignored)
```

---

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # installs the `demand-forecast` command
```

---

## Usage

### Generate synthetic data
```bash
demand-forecast generate --start 2023-01-01 --end 2024-12-31 --seed 42 --out data/daily_sales.csv
```

### Run the forecast
```bash
demand-forecast forecast --input data/daily_sales.csv --horizon 90 --val_days 60 --outdir outputs
```

**Outputs** (written to `outputs/`, which is git-ignored)
- `outputs/metrics.json` – RMSE & MAPE (validation) + selected orders
- `outputs/forecast.csv` – point forecast + confidence intervals
- `outputs/fig_history_forecast.png`
- `outputs/fig_residuals.png`

### Backtest vs. baseline
```bash
demand-forecast backtest --input data/daily_sales.csv --horizon 30 --folds 3 --outdir outputs
```
Writes `outputs/backtest.json` with mean RMSE/MAE/MAPE/sMAPE per model across
folds. On the sample data the SARIMA model beats the seasonal-naive baseline on
every metric.

> The subcommands are also runnable without installing via
> `python -m demand_forecasting.cli <command> ...`.

---

## Development
```bash
pip install -r requirements-dev.txt
pip install -e .

ruff check .          # lint (E, F, I, B, SIM, UP)
black --check .       # formatting
mypy src              # type checking
pytest                # unit + smoke tests
```

---

## Sample Results

### Forecast vs History
<img width="1920" height="640" alt="fig_history_forecast" src="https://github.com/user-attachments/assets/89f34f31-bba2-4225-a6f0-c5c55fe09f79" />

### Residual Diagnostics
<img width="1920" height="640" alt="fig_residuals" src="https://github.com/user-attachments/assets/983d4cbd-5162-45c3-9b73-b839049987e2" />

### Key Metrics
| Metric | Value |
|--------|-------|
| RMSE   | **2.11** |
| MAPE   | **2.77%** |
| ARIMA Order | (2,1,2) |
| Seasonal Order | (0,1,1,7) |
| AIC | 2836.7 |

---

## Data Schema
| column | description       |
|--------|-------------------|
| date   | daily timestamp   |
| sales  | units sold (int)  |
