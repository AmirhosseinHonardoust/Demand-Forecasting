<div align="center">
          
# Demand Forecasting (Time-Series)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA-green)
![Backtesting](https://img.shields.io/badge/Backtesting-Rolling%20Origin-orange)
![Metrics](https://img.shields.io/badge/Metrics-RMSE%20%7C%20MAPE%20%7C%20sMAPE-teal)
![Status](https://img.shields.io/badge/Status-Portfolio%20MVP-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Demand-Forecasting/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Demand-Forecasting/actions/workflows/ci.yml)

</div>

An end-to-end demand-forecasting workflow that turns a daily sales series into **calibrated ARIMA/SARIMA forecasts**, **confidence intervals**, **baseline comparisons**, and a **rolling-origin backtest** — packaged as an installable CLI with tests and CI.

> **Important:** This project is a **portfolio and research demo**, not a production forecasting or planning system.
>
> The data is synthetic, the model search is intentionally small, and the results are illustrative. They should not be used for real inventory, procurement, or financial planning without domain review, real data, and validation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Forecasting and Evaluation](#forecasting-and-evaluation)
- [Backtesting and Baselines](#backtesting-and-baselines)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Forecasting is not only about fitting a single model. A useful forecast has to be reproducible, come with a sense of uncertainty, and be measurably better than a naive rule of thumb.

This project demonstrates an end-to-end time-series workflow on a synthetic daily-sales series:

- generate reproducible data with trend and seasonality
- clean and gap-fill the series onto a daily index
- select an ARIMA/SARIMA model by AIC over a small grid
- evaluate on a hold-out validation window
- produce a horizon forecast with confidence intervals
- backtest with a rolling origin against a seasonal-naive baseline
- render diagnostic figures

The goal is to show how a raw series becomes a **decision-support forecast** with uncertainty and a defensible baseline, not just a single error number.

---

## What This Project Does

This project can:

- Generate a deterministic synthetic daily-sales dataset
- Load and gap-fill a `date,sales` CSV onto a daily frequency
- Search ARIMA/SARIMA orders by AIC (weekly seasonality)
- Evaluate RMSE, MAE, MAPE, and sMAPE on a validation window
- Produce a multi-day forecast with 95% confidence intervals
- Compare a SARIMA model against a seasonal-naive baseline
- Run a rolling-origin (expanding-window) backtest across folds
- Render history/forecast and residual diagnostic figures
- Expose everything through a `demand-forecast` command-line interface
- Run automated tests and a GitHub Actions quality gate

---

## What This Project Does Not Do

This project does **not**:

- Use a real, production sales dataset
- Provide business, inventory, or financial planning advice
- Guarantee forecast accuracy on real-world demand
- Include exogenous regressors, promotions, or holiday calendars
- Handle multiple series, hierarchies, or cross-sectional demand
- Provide live retraining, monitoring, or drift detection
- Serve forecasts over an API or scheduler

A production forecasting system would need real data, feature engineering, out-of-sample monitoring, and stakeholder validation.

---

## Key Features

- **Reproducible data generation** with a fixed seed (trend + weekly + annual seasonality + noise)
- **Daily gap-filling** so irregular inputs become a clean daily series
- **AIC-based SARIMA search** over a small, transparent grid
- **Confidence intervals** on every forecast point
- **Seasonal-naive baseline** so model value is measured, not assumed
- **Rolling-origin backtest** with RMSE, MAE, MAPE, and sMAPE per fold
- **Installable package** with a `demand-forecast` CLI (`generate` / `forecast` / `backtest`)
- **Diagnostic figures** for history vs. forecast and residuals
- **Unit tests and GitHub Actions CI** across Python 3.10–3.12
- **Typed, linted, formatted** codebase (ruff, black, mypy)

---

## System Workflow

```text
Synthetic or real daily sales CSV
        ↓
Load, sort, gap-fill onto daily index
        ↓
Train / validation split
        ↓
AIC grid search over ARIMA/SARIMA orders
        ↓
Refit best model + validation metrics
        ↓
Horizon forecast with confidence intervals
        ↓
Rolling-origin backtest vs seasonal-naive baseline
        ↓
Diagnostic figures and JSON/CSV artifacts
```

---

## Project Structure

```text
Demand-Forecasting/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── daily_sales.csv
│
├── src/
│   └── demand_forecasting/
│       ├── __init__.py
│       ├── metrics.py
│       ├── data.py
│       ├── models.py
│       ├── backtest.py
│       ├── plots.py
│       └── cli.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_backtest.py
│   └── test_cli_smoke.py
│
├── outputs/                     # generated at runtime (git-ignored)
│
├── .gitignore
├── CHANGELOG.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Demand-Forecasting.git
cd Demand-Forecasting
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

Optionally install the project as a package to get the `demand-forecast`
command:

```bash
pip install -e .
```

---

## Quick Start

No dataset on hand? Generate a synthetic one and run the whole pipeline with zero
external data:

```bash
demand-forecast generate --start 2023-01-01 --end 2024-12-31 --seed 42 --out data/daily_sales.csv
demand-forecast forecast --input data/daily_sales.csv --horizon 90 --val_days 60 --outdir outputs
```

Compare the model against a baseline with a rolling-origin backtest:

```bash
demand-forecast backtest --input data/daily_sales.csv --horizon 30 --folds 3 --outdir outputs
```

Without an editable install, the same commands are available as a module:

```bash
python -m demand_forecasting.cli forecast --input data/daily_sales.csv --horizon 90
```

---

## Forecasting and Evaluation

The forecast command loads the series, searches ARIMA/SARIMA orders by AIC on the
training window, refits the best model on the full series, evaluates it on the
hold-out window, and writes the forecast, metrics, and figures.

```bash
demand-forecast forecast \
  --input data/daily_sales.csv \
  --horizon 90 \
  --val_days 60 \
  --outdir outputs \
  --seasonal_period 7
```

Generated outputs include:

```text
outputs/metrics.json
outputs/forecast.csv
outputs/fig_history_forecast.png
outputs/fig_residuals.png
```

`metrics.json` records the validation RMSE and MAPE alongside the selected ARIMA
order, seasonal order, and AIC, so any run can be traced back to the model that
produced it. `forecast.csv` contains the point forecast plus the lower and upper
bounds of the 95% confidence interval for each future day.

---

## Backtesting and Baselines

A single hold-out window can be lucky. The backtest command re-evaluates the model
over several expanding-window folds and compares it against a seasonal-naive
baseline, so improvement is measured rather than assumed.

```bash
demand-forecast backtest --input data/daily_sales.csv --horizon 30 --folds 3 --outdir outputs
```

Example backtest from the included sample data:

<div align="center">

| Model | RMSE | MAE | MAPE | sMAPE |
|---|---|---|---|---|
| `seasonal_naive` | 3.54 | 2.80 | 4.56% | 4.71% |
| `sarima_111` | 2.87 | 2.35 | 3.85% | 3.95% |

</div>

> The SARIMA model beats the seasonal-naive baseline on every metric here, which is
> the point of keeping a baseline: it turns "the model works" into a number.

---

## Evaluation Metrics

The evaluation layer reports both scale-dependent and percentage errors so the
forecast can be judged from more than one angle.

<div align="center">

| Metric | Why it matters |
|---|---|
| RMSE | Penalizes large errors; scale-dependent overall accuracy |
| MAE | Average absolute error; robust and easy to read |
| MAPE | Percentage error; comparable across series scales |
| sMAPE | Symmetric percentage error; less biased near zero |
| AIC | Model-selection criterion balancing fit and complexity |

</div>

Example validation results from the included run:

<div align="center">

| Metric | Example value |
|---|---|
| RMSE | 2.11 |
| MAPE | 2.77% |
| ARIMA order | (2, 1, 2) |
| Seasonal order | (0, 1, 1, 7) |
| AIC | 2836.7 |

</div>

> These values come from a synthetic dataset and should not be read as real-world
> forecasting performance.

---

## Visual Reports

### Forecast and residual diagnostics

<div align="center">

| History and Forecast | Residual Diagnostics |
|---|---|
| ![History and forecast](https://github.com/user-attachments/assets/89f34f31-bba2-4225-a6f0-c5c55fe09f79) | ![Residual diagnostics](https://github.com/user-attachments/assets/983d4cbd-5162-45c3-9b73-b839049987e2) |
| **Analysis:** The history-and-forecast chart overlays the observed series, the point forecast, and a shaded 95% confidence band. This matters because planning depends on a range of outcomes, not a single line. | **Analysis:** The residual chart shows the model's fit errors over time. Structure or drift in residuals is a signal that the model is missing seasonality or a changing trend. |

</div>

---

## Testing and CI

Run the test suite locally:

```bash
pytest
```

Set up the development tools (linting, formatting, type checking):

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Run the quality gate locally (matches CI):

```bash
ruff check .
black --check .
mypy src
pytest
```

The GitHub Actions workflow runs on Python 3.10, 3.11, and 3.12 and checks:

- linting (ruff, rules E/F/I/B/SIM/UP)
- formatting (black)
- type checking (mypy with pandas-stubs)
- dependency installation and editable install
- unit and smoke tests (pytest)

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates major responsibilities across small modules:

<div align="center">

| Module | Purpose |
|---|---|
| `demand_forecasting/data.py` | Generates synthetic data and loads/gap-fills the CSV series |
| `demand_forecasting/metrics.py` | Computes RMSE, MAE, MAPE, and sMAPE |
| `demand_forecasting/models.py` | AIC grid search, SARIMA fitting, and the seasonal-naive baseline |
| `demand_forecasting/backtest.py` | Runs the rolling-origin backtest and summarizes folds |
| `demand_forecasting/plots.py` | Renders history/forecast and residual figures |
| `demand_forecasting/cli.py` | Orchestrates the generate / forecast / backtest commands |

</div>

---

## Limitations

This project has important limitations:

- The dataset is synthetic, not real demand data
- The ARIMA/SARIMA grid is intentionally small
- A single model family is searched; no ETS, Prophet, or ML models
- No exogenous variables, promotions, or holiday effects are modeled
- Only one series is handled at a time
- Confidence intervals assume the model's error structure is correct
- No live monitoring, drift detection, or retraining is included

The project is strongest as a portfolio demonstration of a reproducible,
baseline-checked forecasting workflow.

---

## Future Improvements

Potential next improvements:

- Add real open datasets alongside the synthetic generator
- Add ETS/Holt-Winters and Prophet models with a comparison table
- Add exogenous regressors (promotions, holidays, weather)
- Add multi-series / hierarchical forecasting
- Add probabilistic-forecast calibration checks (interval coverage)
- Add a wider, configurable model search with parallel fitting
- Add a FastAPI scoring endpoint and Docker support
- Add scheduled retraining and drift monitoring examples

---

## Tech Stack

- Python
- pandas
- NumPy
- statsmodels
- scikit-learn
- matplotlib
- pytest
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational, research, and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and
limitations clear.
