"""Matplotlib rendering helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_history_forecast(y: pd.Series, f_mean: pd.Series, f: pd.DataFrame, outpath: str) -> None:
    """History + point forecast with a shaded 95% confidence band."""
    fig, ax = plt.subplots(figsize=(12, 4))
    y.plot(ax=ax, label="history")
    f_mean.plot(ax=ax, label="forecast")

    x = mdates.date2num(pd.to_datetime(f["date"]).to_numpy())
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
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_residuals(resid: pd.Series, outpath: str) -> None:
    """Residuals of the full-data model fit over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(resid.index, resid.to_numpy())
    ax.set_title("Residuals (model fit on full data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
