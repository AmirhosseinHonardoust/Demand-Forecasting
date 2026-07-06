"""Synthetic data generation and CSV loading."""

import numpy as np
import pandas as pd


def make_series(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic daily ``date,sales`` series for ``[start, end]``.

    Signal = base + linear trend + weekly + annual seasonality + Gaussian noise,
    clipped at zero and rounded to integer units.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)

    t = np.arange(n)
    trend = 0.02 * t
    weekly = 3.0 * np.sin(2 * np.pi * (t % 7) / 7)
    annual = 5.0 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 2.0, size=n)

    base = 50
    sales = base + trend + weekly + annual + noise
    sales = np.clip(np.round(sales), 0, None).astype(int)

    return pd.DataFrame({"date": dates, "sales": sales})


def load_series(path: str) -> pd.DataFrame:
    """Load a daily ``date,sales`` CSV into a gap-filled, daily-indexed frame."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date").asfreq("D")
    df["sales"] = df["sales"].interpolate("linear").round().astype(int)
    return df
