"""Synthetic daily-sales generator (trend + weekly + annual seasonality + noise)."""

import argparse

import numpy as np
import pandas as pd


def make_series(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic daily ``date,sales`` series for ``[start, end]``."""
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/daily_sales.csv")
    args = ap.parse_args(argv)

    df = make_series(args.start, args.end, args.seed)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df):,} rows")


if __name__ == "__main__":
    main()
