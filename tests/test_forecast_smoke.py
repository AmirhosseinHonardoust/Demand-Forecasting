"""Fast smoke tests for the forecast pipeline (heavy deps skip cleanly)."""

import numpy as np
import pandas as pd
import pytest

import forecast_arima as fa


def _write_csv(tmp_path):
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    sales = (50 + np.arange(120) * 0.1 + rng.normal(0, 1, 120)).round().astype(int)
    path = tmp_path / "sales.csv"
    pd.DataFrame({"date": dates, "sales": sales}).to_csv(path, index=False)
    return path


def test_load_series_is_daily_and_int(tmp_path) -> None:
    df = fa.load_series(str(_write_csv(tmp_path)))
    assert df.index.freqstr == "D"
    assert df["sales"].dtype.kind == "i"
    assert not df["sales"].isna().any()


def test_default_pdq_grid_size() -> None:
    assert len(fa.default_pdq()) == 18


def test_grid_search_returns_valid_order(tmp_path) -> None:
    pytest.importorskip("statsmodels")
    df = fa.load_series(str(_write_csv(tmp_path)))
    best = fa.grid_search_aic(df["sales"], pdq_list=[(1, 1, 1)], seasonal_period=None)
    assert best["order"] == (1, 1, 1)
    assert np.isfinite(best["aic"])


@pytest.mark.slow
def test_main_writes_outputs(tmp_path) -> None:
    pytest.importorskip("statsmodels")
    csv = _write_csv(tmp_path)
    outdir = tmp_path / "out"
    fa.main(
        [
            "--input",
            str(csv),
            "--horizon",
            "10",
            "--val_days",
            "20",
            "--outdir",
            str(outdir),
            "--seasonal_period",
            "0",
        ]
    )
    assert (outdir / "forecast.csv").exists()
    assert (outdir / "metrics.json").exists()
    fc = pd.read_csv(outdir / "forecast.csv")
    assert list(fc.columns) == ["date", "forecast", "lower", "upper"]
    assert len(fc) == 10
