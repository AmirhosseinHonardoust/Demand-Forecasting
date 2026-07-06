"""End-to-end CLI smoke tests."""

import json

import pandas as pd
import pytest

from demand_forecasting.cli import main


def _make_csv(tmp_path):
    csv = tmp_path / "sales.csv"
    main(["generate", "--start", "2023-01-01", "--end", "2023-05-01", "--out", str(csv)])
    return csv


def test_generate_writes_csv(tmp_path) -> None:
    csv = _make_csv(tmp_path)
    df = pd.read_csv(csv)
    assert list(df.columns) == ["date", "sales"]
    assert len(df) == 121


@pytest.mark.slow
def test_forecast_writes_outputs(tmp_path) -> None:
    pytest.importorskip("statsmodels")
    csv = _make_csv(tmp_path)
    outdir = tmp_path / "out"
    main(
        [
            "forecast",
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
    fc = pd.read_csv(outdir / "forecast.csv")
    assert list(fc.columns) == ["date", "forecast", "lower", "upper"]
    assert len(fc) == 10
    assert (outdir / "fig_history_forecast.png").exists()


@pytest.mark.slow
def test_backtest_writes_json(tmp_path) -> None:
    pytest.importorskip("statsmodels")
    csv = _make_csv(tmp_path)
    outdir = tmp_path / "out"
    main(
        [
            "backtest",
            "--input",
            str(csv),
            "--horizon",
            "14",
            "--folds",
            "2",
            "--outdir",
            str(outdir),
        ]
    )
    data = json.loads((outdir / "backtest.json").read_text())
    assert "seasonal_naive" in data and "sarima_111" in data
    assert "rmse" in data["seasonal_naive"]
