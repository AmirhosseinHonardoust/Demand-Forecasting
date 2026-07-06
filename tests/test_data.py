"""Tests for data generation and loading."""

from demand_forecasting.data import load_series, make_series


def test_columns_and_int_dtype() -> None:
    df = make_series("2023-01-01", "2023-01-31", seed=42)
    assert list(df.columns) == ["date", "sales"]
    assert df["sales"].dtype.kind == "i"


def test_length_and_determinism() -> None:
    a = make_series("2023-01-01", "2023-01-10", seed=7)
    b = make_series("2023-01-01", "2023-01-10", seed=7)
    assert len(a) == 10
    assert a.equals(b)


def test_seed_changes_values() -> None:
    a = make_series("2023-01-01", "2023-03-01", seed=1)
    b = make_series("2023-01-01", "2023-03-01", seed=2)
    assert not a["sales"].equals(b["sales"])


def test_non_negative() -> None:
    df = make_series("2023-01-01", "2023-12-31", seed=3)
    assert (df["sales"] >= 0).all()


def test_load_series_daily_and_gapfilled(tmp_path) -> None:
    csv = tmp_path / "s.csv"
    make_series("2023-01-01", "2023-02-01", seed=5).to_csv(csv, index=False)
    df = load_series(str(csv))
    assert df.index.freqstr == "D"
    assert not df["sales"].isna().any()
