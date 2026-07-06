"""Unit tests for the synthetic data generator."""

from generate_timeseries import make_series


def test_columns_and_dtype() -> None:
    df = make_series("2023-01-01", "2023-01-31", seed=42)
    assert list(df.columns) == ["date", "sales"]
    assert df["sales"].dtype.kind == "i"


def test_length_matches_date_range() -> None:
    df = make_series("2023-01-01", "2023-01-10", seed=1)
    assert len(df) == 10


def test_deterministic_for_same_seed() -> None:
    a = make_series("2023-01-01", "2023-03-01", seed=7)
    b = make_series("2023-01-01", "2023-03-01", seed=7)
    assert a.equals(b)


def test_different_seed_changes_values() -> None:
    a = make_series("2023-01-01", "2023-03-01", seed=1)
    b = make_series("2023-01-01", "2023-03-01", seed=2)
    assert not a["sales"].equals(b["sales"])


def test_sales_non_negative() -> None:
    df = make_series("2023-01-01", "2023-12-31", seed=3)
    assert (df["sales"] >= 0).all()
