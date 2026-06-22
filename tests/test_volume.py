"""Tests for weekly volume aggregation."""

import numpy as np
import pandas as pd

from trainload.config import load_settings
from trainload.metrics.volume import weekly_volume_by_sport


def _frame():
    ts = pd.to_datetime([
        "2024-01-01 08:00", "2024-01-03 18:00", "2024-01-08 07:00",
        "2024-01-10 19:00", "2024-01-15 06:00",
    ], utc=True)
    df = pd.DataFrame({
        "athlete_id": ["ath_00"] * 5,
        "sport": ["run", "ride", "run", "swim", "ride"],
        "duration_min": [50.0, 90.0, 40.0, 30.0, 100.0],
        "load": [55.0, 80.0, 45.0, 35.0, 95.0],
        "hr_mean": [140, 150, 138, np.nan, 152],
        "lthr": [165] * 5,
        "ftp": [240] * 5,
        "start_time": ts,
    }, index=ts)
    df.index.name = "ts"
    return df


def test_volume_total_matches_sum():
    s = load_settings()
    out = weekly_volume_by_sport(_frame(), s)
    # the grand total of weekly load must equal the raw total
    assert np.isclose(out["load_total"].sum(), 55 + 80 + 45 + 35 + 95)


def test_volume_has_a_row_per_active_week():
    s = load_settings()
    out = weekly_volume_by_sport(_frame(), s)
    assert len(out) >= 3
    assert "mins_run" in out.columns
    assert "load_total" in out.columns


def test_volume_sport_split_sums_to_total():
    s = load_settings()
    out = weekly_volume_by_sport(_frame(), s)
    sport_cols = [c for c in out.columns if c.startswith("load_") and c != "load_total"]
    per_row = out[sport_cols].sum(axis=1)
    assert np.allclose(per_row.values, out["load_total"].values)
