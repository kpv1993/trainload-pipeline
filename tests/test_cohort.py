"""Tests for the cohort / multi-athlete views."""

import numpy as np
import pandas as pd

from trainload.config import load_settings
from trainload.metrics import cohort


def _two_athlete_frame(days=60):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for d in range(days):
        for a, base in (("ath_00", 60.0), ("ath_01", 75.0)):
            rows.append({
                "athlete_id": a, "sport": "run",
                "start_time": start + pd.Timedelta(days=d, hours=8),
                "duration_min": 50.0, "load": base, "hr_mean": 140.0,
                "lthr": 165.0, "ftp": 240.0,
            })
    df = pd.DataFrame(rows).set_index("start_time", drop=False)
    df.index.name = "ts"
    return df


def test_cohort_zscores_has_columns():
    s = load_settings()
    z = cohort.group_load_zscores(_two_athlete_frame(), s)
    for col in ("date", "athlete_id", "load_z", "outlier"):
        assert col in z.columns


def test_rolling_trend_runs():
    s = load_settings()
    t = cohort.rolling_load_trend(_two_athlete_frame())
    assert "load_trend" in t.columns
    assert len(t) > 0


def test_rank_by_ramp_is_sorted():
    s = load_settings()
    r = cohort.rank_by_ramp(_two_athlete_frame(), s)
    ramps = r["ramp"].dropna().values
    assert all(ramps[i] >= ramps[i + 1] for i in range(len(ramps) - 1))


def test_outlier_flag_is_boolean():
    s = load_settings()
    z = cohort.group_load_zscores(_two_athlete_frame(), s)
    assert z["outlier"].dtype == bool
