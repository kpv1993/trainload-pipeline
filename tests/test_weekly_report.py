"""Tests for the weekly cohort report."""

import numpy as np
import pandas as pd

from trainload.config import load_settings
from trainload.metrics import weekly_report


def _frame():
    start = pd.Timestamp("2024-01-01", tz="UTC")  # a Monday
    rows = []
    for d in range(28):
        rows.append({
            "athlete_id": "ath_00", "sport": "run",
            "start_time": start + pd.Timedelta(days=d, hours=8),
            "duration_min": 50.0, "load": 60.0, "hr_mean": 140.0,
            "lthr": 165.0, "ftp": 240.0,
        })
    df = pd.DataFrame(rows).set_index("start_time", drop=False)
    df.index.name = "ts"
    return df


def test_weekly_summary_has_rows():
    s = load_settings()
    out = weekly_report.weekly_cohort_summary(_frame(), s)
    assert len(out) >= 4
    assert set(["week_start", "athlete_id", "load"]).issubset(out.columns)


def test_weekly_summary_conserves_load():
    s = load_settings()
    out = weekly_report.weekly_cohort_summary(_frame(), s)
    assert np.isclose(out["load"].sum(), 28 * 60.0)
