"""Tests for ACWR computation and flagging."""

import numpy as np
import pandas as pd

from trainload.config import load_settings
from trainload.metrics.acwr import compute_acwr


def _frame_with_late_spike():
    """Steady load for 8 weeks, then a big spike well after the warmup."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for d in range(70):
        load = 50.0
        if 56 <= d <= 60:  # spike in week 9, long after chronic warmup
            load = 180.0
        rows.append({
            "athlete_id": "ath_00",
            "sport": "run",
            "start_time": start + pd.Timedelta(days=d, hours=8),
            "duration_min": 50.0,
            "load": load,
            "hr_mean": 140.0,
            "lthr": 165.0,
            "ftp": 240.0,
        })
    df = pd.DataFrame(rows)
    df = df.set_index("start_time", drop=False)
    df.index.name = "ts"
    return df


def test_acwr_columns_present():
    s = load_settings()
    out = compute_acwr(_frame_with_late_spike(), s)
    for col in ("athlete_id", "date", "acute", "chronic", "acwr", "flag"):
        assert col in out.columns


def test_acwr_flag_is_boolean():
    s = load_settings()
    out = compute_acwr(_frame_with_late_spike(), s)
    assert out["flag"].dtype == bool


def test_late_spike_is_flagged():
    s = load_settings()
    out = compute_acwr(_frame_with_late_spike(), s)
    # a spike well after the warmup window should raise at least one flag
    assert out["flag"].any()
