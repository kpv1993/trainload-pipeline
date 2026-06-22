"""Tests for the PMC (CTL/ATL/TSB) curves."""

import numpy as np
import pandas as pd

from trainload.config import load_settings
from trainload.metrics.pmc import compute_pmc


def _frame(days=120, seed=3):
    rng = np.random.RandomState(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for d in range(days):
        # not every day has a session
        if rng.rand() < 0.7:
            rows.append({
                "athlete_id": "ath_00",
                "sport": "run",
                "start_time": start + pd.Timedelta(days=d, hours=8),
                "duration_min": 50.0,
                "load": float(rng.normal(60, 15)),
                "hr_mean": 140.0,
                "lthr": 165.0,
                "ftp": 240.0,
            })
    df = pd.DataFrame(rows)
    df = df.set_index("start_time", drop=False)
    df.index.name = "ts"
    return df


def test_pmc_has_all_columns():
    s = load_settings()
    out = compute_pmc(_frame(), s)
    for col in ("athlete_id", "date", "ctl", "atl", "tsb"):
        assert col in out.columns


def test_ctl_is_smoother_than_atl():
    s = load_settings()
    out = compute_pmc(_frame(), s)
    # chronic load should vary less day-to-day than acute load
    assert out["ctl"].diff().abs().mean() < out["atl"].diff().abs().mean()


def test_pmc_non_negative():
    s = load_settings()
    out = compute_pmc(_frame(), s)
    assert (out["ctl"].dropna() >= 0).all()
    assert (out["atl"].dropna() >= 0).all()
