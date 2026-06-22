"""The Performance Management Chart: CTL, ATL and TSB.

CTL (chronic training load, "fitness") and ATL (acute training load,
"fatigue") are exponentially weighted moving averages of daily load with
time constants of 42 and 7 days respectively.  TSB (training stress balance,
"form") is yesterday's CTL minus yesterday's ATL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def _daily_load(grp: pd.DataFrame) -> pd.Series:
    """Collapse an athlete's activities into a daily total-load series."""
    s = grp.set_index("start_time")["load"].sort_index()
    daily = s.resample("D").sum()
    return daily


def compute_pmc(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compute the PMC curves per athlete.

    Returns a tidy frame with columns ``athlete_id``, ``date``, ``ctl``,
    ``atl`` and ``tsb``.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "start_time" not in work.columns:
        work["start_time"] = work.index

    ctl_days = settings.pmc.ctl_days
    atl_days = settings.pmc.atl_days

    frames = []
    for athlete_id, grp in work.groupby("athlete_id", sort=True):
        daily = _daily_load(grp)

        # Standard Banister PMC: recursive EWMA seeded at zero so fitness and
        # fatigue build up from a cold start rather than starting at the first
        # day's load.  Prepend a zero-load day, run the recursive (adjust=False)
        # EWMA, then drop the seed row.
        seed_idx = daily.index[:1] - pd.Timedelta(days=1)
        seeded = pd.concat([pd.Series([0.0], index=seed_idx), daily])

        ctl = seeded.ewm(span=ctl_days, adjust=False).mean().iloc[1:]
        atl = seeded.ewm(span=atl_days, adjust=False).mean().iloc[1:]

        out = pd.DataFrame({
            "date": daily.index,
            "ctl": ctl.values,
            "atl": atl.values,
        })
        out["athlete_id"] = athlete_id
        # form is the prior day's fitness minus fatigue
        out["tsb"] = (out["ctl"] - out["atl"]).shift(1)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True)
    return result[["athlete_id", "date", "ctl", "atl", "tsb"]]
