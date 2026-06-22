"""Acute:chronic workload ratio (ACWR) and the overreach flag.

The acute load is the rolling sum of the last ``acute_days`` of daily load;
the chronic load is the rolling average daily load over ``chronic_days``,
scaled to the same window length.  When the ratio climbs past the configured
threshold the athlete is flagged as ramping load too fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def _daily_load(grp: pd.DataFrame) -> pd.Series:
    s = grp.set_index("start_time")["load"].sort_index()
    return s.resample("D").sum()


def compute_acwr(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compute ACWR per athlete and flag overreaching days.

    Returns a tidy frame with ``athlete_id``, ``date``, ``acute``, ``chronic``,
    ``acwr`` and a boolean ``flag`` column.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "start_time" not in work.columns:
        work["start_time"] = work.index

    acute_days = settings.acwr.acute_days
    chronic_days = settings.acwr.chronic_days
    threshold = settings.acwr.flag_threshold
    min_chronic = settings.acwr.min_chronic_load

    frames = []
    for athlete_id, grp in work.groupby("athlete_id", sort=True):
        daily = _daily_load(grp)

        acute = daily.rolling(acute_days, min_periods=1).sum()
        # chronic: average daily load over the long window, scaled to an
        # acute-length window so the ratio is dimensionless.
        chronic = daily.rolling(chronic_days, min_periods=acute_days).mean() * acute_days

        acwr = acute / chronic
        acwr = acwr.replace([np.inf, -np.inf], np.nan)

        flag = (acwr > threshold) & (chronic >= min_chronic)

        out = pd.DataFrame({
            "date": daily.index,
            "acute": acute.values,
            "chronic": chronic.values,
            "acwr": acwr.values,
            "flag": flag.values,
        })
        out["athlete_id"] = athlete_id
        frames.append(out)

    result = pd.concat(frames, ignore_index=True)
    return result[["athlete_id", "date", "acute", "chronic", "acwr", "flag"]]
