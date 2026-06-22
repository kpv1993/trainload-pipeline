"""Weekly training volume, split by sport.

Produces a tidy frame indexed by (athlete, week-start) with one column per
sport plus a total, measured in both minutes and load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def weekly_volume_by_sport(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Aggregate duration and load into weekly, per-sport buckets.

    Weeks are anchored according to ``settings.week_anchor`` (Monday by
    default).  The result has a row per (athlete, week) and columns
    ``mins_<sport>``, ``load_<sport>`` plus ``mins_total`` / ``load_total``.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "start_time" not in work.columns:
        work["start_time"] = work.index

    work = work.set_index("start_time")

    frames = []
    for athlete_id, grp in work.groupby("athlete_id", sort=True):
        # Resample each athlete's activities into weekly bins.
        weekly = grp.resample(settings.week_rule)
        mins = weekly["duration_min"].sum()
        load = weekly["load"].sum()

        per_sport_mins = (
            grp.groupby([pd.Grouper(freq=settings.week_rule), "sport"])["duration_min"]
            .sum()
            .unstack(fill_value=0.0)
        )
        per_sport_load = (
            grp.groupby([pd.Grouper(freq=settings.week_rule), "sport"])["load"]
            .sum()
            .unstack(fill_value=0.0)
        )

        out = pd.DataFrame(index=mins.index)
        out["athlete_id"] = athlete_id
        for sport in settings.sports:
            out["mins_{}".format(sport)] = per_sport_mins.get(sport, 0.0)
            out["load_{}".format(sport)] = per_sport_load.get(sport, 0.0)
        out["mins_total"] = mins
        out["load_total"] = load
        frames.append(out)

    result = pd.concat(frames)
    result = result.reset_index().rename(columns={"start_time": "week_start",
                                                  "index": "week_start"})
    # the resample index column is named after the original index ("start_time")
    if "week_start" not in result.columns and "ts" in result.columns:
        result = result.rename(columns={"ts": "week_start"})
    return result
