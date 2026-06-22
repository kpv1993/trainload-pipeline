"""Remove activities that were exported from more than one source.

When you pull the same ride from both Strava and your Garmin, you get two rows
a few seconds apart.  We collapse them to one, keeping the Strava copy when a
choice has to be made (its load numbers tend to be the cleaner ones).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


_SOURCE_PRIORITY = {"strava": 0, "garmin": 1}


def deduplicate(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Collapse near-duplicate activities across sources.

    Two rows are considered the same session when they share an athlete and a
    sport and their start times fall within ``dedup.overlap_tolerance_s`` of
    each other.  The surviving row is the one whose source has higher priority
    (Strava over Garmin).
    """
    if df.empty:
        return df

    tol_s = float(settings.dedup.get("overlap_tolerance_s", 40))

    work = df.copy()
    work["_src_rank"] = work["source"].map(_SOURCE_PRIORITY).fillna(9)

    # Bucket start times so that near-identical timestamps share a key, then
    # keep the best-ranked row per (athlete, sport, bucket).
    bucket_ns = int(tol_s * 1e9)
    work["_bucket"] = (work["start_time"].astype("int64") // bucket_ns)

    work = work.sort_values(["athlete_id", "source", "_bucket", "_src_rank"])

    # Coalesce stamped attributes across the duplicate pair so a known ftp/lthr
    # carried only on the discarded copy is not lost.
    for attr in ("ftp", "lthr", "css_pace_s_per_100m"):
        if attr in work.columns:
            filled = work.groupby(
                ["athlete_id", "sport", "_bucket"], sort=False
            )[attr].transform(lambda s: s.ffill().bfill())
            work[attr] = filled

    deduped = work.drop_duplicates(
        subset=["athlete_id", "sport", "_bucket"], keep="first"
    )

    deduped = deduped.drop(columns=["_src_rank", "_bucket"])
    # Restore chronological order; downstream fills assume time order.
    deduped = deduped.sort_values("start_time")
    return deduped
