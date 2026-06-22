"""Load raw activity exports into a single normalised frame.

The on-disk CSV mixes two timezone conventions (see ``data/generate_sample``):

* ``strava`` rows carry tz-aware UTC timestamps,
* ``garmin`` rows carry tz-naive *local* (US/Eastern) timestamps.

:func:`load_activities` is responsible for collapsing both onto a single,
tz-aware UTC timeline so that everything downstream can assume UTC.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from trainload.config import Settings


REQUIRED_COLUMNS = [
    "athlete_id",
    "source",
    "start_time",
    "sport",
    "duration_min",
    "load",
]

# garmin watch files are recorded in the athlete's home timezone.  For this
# dataset that is US/Eastern.  (A real loader would read this per-athlete; the
# sample set uses a single home tz.)
_LOCAL_TZ = "US/Eastern"


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("duration_min", "load", "hr_mean", "ftp", "lthr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalise_timestamps(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Bring both sources onto one UTC timeline.

    strava timestamps already encode their offset, so ``to_datetime`` parses
    them as UTC directly.  garmin timestamps are bare local-time strings, so we
    localise them to the home timezone first and then convert to UTC.
    """
    out = df.copy()

    is_garmin = (out["source"] == "garmin").to_numpy()
    raw = out["start_time"].astype(str)

    # Parse everything to a single UTC datetime column.
    normalised = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

    # strava: already tz-aware in the string, parse straight through.
    if (~is_garmin).any():
        strava_ts = pd.to_datetime(raw[~is_garmin], utc=True)
        normalised[~is_garmin] = strava_ts

    # garmin: bare local-time strings.  Read them as naive wall-clock and put
    # them on the UTC timeline.
    if is_garmin.any():
        garmin_naive = pd.to_datetime(raw[is_garmin])
        garmin_ts = garmin_naive.dt.tz_localize(_LOCAL_TZ).dt.tz_convert("UTC")
        normalised[is_garmin] = garmin_ts

    out["start_time"] = normalised
    return out


def load_activities(path: str, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Read the activities CSV and return a normalised, UTC-indexed frame.

    The returned frame is sorted by ``start_time`` and carries a tz-aware UTC
    ``DatetimeIndex``; the original ``start_time`` column is retained too.
    """
    if settings is None:
        from trainload.config import load_settings
        settings = load_settings()

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("activities file missing columns: {}".format(missing))

    df = _coerce_numeric(df)
    df = _normalise_timestamps(df, settings)

    # sport hygiene
    df["sport"] = df["sport"].fillna("other").str.lower()
    df.loc[~df["sport"].isin(settings.sports), "sport"] = "other"

    df = df.sort_values("start_time").reset_index(drop=True)
    df = df.set_index("start_time", drop=False)
    df.index.name = "ts"
    return df
