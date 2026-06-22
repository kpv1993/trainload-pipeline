"""Time-in-heart-rate-zone.

Each activity carries a mean heart rate; we assign it to a zone based on the
athlete's LTHR and tally weekly minutes per zone.  Zones are defined as
fractions of LTHR (see ``settings.zones``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def _zone_for_ratio(ratio: float, edges, labels):
    """Return the zone label for a single hr/lthr ratio.

    Edges are upper bounds; a ratio falls in the first bin whose upper edge it
    is below.
    """
    if ratio != ratio:  # NaN
        return np.nan
    # bins are [edges[i], edges[i+1]); find the bin index
    idx = pd.cut([ratio], bins=edges, labels=labels)
    return idx[0]


def time_in_zones(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Weekly minutes spent in each heart-rate zone, per athlete.

    Activities without heart rate (most ``other`` sessions, indoor rides) are
    excluded from the zone tally but still counted in volume elsewhere.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "start_time" not in work.columns:
        work["start_time"] = work.index

    edges = list(settings.zones.edges_pct_lthr)
    labels = list(settings.zones.labels)

    has_hr = work["hr_mean"].notna() & work["lthr"].notna()
    work = work[has_hr].copy()
    if work.empty:
        return pd.DataFrame()

    work["hr_ratio"] = work["hr_mean"] / work["lthr"]
    # Assign every activity to a zone using the configured fractional edges.
    work["zone"] = pd.cut(work["hr_ratio"], bins=edges, labels=labels, right=False, include_lowest=True)

    work = work.set_index("start_time")

    frames = []
    for athlete_id, grp in work.groupby("athlete_id", sort=True):
        wk = (
            grp.groupby([pd.Grouper(freq="W-MON"), "zone"], observed=False)["duration_min"]
            .sum()
            .unstack(fill_value=0.0)
        )
        wk["athlete_id"] = athlete_id
        frames.append(wk)

    result = pd.concat(frames).reset_index().rename(columns={"start_time": "week_start"})
    return result
