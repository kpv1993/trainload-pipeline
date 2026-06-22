"""Forward-fill the sparse per-athlete attributes.

``ftp`` and ``lthr`` are only stamped on a fraction of rows, but every row
needs them for the zone and load maths.  Within an athlete these change slowly,
so the accepted approach is: carry the most recent known value forward in time,
then back-fill the very start of the history (before the first known value)
and finally fall back to the configured athlete defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


_ATTRS = ["ftp", "lthr", "css_pace_s_per_100m"]


def fill_athlete_attributes(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Fill ``ftp`` / ``lthr`` / ``css`` per athlete by carrying values forward.

    The fill is done per athlete with a forward fill (most recent known value
    wins) followed by a backward fill for the pre-history rows, then the
    athlete defaults for anyone still missing.
    """
    if df.empty:
        return df

    out = df.copy()

    for attr in _ATTRS:
        if attr not in out.columns:
            out[attr] = np.nan

    # Carry known values forward within each athlete, then backward to cover
    # the leading rows that predate the first stamped value.
    grouped = out.groupby("athlete_id", sort=False)
    for attr in _ATTRS:
        out[attr] = grouped[attr].ffill()
        out[attr] = out.groupby("athlete_id", sort=False)[attr].bfill()

    # Anyone still missing (an athlete who never stamped the attribute at all)
    # falls back to the configured defaults.
    defaults = settings.athlete_defaults
    out["ftp"] = out["ftp"].fillna(defaults.get("ftp"))
    out["lthr"] = out["lthr"].fillna(defaults.get("lthr"))
    out["css_pace_s_per_100m"] = out["css_pace_s_per_100m"].fillna(
        defaults.get("css_pace_s_per_100m")
    )

    return out
