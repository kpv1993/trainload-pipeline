"""Per-athlete readiness scoring and a weekly cohort report.

Readiness blends form (TSB) with how fresh the athlete is relative to their own
recent training. It also produces a weekly cohort roll-up the coach reads on
Mondays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def _per_athlete_readiness(g: pd.DataFrame) -> pd.DataFrame:
    """Readiness rows for a single athlete's PMC frame.

    Readiness is the athlete's TSB normalised against their own trailing
    21-day TSB spread, so a value near zero means "typical for them" and a
    high value means unusually fresh.
    """
    g = g.sort_values("date").copy()
    roll = g["tsb"].rolling(21, min_periods=5)
    g["tsb_baseline"] = roll.mean()
    g["tsb_spread"] = roll.std()
    g["readiness"] = (g["tsb"] - g["tsb_baseline"]) / g["tsb_spread"]
    return g


def athlete_readiness(pmc: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compute readiness for every athlete from the PMC frame.

    Returns one row per (athlete, date) with the readiness score.
    """
    if pmc.empty:
        return pd.DataFrame()

    out = pmc.sort_values(["athlete_id", "date"]).copy()
    g = out.groupby("athlete_id")["tsb"]
    out["tsb_baseline"] = g.transform(lambda s: s.rolling(21, min_periods=5).mean())
    spread = g.transform(lambda s: s.rolling(21, min_periods=5).std())
    out["readiness"] = (out["tsb"] - out["tsb_baseline"]) / spread
    return out[["athlete_id", "date", "tsb", "tsb_baseline", "readiness"]].reset_index(drop=True)


def cohort_readiness_matrix(readiness: pd.DataFrame,
                            settings: Settings) -> pd.DataFrame:
    """Build the coach's grid: athletes x recent dates of readiness.

    For each athlete we look up their readiness on each of the recent dates and
    lay it out as a matrix the coach can scan down a column.
    """
    if readiness.empty:
        return pd.DataFrame()

    dates = sorted(readiness["date"].unique())[-21:]
    athletes = sorted(readiness["athlete_id"].unique())

    # Pivot once instead of cell-by-cell lookups so this stays fast at scale.
    recent = readiness[readiness["date"].isin(dates)]
    out = recent.pivot_table(
        index="date", columns="athlete_id", values="readiness", aggfunc="first"
    )
    out = out.reindex(index=pd.to_datetime(dates), columns=athletes)
    out.index.name = "date"
    return out
