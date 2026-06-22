"""Cohort analytics: compare each athlete against the group.

Once several athletes are on the system we can do things a single athlete
can't: flag who is carrying more fatigue than their training group, and rank
athletes by how hard they are ramping relative to everyone else.

This module consumes the cleaned activity frame plus the per-athlete PMC frame
and produces a cohort view: a rolling load trend per athlete, a z-score of that
trend against the group, and an ``outlier`` flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def _daily_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long frame of daily load: one row per (athlete, date)."""
    work = df.copy()
    if "start_time" not in work.columns:
        work["start_time"] = work.index

    daily = (
        work.set_index("start_time")
        .groupby([pd.Grouper(freq="D"), "athlete_id"])["load"]
        .sum()
        .reset_index()
        .rename(columns={"start_time": "date"})
    )
    return daily


def rolling_load_trend(df: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """28-day rolling mean daily load per athlete.

    Smooths the noisy day-to-day load into a trend so the cohort comparison is
    on training direction rather than a single session.
    """
    daily = _daily_long(df)
    if daily.empty:
        return daily

    daily = daily.sort_values(["athlete_id", "date"])
    # Rolling mean of daily load per athlete to get each one's load trend.
    daily["load_trend"] = (
        daily.groupby("athlete_id")["load"]
        .transform(lambda s: s.rolling(window, min_periods=7).mean())
    )
    return daily


def group_load_zscores(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per-athlete daily z-score of load trend against the cohort.

    For each calendar day we take the cross-athlete distribution of the load
    trend and express each athlete as a z-score against it.  A high positive
    z-score means the athlete is training much harder than the group.
    """
    trend = rolling_load_trend(df)
    if trend.empty:
        return pd.DataFrame()

    # cohort mean / spread across athletes, per day. Require at least two
    # athletes present for a meaningful spread; thin days get NaN spread but
    # do not blank out athletes who have data on well-populated days.
    daily_stats = trend.groupby("date")["load_trend"].agg(
        mean="mean", std="std", n="count"
    )
    daily_stats.loc[daily_stats["n"] < 2, "std"] = np.nan
    trend = trend.merge(daily_stats[["mean", "std"]], on="date", how="left")

    trend["load_z"] = (trend["load_trend"] - trend["mean"]) / trend["std"]

    threshold = 1.5
    trend["outlier"] = trend["load_z"] > threshold
    return trend[["date", "athlete_id", "load", "load_trend", "load_z", "outlier"]]


def rank_by_ramp(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Rank athletes by how steeply they ramped load over the last 4 weeks.

    For each athlete we compare their most recent fortnight of mean daily load
    to the fortnight before it; the ratio is the ramp.  Athletes are returned
    sorted from steepest ramp down.
    """
    daily = _daily_long(df)
    if daily.empty:
        return pd.DataFrame()

    athletes = sorted(daily["athlete_id"].unique())
    end = daily["date"].max()
    recent_lo = end - pd.Timedelta(days=14)
    prior_lo = end - pd.Timedelta(days=28)

    rows = []
    for a in athletes:
        # pull this athlete's two fortnight windows
        sub = daily[daily["athlete_id"] == a]
        recent = sub[(sub["date"] > recent_lo) & (sub["date"] <= end)]
        prior = sub[(sub["date"] > prior_lo) & (sub["date"] <= recent_lo)]

        # mean *daily* load over the fixed 14-day window: rest days count as
        # zero, so divide the window sum by 14 not by the active-day count.
        recent_mean = recent["load"].sum() / 14.0
        prior_mean = prior["load"].sum() / 14.0
        ramp = recent_mean / prior_mean if prior_mean else np.nan
        rows.append({
            "athlete_id": a,
            "recent_load": recent_mean,
            "prior_load": prior_mean,
            "ramp": ramp,
        })

    out = pd.DataFrame(rows).sort_values("ramp", ascending=False)
    return out.reset_index(drop=True)
