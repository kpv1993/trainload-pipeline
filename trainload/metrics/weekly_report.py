"""Weekly cohort report and plan-vs-actual reconciliation.

Two coach-facing roll-ups:

* a weekly cohort summary (per athlete, per week: total load and a simple
  acute:chronic style ramp), and
* a plan-vs-actual view that joins each athlete's actual weekly load against a
  coach-provided training plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trainload.config import Settings


def weekly_cohort_summary(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per-athlete weekly load totals for the cohort view.

    Weeks are anchored on Monday to match the rest of the pipeline.
    """
    work = df.copy()
    if "start_time" not in work.columns:
        work["start_time"] = work.index
    work = work.set_index("start_time")

    weekly = (
        work.groupby([pd.Grouper(freq=settings.week_rule), "athlete_id"])["load"]
        .sum()
        .reset_index()
        .rename(columns={"start_time": "week_label"})
    )
    # normalise to the Monday that starts each week, matching the volume module
    weekly["week_start"] = weekly["week_label"] - pd.to_timedelta(
        (weekly["week_label"].dt.weekday), unit="D"
    )
    return weekly[["week_start", "athlete_id", "load"]]


def load_training_plan(path: str) -> pd.DataFrame:
    """Read a coach training plan CSV.

    The plan has columns ``athlete_id``, ``week_start`` (a plain ``YYYY-MM-DD``
    date) and ``planned_load``.
    """
    plan = pd.read_csv(path)
    plan["week_start"] = pd.to_datetime(plan["week_start"], utc=True)
    return plan


def plan_vs_actual(df: pd.DataFrame, plan: pd.DataFrame,
                   settings: Settings) -> pd.DataFrame:
    """Join actual weekly load against the planned load.

    Returns one row per (athlete, week) with planned vs actual load and the
    percentage of plan completed.
    """
    actual = weekly_cohort_summary(df, settings).rename(columns={"load": "actual_load"})

    merged = actual.merge(
        plan, on=["athlete_id", "week_start"], how="left"
    )
    merged["pct_of_plan"] = 100.0 * merged["actual_load"] / merged["planned_load"]
    return merged
