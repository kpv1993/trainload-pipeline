"""End-to-end orchestration.

``run_pipeline`` ties the stages together::

    load -> dedup -> stitch -> fill attributes -> metrics

and returns a dict of result frames (volume, zones, pmc, acwr) plus the
cleaned activity frame itself.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from trainload.config import Settings, load_settings
from trainload.io import load_activities
from trainload.cleaning import deduplicate, stitch_sessions, fill_athlete_attributes
from trainload.metrics import (
    weekly_volume_by_sport,
    time_in_zones,
    compute_pmc,
    compute_acwr,
    group_load_zscores,
    rank_by_ramp,
    athlete_readiness,
)


def run_pipeline(activities_path: str,
                 config_path: Optional[str] = None,
                 settings: Optional[Settings] = None) -> Dict[str, pd.DataFrame]:
    """Run the full pipeline and return all result frames.

    Parameters
    ----------
    activities_path:
        Path to the activities CSV.
    config_path:
        Optional path to a YAML config; ignored if ``settings`` is given.
    settings:
        A pre-built :class:`Settings`; takes precedence over ``config_path``.
    """
    if settings is None:
        settings = load_settings(config_path)

    raw = load_activities(activities_path, settings)

    # cleaning
    deduped = deduplicate(raw, settings)
    stitched = stitch_sessions(deduped, settings)
    clean = fill_athlete_attributes(stitched, settings)

    # metrics
    volume = weekly_volume_by_sport(clean, settings)
    zones = time_in_zones(clean, settings)
    pmc = compute_pmc(clean, settings)
    acwr = compute_acwr(clean, settings)

    # cohort / multi-athlete views
    cohort_z = group_load_zscores(clean, settings)
    ramp = rank_by_ramp(clean, settings)
    readiness = athlete_readiness(pmc, settings)

    return {
        "activities": clean,
        "volume": volume,
        "zones": zones,
        "pmc": pmc,
        "acwr": acwr,
        "cohort_z": cohort_z,
        "ramp": ramp,
        "readiness": readiness,
    }
