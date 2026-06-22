"""Configuration loading and the :class:`Settings` object.

Settings are read from a YAML file (see ``config/default.yaml``) and merged
over a set of built-in defaults.  Every tunable the pipeline uses lives here
so that the analytics modules never hard-code a magic number.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - yaml is optional for the defaults path
    yaml = None


# ---------------------------------------------------------------------------
# Built-in defaults.  A user-supplied YAML file is merged *over* this mapping.
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "timezone": "UTC",
    "week_anchor": "MON",            # ISO weeks start Monday
    "sports": ["run", "ride", "swim", "other"],
    "pmc": {
        "ctl_days": 42,             # chronic training load time constant
        "atl_days": 7,              # acute training load time constant
        "seed_days": 0,             # ramp-in: days of zero-load priming
    },
    "acwr": {
        "acute_days": 7,
        "chronic_days": 28,
        "flag_threshold": 1.5,
        "min_chronic_load": 10.0,   # suppress ratio until baseline exists
    },
    "zones": {
        # percentage of LTHR; upper-exclusive bin edges
        "edges_pct_lthr": [0.0, 0.68, 0.83, 0.94, 1.05, 2.00],
        "labels": ["Z1", "Z2", "Z3", "Z4", "Z5"],
    },
    "athlete_defaults": {
        "lthr": 165,
        "ftp": 230,
        "css_pace_s_per_100m": 95,
    },
    "dedup": {
        "overlap_tolerance_s": 40,
    },
}


@dataclass
class PMCSettings:
    ctl_days: int = 42
    atl_days: int = 7
    seed_days: int = 0


@dataclass
class ACWRSettings:
    acute_days: int = 7
    chronic_days: int = 28
    flag_threshold: float = 1.5
    min_chronic_load: float = 10.0


@dataclass
class ZoneSettings:
    edges_pct_lthr: List[float] = field(
        default_factory=lambda: [0.0, 0.68, 0.83, 0.94, 1.05, 2.00]
    )
    labels: List[str] = field(
        default_factory=lambda: ["Z1", "Z2", "Z3", "Z4", "Z5"]
    )


@dataclass
class Settings:
    timezone: str = "UTC"
    week_anchor: str = "MON"
    sports: List[str] = field(default_factory=lambda: ["run", "ride", "swim", "other"])
    pmc: PMCSettings = field(default_factory=PMCSettings)
    acwr: ACWRSettings = field(default_factory=ACWRSettings)
    zones: ZoneSettings = field(default_factory=ZoneSettings)
    athlete_defaults: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(_DEFAULTS["athlete_defaults"])
    )
    dedup: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(_DEFAULTS["dedup"])
    )

    @property
    def week_rule(self) -> str:
        """Translate the configured week anchor into a pandas offset alias.

        pandas weekly resampling uses anchored offsets like ``W-MON`` meaning
        "weeks ending on Monday".  We expose a friendly ``week_anchor`` and
        convert it here.
        """
        anchor = self.week_anchor.upper()
        return "W-{}".format(anchor)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_settings(path: Optional[str] = None) -> Settings:
    """Load :class:`Settings`, merging an optional YAML file over the defaults."""
    merged = copy.deepcopy(_DEFAULTS)
    if path is not None:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a config file")
        with open(path, "r") as fh:
            user = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, user)

    return Settings(
        timezone=merged["timezone"],
        week_anchor=merged["week_anchor"],
        sports=list(merged["sports"]),
        pmc=PMCSettings(**merged["pmc"]),
        acwr=ACWRSettings(**merged["acwr"]),
        zones=ZoneSettings(**merged["zones"]),
        athlete_defaults=dict(merged["athlete_defaults"]),
        dedup=dict(merged["dedup"]),
    )
