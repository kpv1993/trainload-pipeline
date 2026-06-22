#!/usr/bin/env python
"""Run the trainload pipeline from the command line.

Example::

    python scripts/run_pipeline.py \
        --activities data/activities.csv \
        --config config/default.yaml \
        --out out/
"""

from __future__ import annotations

import argparse
import os
import sys

# allow running from a checkout without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainload.pipeline import run_pipeline  # noqa: E402
from trainload.report import summarise, export  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activities", default="data/activities.csv")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    results = run_pipeline(args.activities, config_path=args.config)
    print(summarise(results))
    export(results, args.out)
    print("\nwrote result frames to {}/".format(args.out))


if __name__ == "__main__":
    main()
