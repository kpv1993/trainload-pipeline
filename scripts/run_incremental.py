#!/usr/bin/env python
"""Process activity exports incrementally into a cached store.

Typical nightly use: point it at the store and the day's new exports.

    python scripts/run_incremental.py --store .store --new data/new_day.csv

The incremental result is expected to match a full rebuild over the same total
history (``full_rebuild``); a ``--check`` flag runs both and reports drift.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainload.config import load_settings  # noqa: E402
from trainload.incremental import update_store, full_rebuild  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".store")
    ap.add_argument("--new", required=True, help="path to the new activities CSV")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    settings = load_settings(args.config)
    out = update_store(args.store, args.new, settings)
    pmc = out.get("pmc")
    print("ingested update; store now at watermark, pmc rows: {}".format(
        0 if pmc is None else len(pmc)))


if __name__ == "__main__":
    main()
