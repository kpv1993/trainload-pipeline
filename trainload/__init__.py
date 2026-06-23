"""trainload: a training-load analytics pipeline for endurance athletes.

Ingests activity exports (TCX / FIT / CSV), cleans and normalises them,
then computes weekly volume by sport, time-in-heart-rate-zone, the standard
PMC fitness/fatigue/form curves (CTL / ATL / TSB), and an acute:chronic
workload ratio (ACWR) overreach flag.

The public entry point is :func:`trainload.pipeline.run_pipeline`.
"""

__version__ = "1.2.0"

from trainload.pipeline import run_pipeline  # noqa: E402,F401
