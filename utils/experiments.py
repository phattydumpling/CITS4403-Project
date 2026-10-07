"""The parameter sweep: run every combination of settings and save the results.

Every combination of the values in SWEEP is run once for each seed in SEEDS,
using run_and_measure() (500 warm-up, 2000 measured and 300 cool-down steps).
The fixed-time and adaptive runs of a setting use the same seeds, so they see
exactly the same arriving cars and can be compared pair by pair.

Run the whole experiment from the project folder with

    python -m utils.experiments

which writes data/sweep_results.csv (one row per run) and
data/example_runs.npz (queue lengths and signal phases at every step for a
few example runs, used for the time-series figures).
"""
import csv
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.intersection import Intersection
from utils.metrics import run_and_measure

RESULTS_DIR = Path(__file__).resolve().parents[1] / "notebooks/Results"

SEEDS = tuple(range(10))

# The factors that are varied, and their values.
sweep1 = {
    "controller": ("fixed", "adaptive"),
}

sweep2 = {
    "controller": ("fixed", "adaptive"),
    "demand": (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
}
sweep3 = {
    "controller": ("fixed", "adaptive"),
    "ns_share": (0.5, 0.6, 0.7, 0.8, 0.9),
}
sweep4 = {
    "controller": ("fixed", "adaptive"),
    "reaction_delay": (0.0, 0.1, 0.2, 0.35, 0.5),
}
sweep5 = {
    "controller": ("fixed", "adaptive"),
    "turning": (0.1, 0.3, 0.5, 0.7, 0.9),
}

SWEEPS=[sweep1, sweep2, sweep3, sweep4, sweep5]

def settings_grid(sweep=SWEEPS, seeds=SEEDS):
    """Every combination of the sweep values and seeds, as a list of dictionaries."""
    names = list(sweep) + ["seed"]
    return [dict(zip(names, values))
            for values in itertools.product(*sweep.values(), seeds)]


def run_one(settings):
    """Run one simulation with `settings` and return its settings plus results."""
    results = run_and_measure(Intersection(**settings))
    return {**settings, **results}


def run_sweep(path=RESULTS_DIR / "sweep_results.csv", sweep=SWEEPS, seeds=SEEDS, processes=None):
    """Run the whole sweep in parallel (one process per CPU core by default),
    write one row per run to `path` and return the rows."""
    grid = settings_grid(sweep, seeds)
    with Pool(processes) as pool:
        rows = pool.map(run_one, grid, chunksize=4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writerows(rows)
    return len(rows)

if __name__ == "__main__":
    start = time.perf_counter()
    rows = 0
    for sweep in SWEEPS:
        rows += run_sweep(sweep=sweep)
    print(f"{rows} runs saved to {RESULTS_DIR / 'sweep_results.csv'} "
          f"in {time.perf_counter() - start:.0f} s")
