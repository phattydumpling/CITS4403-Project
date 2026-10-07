"""The parameter sweep: run every combination of settings and save the results.

Every combination of the values in SWEEP is run once for each seed in SEEDS,
using run_and_measure() (500 warm-up, 2000 measured and 300 cool-down steps).
The runs of the three controllers (untuned fixed-time, Webster fixed-time and
adaptive) in a setting use the same seeds, so they see exactly the same
arriving cars and can be compared directly.

Run the whole experiment from the project folder with

    python -m utils.experiments

which writes data/sweep_results.csv (one row per run) and
data/example_runs.npz (queue lengths and signal phases at every step for a
few example runs, used for the time-series figures).

The pocket-length sensitivity test (straight pocket and turning bays of
different lengths; the main experiment keeps every pocket at 10 cells) runs with

    python -m utils.experiments --pockets

and writes data/sensitivity_pockets.csv.

The settings use the Intersection argument names in the code (demand,
ns_share, right_share); the saved files use the clearer names in
COLUMN_NAMES (arrival_rate, north_south_share, right_turn_share).
"""
import csv
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.intersection import Intersection
from utils.metrics import run_and_measure

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# The factors that are varied, and their values.
SWEEP = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": (0.05, 0.1, 0.15, 0.2),
    "ns_share": (0.5, 0.7, 0.9),
    "right_share": (0.1, 0.3, 0.5),
    "reaction_delay": (0.0, 0.1, 0.2),
}
SEEDS = tuple(range(10))

# Sensitivity test: how the length of the straight pocket and of the turning
# bays changes spillback and waiting (balanced directions, no reaction delay).
POCKET_SENSITIVITY = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": (0.1, 0.15),
    "right_share": (0.3, 0.5),
    "pocket_length": (10, 20),
    "turn_pocket_length": (5, 10, 20),
}

# Names used for these settings in the saved files (code name -> file name).
COLUMN_NAMES = {
    "demand": "arrival_rate",          # chance a car arrives per approach per step
    "ns_share": "north_south_share",   # fraction of arrivals from north and south
    "right_share": "right_turn_share",  # fraction of cars turning right
    "pocket_length": "straight_pocket_length",      # cells in the straight pocket
    "turn_pocket_length": "turn_pocket_length",     # cells in each turning bay
}


def file_names(row):
    """Copy of `row` with settings renamed for the saved files (see COLUMN_NAMES)."""
    return {COLUMN_NAMES.get(name, name): value for name, value in row.items()}


# Settings of the example runs saved with full time series.
EXAMPLES = [
    {"controller": c, "demand": 0.15, "ns_share": s, "right_share": 0.3,
     "reaction_delay": 0.0, "seed": 0}
    for s in (0.5, 0.9) for c in ("fixed", "webster", "adaptive")
]


def settings_grid(sweep=SWEEP, seeds=SEEDS):
    """Every combination of the sweep values and seeds, as a list of dictionaries."""
    names = list(sweep) + ["seed"]
    return [dict(zip(names, values))
            for values in itertools.product(*sweep.values(), seeds)]


def run_one(settings):
    """Run one simulation with `settings` and return its settings plus results."""
    results = run_and_measure(Intersection(**settings))
    return {**settings, **results}


def run_sweep(path=DATA_DIR / "sweep_results.csv", sweep=SWEEP, seeds=SEEDS, processes=None):
    """Run the whole sweep in parallel (one process per CPU core by default),
    write one row per run to `path` and return the rows (with file names)."""
    grid = settings_grid(sweep, seeds)
    with Pool(processes) as pool:
        rows = [file_names(row) for row in pool.map(run_one, grid, chunksize=4)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def record_examples(path=DATA_DIR / "example_runs.npz", examples=EXAMPLES):
    """Run each example setting with full recording and save its queue lengths
    (steps x approaches x moves) and green phase (-1 = all-red) at every step."""
    arrays = {}
    for i, settings in enumerate(examples):
        results = run_and_measure(Intersection(**settings), record=True)
        arrays[f"queues_{i}"] = results["queues"]
        arrays[f"phases_{i}"] = results["phases"]
    labels = [", ".join(f"{k}={v}" for k, v in file_names(s).items()) for s in examples]
    np.savez_compressed(path, labels=np.array(labels), **arrays)


if __name__ == "__main__":
    start = time.perf_counter()
    if "--pockets" in sys.argv:
        path = DATA_DIR / "sensitivity_pockets.csv"
        rows = run_sweep(path, POCKET_SENSITIVITY)
        print(f"{len(rows)} runs saved to {path} in {time.perf_counter() - start:.0f} s")
    else:
        rows = run_sweep()
        print(f"{len(rows)} runs saved to {DATA_DIR / 'sweep_results.csv'} "
              f"in {time.perf_counter() - start:.0f} s")
        record_examples()
        print(f"{len(EXAMPLES)} example runs saved to {DATA_DIR / 'example_runs.npz'}")
