"""The parameter sweeps: run every combination of settings and save the results.

Main experiment: every combination of the values in SWEEP is run once for
each seed in SEEDS, using run_and_measure() (500 warm-up, 2000 measured and
300 cool-down steps). The runs of the three controllers (untuned fixed-time,
Webster fixed-time and adaptive) in a setting use the same seeds, so they see
exactly the same arriving cars and can be compared directly. Run it from the
project folder with

    python -m utils.experiments

which writes data/sweep_results.csv (one row per run) and
data/example_runs.npz (queue lengths and signal phases at every step for a
few example runs, used for the time-series figures).

The pocket-length sensitivity test (straight pocket and turning bays of
different lengths; the main experiment keeps every pocket at 10 cells) runs with

    python -m utils.experiments --pockets

and writes data/sensitivity_pockets.csv.

The left-turn sensitivity test (the main experiment keeps 20% of cars turning
left and varies only the right-turn probability) runs with

    python -m utils.experiments --left

and writes data/sensitivity_left.csv.

The one-factor experiments used by notebooks/Results/Analysis.ipynb vary one
setting at a time from ONE_FACTOR_BASELINE, for fixed-time and adaptive
control. They run with

    python -m utils.experiments --one-factor

and write notebooks/Results/sweep_results.csv (no header; see
run_one_factor_experiments for the column layout).

Arrivals that find the entry full are turned away in these experiments (the
model default). As a robustness check, the main sweep can be repeated with such
cars queuing off the map instead (queue_blocked_arrivals=True):

    python -m utils.experiments --queued

which writes data/sweep_results_queued.csv.

The settings use the Intersection argument names in the code (demand,
ns_probability, right_probability); the main results files use the clearer names in
COLUMN_NAMES (arrival_rate, north_south_probability, right_turn_probability).
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
RESULTS_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "Results"

# The factors that are varied, and their values.
SWEEP = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": (0.05, 0.1, 0.15, 0.2),
    "ns_probability": (0.5, 0.7, 0.9),
    "right_probability": (0.1, 0.3, 0.5),
    "reaction_delay": (0.0, 0.1, 0.2),
}
SEEDS = tuple(range(10))

# Sensitivity test: how the length of the straight pocket and of the turning
# bays changes spillback and waiting (balanced directions, no reaction delay).
POCKET_SENSITIVITY = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": (0.1, 0.15),
    "right_probability": (0.3, 0.5),
    "pocket_length": (10, 20),
    "turn_pocket_length": (5, 10, 20),
}

# Sensitivity test: how the left-turn probability changes waiting. Left
# turns share the green with straight-on cars, so they mainly change how the
# load is split between the left and straight pockets.
LEFT_SENSITIVITY = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": (0.05, 0.1, 0.15, 0.2),
    "ns_probability": (0.5,),
    "left_probability": (0.1, 0.2, 0.3, 0.4),
    "right_probability": (0.3,),
    "reaction_delay": (0.0,),
}

# One-factor experiments (notebooks/Results/Analysis.ipynb). Every setting not
# varied in a sweep takes its value from this baseline.
ONE_FACTOR_BASELINE = {"demand": 0.2, "ns_probability": 0.5, "left_probability": 0.25, "right_probability": 0.25,
                       "reaction_delay": 0.0}

sweep1 = {
    "controller": ("fixed", "adaptive"),
}
sweep2 = {
    "controller": ("fixed", "adaptive"),
    "demand": (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
}
sweep3 = {
    "controller": ("fixed", "adaptive"),
    "ns_probability": (0.5, 0.6, 0.7, 0.8, 0.9),
}
sweep4 = {
    "controller": ("fixed", "adaptive"),
    "reaction_delay": (0.0, 0.1, 0.2, 0.35, 0.5),
}
sweep5 = {
    "controller": ("fixed", "adaptive"),
    "left_probability": (0.1, 0.2, 0.3, 0.4, 0.5),
}
sweep6 = {
    "controller": ("fixed", "adaptive"),
    "right_probability": (0.1, 0.2, 0.3, 0.4, 0.5),
}
ONE_FACTOR_SWEEPS = [sweep1, sweep2, sweep3, sweep4, sweep5, sweep6]

# Results written for the one-factor experiments, in the order the notebook reads them.
ONE_FACTOR_RESULTS = (
    "mean_wait", "wait_ns", "wait_ew", "wait_left", "wait_straight", "wait_right",
    "cars_measured", "cars_unfinished", "throughput", "blocked_arrivals",
    "phase_changes", "stalls", "queue_left", "queue_straight", "queue_right",
)

# Names used for these settings in the main results files (code name -> file name).
COLUMN_NAMES = {
    "demand": "arrival_rate",          # chance a car arrives per approach per step
    "ns_probability": "north_south_probability",   # chance an arriving car comes from N or S
    "left_probability": "left_turn_probability",    # chance a car turns left
    "right_probability": "right_turn_probability",  # chance a car turns right
    "pocket_length": "straight_pocket_length",      # cells in the straight pocket
    "turn_pocket_length": "turn_pocket_length",     # cells in each turning bay
}


def file_names(row):
    """Copy of `row` with settings renamed for the saved files (see COLUMN_NAMES)."""
    return {COLUMN_NAMES.get(name, name): value for name, value in row.items()}


# Settings of the example runs saved with full time series.
EXAMPLES = [
    {"controller": c, "demand": 0.15, "ns_probability": s, "right_probability": 0.3,
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


def run_one_factor(settings):
    """Run one one-factor setting (baseline plus `settings`) and return its row:
    the varied settings in sweep order (controller, the varied factor if any,
    seed) followed by the ONE_FACTOR_RESULTS values."""
    results = run_and_measure(Intersection(**{**ONE_FACTOR_BASELINE, **settings}))
    return list(settings.values()) + [results[name] for name in ONE_FACTOR_RESULTS]


def run_one_factor_experiments(path=RESULTS_DIR / "sweep_results.csv", sweeps=ONE_FACTOR_SWEEPS,
                               seeds=SEEDS, processes=None):
    """Run every one-factor sweep and write all rows, sweep after sweep, to
    `path` without a header (the file is replaced, not appended to). Each
    setting's runs are consecutive rows, one per seed. Returns the number of rows."""
    rows = []
    with Pool(processes) as pool:
        for sweep in sweeps:
            rows += pool.map(run_one_factor, settings_grid(sweep, seeds), chunksize=4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return len(rows)


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
    elif "--left" in sys.argv:
        path = DATA_DIR / "sensitivity_left.csv"
        rows = run_sweep(path, LEFT_SENSITIVITY)
        print(f"{len(rows)} runs saved to {path} in {time.perf_counter() - start:.0f} s")
    elif "--one-factor" in sys.argv:
        n = run_one_factor_experiments()
        print(f"{n} runs saved to {RESULTS_DIR / 'sweep_results.csv'} "
              f"in {time.perf_counter() - start:.0f} s")
    elif "--queued" in sys.argv:
        path = DATA_DIR / "sweep_results_queued.csv"
        rows = run_sweep(path, {**SWEEP, "queue_blocked_arrivals": (True,)})
        print(f"{len(rows)} runs saved to {path} in {time.perf_counter() - start:.0f} s")
    else:
        rows = run_sweep()
        print(f"{len(rows)} runs saved to {DATA_DIR / 'sweep_results.csv'} "
              f"in {time.perf_counter() - start:.0f} s")
        record_examples()
        print(f"{len(EXAMPLES)} example runs saved to {DATA_DIR / 'example_runs.npz'}")
