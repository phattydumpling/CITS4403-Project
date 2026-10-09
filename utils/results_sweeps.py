"""Finer sweeps for notebooks/Results.ipynb.

Every run starts from BASELINE and changes one or more settings:

  baseline     the baseline itself, with more seeds, to compare the three controllers
  fine sweeps  one setting at a time over many values (FINE_SWEEPS)
  scenarios    several settings changed at once (SCENARIOS)
  pockets      the length of the turning pockets and of the straight pocket,
               at two right-turn probabilities (POCKET_SWEEPS)

All three controllers run with the same seeds, so they see the same arriving
cars. Cars that find the entry full are turned away (the model default). Run
everything from the project folder with

    python -m utils.results_sweeps

which writes data/fine_sweeps.csv, data/scenarios.csv and data/fine_pockets.csv.
With --queued it instead repeats the baseline and fine sweeps with blocked
cars waiting in a queue off the map (queue_blocked_arrivals=True) instead of
being turned away, and writes data/fine_sweeps_queued.csv (analysed with the
overload extension in notebooks/Results/Extensions.ipynb).
Each row is one run; the first column says which sweep or scenario it belongs to.
"""
import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

from utils.experiments import DATA_DIR, SEEDS, file_names, run_one

CONTROLLERS = ("fixed", "webster", "adaptive")

# Busy but below the capacity of every controller, balanced directions and
# equal left and right turns.
BASELINE = {"demand": 0.15, "ns_probability": 0.5, "left_probability": 0.2,
            "right_probability": 0.2, "reaction_delay": 0.0}
BASELINE_SEEDS = tuple(range(30))


def steps(start, stop, step):
    """Values from start to stop inclusive, rounded so that they print cleanly."""
    return tuple(np.round(np.arange(start, stop + step / 2, step), 3))


FINE_SWEEPS = {
    "demand": steps(0.01, 0.30, 0.01),
    "ns_probability": steps(0.5, 1.0, 0.05),
    "reaction_delay": steps(0.0, 0.5, 0.05),
    "left_probability": steps(0.0, 0.6, 0.05),
    "right_probability": steps(0.0, 0.6, 0.05),
}

SCENARIOS = {
    "baseline": {},
    "busy, one-sided": {"demand": 0.18, "ns_probability": 0.8},
    "busy, slow drivers": {"demand": 0.18, "reaction_delay": 0.2},
    "one-sided, many right turns": {"ns_probability": 0.8, "right_probability": 0.4},
    "slow drivers, many right turns": {"reaction_delay": 0.2, "right_probability": 0.4},
    "light, very one-sided": {"demand": 0.08, "ns_probability": 0.9},
    "all four harder": {"demand": 0.18, "ns_probability": 0.8, "right_probability": 0.4,
                        "reaction_delay": 0.2},
}

# Turning pockets at two right-turn probabilities (straight pocket 10 cells),
# and the straight pocket with 10-cell turning pockets.
POCKET_SWEEPS = {
    "turn_pocket_length": [{"pocket_length": 10, "turn_pocket_length": n, "right_probability": r}
                           for r in (0.2, 0.4) for n in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)],
    "pocket_length": [{"pocket_length": n, "turn_pocket_length": 10} for n in (4, 6, 8, 10, 15, 20)],
}


def jobs_for(label, changes, seeds=SEEDS):
    """One (label, settings) job per controller and seed for BASELINE plus `changes`."""
    return [(label, {"controller": c, **BASELINE, **changes, "seed": s}) for c in CONTROLLERS for s in seeds]


def run_job(job):
    """Run one job and return its row: the label, then the settings and results."""
    label, settings = job
    return {**label, **file_names(run_one(settings))}


def run_jobs(path, jobs, processes=None):
    """Run the jobs in parallel and write one row per run to the CSV file `path`."""
    with Pool(processes) as pool:
        rows = pool.map(run_job, jobs, chunksize=4)
    names = list(dict.fromkeys(name for row in rows for name in row))    # every column, in order
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def fine_sweep_jobs(extra=None):
    """The baseline and the one-setting sweeps; `extra` settings (such as
    queue_blocked_arrivals) are added to every run."""
    extra = extra or {}
    jobs = jobs_for({"sweep": "baseline"}, extra, BASELINE_SEEDS)
    for factor, values in FINE_SWEEPS.items():
        for value in values:
            jobs += jobs_for({"sweep": factor}, {factor: float(value), **extra})
    return jobs


def scenario_jobs():
    return [job for name, changes in SCENARIOS.items() for job in jobs_for({"scenario": name}, changes)]


def pocket_jobs():
    return [job for sweep, variants in POCKET_SWEEPS.items() for changes in variants
            for job in jobs_for({"sweep": sweep}, changes)]


if __name__ == "__main__":
    if "--queued" in sys.argv:
        runs = [("fine_sweeps_queued", fine_sweep_jobs({"queue_blocked_arrivals": True}))]
    else:
        runs = [("fine_sweeps", fine_sweep_jobs()), ("scenarios", scenario_jobs()),
                ("fine_pockets", pocket_jobs())]
    for name, jobs in runs:
        start = time.perf_counter()
        n = run_jobs(DATA_DIR / f"{name}.csv", jobs)
        print(f"{name}: {n} runs in {time.perf_counter() - start:.0f} s")
