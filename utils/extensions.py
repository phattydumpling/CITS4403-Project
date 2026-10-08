"""Extension experiments, beyond the main experiment in utils/experiments.py.

  1. Tipping point: a fine sweep of the arrival rate through capacity, looking
     for the signs of a transition (a jump in the number of cars on the road,
     large fluctuations and slow settling near the critical arrival rate).
  2. Self-organised timing: the green and cycle lengths that emerge under
     adaptive control, compared with the plan Webster's method designs.
  3. Distributions: percentiles of the wait of individual cars, not only means.
     Extensions 1-3 all come from the same runs (FINE_SWEEP).
  4. Limited sensing: the adaptive controller sees only the cells of each
     pocket nearest the stop line (SENSING).
  5. Different drivers: each driver is slow with some probability, and slow
     drivers are much slower to start, with the same average reaction delay
     (DRIVERS).
  6. Time-varying demand: demand rises above capacity and falls back, and we
     follow how each controller copes and recovers (SURGE).

Run them all from the project folder with

    python -m utils.extensions

or one at a time with --fine, --sensing, --drivers or --surge. Results are
written to data/ext_*.csv and data/ext_*.npz and analysed in
notebooks/Results/Extensions.ipynb.
"""
import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

from src.intersection import Intersection
from src.signals import PHASES, webster_green_times
from utils.experiments import DATA_DIR, SEEDS, file_names, run_sweep, settings_grid
from utils.metrics import run_and_measure

WARMUP = 500   # as in run_and_measure

# Extensions 1-3: arrival rate from light traffic to well past capacity.
FINE_SWEEP = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": tuple(np.round(np.arange(0.02, 0.305, 0.01), 2)),
    "ns_probability": (0.5, 0.9),
    "right_probability": (0.3,),
    "reaction_delay": (0.0, 0.2),
}
# Arrival rates whose individual car waits are kept for the distribution plots.
WAIT_SAMPLE_DEMANDS = (0.1, 0.15, 0.2)

# Extension 4: how many cells of each pocket the adaptive controller can see
# (0 = nothing, 10 = the whole pocket as in the main experiment).
SENSING = {
    "controller": ("adaptive",),
    "demand": (0.1, 0.15, 0.2),
    "ns_probability": (0.5, 0.9),
    "right_probability": (0.3,),
    "reaction_delay": (0.0,),
    "detector_range": (0, 1, 2, 3, 5, 10),
}

# Extension 5: the same average reaction delay spread over all drivers (probability 1)
# or concentrated in half or a quarter of them.
DRIVERS = {
    "controller": ("fixed", "webster", "adaptive"),
    "demand": (0.1, 0.15, 0.2),
    "ns_probability": (0.5,),
    "right_probability": (0.3,),
    "reaction_delay": (0.1, 0.2),
    "slow_driver_probability": (1.0, 0.5, 0.25),
}

# Extension 6: arrival rate over time (steps) - steady, a linear rise to a peak
# above capacity and back down, then steady again while queues clear.
SURGE = {"base": 0.1, "peak": 0.25, "start": 1000, "ramp": 500, "steps": 3500}
SURGE_NS_PROBABILITIES = (0.5, 0.7)
SURGE_SEEDS = tuple(range(20))


# --- Measures used by extensions 1-3 -------------------------------------

def settling_time(cars, block=100, level=0.9):
    """Steps until the number of cars on the road first reaches `level` of its
    level in the measured window: the end of the first `block`-step block,
    counted from the empty start, whose mean is at least that high. `cars` runs
    from the first step of the warm-up to the end of the window."""
    target = level * cars[WARMUP:].mean()
    blocks = cars[: len(cars) // block * block].reshape(-1, block).mean(axis=1)
    return block * (int(np.argmax(blocks >= target)) + 1)


def green_periods(phases):
    """(phase, first step, length) of every green that starts and ends inside
    the window; `phases` is the green phase at each step (-1 for all-red)."""
    changes = np.flatnonzero(np.diff(phases)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(phases)]))
    return [(int(phases[s]), int(s), int(e - s)) for s, e in zip(starts[1:-1], ends[1:-1])
            if phases[s] >= 0]


def signal_timing(phases):
    """Average green of each phase, the fraction of the window each phase is
    green, and the cycle length: the average time between the starts of two
    greens of the same phase, averaged over the phases."""
    periods = green_periods(phases)
    timing, returns = {}, []
    for p in range(len(PHASES)):
        lengths = [n for q, _, n in periods if q == p]
        starts = [s for q, s, _ in periods if q == p]
        timing[f"green_{p + 1}"] = float(np.mean(lengths)) if lengths else 0.0
        timing[f"green_fraction_{p + 1}"] = float(np.mean(phases == p))
        if len(starts) > 1:
            returns.append(np.mean(np.diff(starts)))
    timing["cycle"] = float(np.mean(returns)) if returns else float("nan")
    return timing


def webster_design(settings):
    """Cycle and green fraction of each phase in Webster's plan for `settings`."""
    model = Intersection(**{**settings, "controller": "fixed"})
    greens = webster_green_times(model)
    cycle = sum(greens) + len(PHASES) * model.controller.all_red
    design = {"design_cycle": cycle}
    design.update({f"design_fraction_{p + 1}": g / cycle for p, g in enumerate(greens)})
    return design


def wait_percentiles(waits, served_ns):
    """Median, 90th and 99th percentile and longest wait, and the 90th
    percentile for north-south and east-west cars separately."""
    def pct(values, q):
        return float(np.percentile(values, q)) if len(values) else float("nan")
    return {"wait_p50": pct(waits, 50), "wait_p90": pct(waits, 90), "wait_p99": pct(waits, 99),
            "wait_max": float(waits.max()) if len(waits) else float("nan"),
            "wait_p90_ns": pct(waits[served_ns], 90), "wait_p90_ew": pct(waits[~served_ns], 90)}


def describe_run(settings):
    """Run one simulation of the fine sweep and return its settings, the usual
    results and the extra measures for extensions 1-3, plus the individual
    waits (under "waits")."""
    model = Intersection(**settings)
    warmup_cars = np.empty(WARMUP, dtype=np.int16)
    for t in range(WARMUP):                      # the warm-up of run_and_measure, recorded
        model.step()
        warmup_cars[t] = model.n_cars()
    results = run_and_measure(model, warmup=0, record=True)
    cars = np.concatenate((warmup_cars, results["cars"]))
    row = {**settings}
    row.update({name: value for name, value in results.items() if np.isscalar(value)})
    row.update({"cars_mean": float(results["cars"].mean()), "cars_std": float(results["cars"].std()),
                "settling_time": settling_time(cars)})
    row.update(signal_timing(results["phases"]))
    row.update(webster_design(settings))
    row.update(wait_percentiles(results["waits"], results["served_ns"]))
    row["waits"] = results["waits"].astype(np.int16)
    return row


def run_fine_sweep(path=DATA_DIR / "ext_fine_sweep.csv", samples=DATA_DIR / "ext_wait_samples.npz",
                   sweep=FINE_SWEEP, seeds=SEEDS, processes=None):
    """Run the fine sweep, write one row per run to `path`, and save the
    individual waits for the arrival rates in WAIT_SAMPLE_DEMANDS (all seeds
    of a setting pooled) to `samples`. Returns the number of runs."""
    with Pool(processes) as pool:
        rows = pool.map(describe_run, settings_grid(sweep, seeds), chunksize=4)
    pooled = {}
    for row in rows:
        waits = row.pop("waits")
        if row["demand"] in WAIT_SAMPLE_DEMANDS:
            key = f"{row['controller']}_{row['demand']}_{row['ns_probability']}_{row['reaction_delay']}"
            pooled.setdefault(key, []).append(waits)
    np.savez_compressed(samples, **{key: np.concatenate(w) for key, w in pooled.items()})
    write_rows(path, [file_names(row) for row in rows])
    return len(rows)


def write_rows(path, rows):
    """Write `rows` (dictionaries with the same keys) to the CSV file `path`."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- Extension 6: time-varying demand ------------------------------------

def surge_demand(t, base=SURGE["base"], peak=SURGE["peak"], start=SURGE["start"], ramp=SURGE["ramp"]):
    """Arrival rate at step t: `base`, then a linear rise to `peak` over `ramp`
    steps from `start`, a linear fall back over the next `ramp` steps, then `base`."""
    if t < start or t >= start + 2 * ramp:
        return base
    if t < start + ramp:
        return base + (peak - base) * (t - start) / ramp
    return peak - (peak - base) * (t - start - ramp) / ramp


def run_surge(args):
    """One run through the surge for (controller, ns_probability, seed). Webster's plan
    is designed for the base demand, as a plan made for normal traffic would be.
    Returns the cars on the road and the cars turned away so far at every step,
    and the mean wait of the cars arriving in each 50-step interval."""
    controller, ns_probability, seed = args
    model = Intersection(controller, demand=SURGE["base"], ns_probability=ns_probability, right_probability=0.3, seed=seed)
    steps = SURGE["steps"]
    cars, turned_away = np.empty(steps), np.empty(steps)
    for t in range(steps):
        model.demand = surge_demand(t)
        model.step()
        cars[t] = model.n_cars()
        turned_away[t] = model.blocked_arrivals
    totals, counts = np.zeros(steps // 50), np.zeros(steps // 50)
    for car in model.finished.values():
        totals[car["spawn_step"] // 50] += car["wait_time"]
        counts[car["spawn_step"] // 50] += 1
    with np.errstate(invalid="ignore"):
        waits = totals / counts
    return cars, turned_away, waits


def run_surge_experiment(path=DATA_DIR / "ext_surge.npz", processes=None):
    """Run the surge for every controller, north-south probability and seed and save the
    averages over seeds (keys like cars_adaptive_0.5). Returns the number of runs."""
    jobs = [(c, s, seed) for c in ("fixed", "webster", "adaptive") for s in SURGE_NS_PROBABILITIES
            for seed in SURGE_SEEDS]
    with Pool(processes) as pool:
        runs = pool.map(run_surge, jobs)
    arrays = {"demand": np.array([surge_demand(t) for t in range(SURGE["steps"])])}
    for c in ("fixed", "webster", "adaptive"):
        for s in SURGE_NS_PROBABILITIES:
            mine = [run for job, run in zip(jobs, runs) if job[:2] == (c, s)]
            for i, name in enumerate(("cars", "turned_away", "waits")):
                arrays[f"{name}_{c}_{s}"] = np.nanmean([run[i] for run in mine], axis=0)
    np.savez_compressed(path, **arrays)
    return len(jobs)


if __name__ == "__main__":
    chosen = [a for a in sys.argv[1:] if a.startswith("--")] or ["--fine", "--sensing", "--drivers", "--surge"]
    for option in chosen:
        start = time.perf_counter()
        if option == "--fine":
            n = run_fine_sweep()
        elif option == "--sensing":
            n = len(run_sweep(DATA_DIR / "ext_sensing.csv", SENSING))
        elif option == "--drivers":
            n = len(run_sweep(DATA_DIR / "ext_drivers.csv", DRIVERS))
        elif option == "--surge":
            n = run_surge_experiment()
        else:
            raise SystemExit(f"unknown option {option}")
        print(f"{option[2:]}: {n} runs in {time.perf_counter() - start:.0f} s")
