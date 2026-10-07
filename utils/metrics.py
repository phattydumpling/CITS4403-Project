"""Running one simulation and turning it into the numbers used in the experiments.

A measured run has three parts:

  warm-up    the intersection starts empty; these steps are simulated but not
             measured, so that queues and signal cycles reach their usual state
  window     the measured period: counters (throughput, blocked arrivals, phase
             changes, stalls) and queue lengths are taken over these steps, and
             the cars that arrive in this period are the ones whose waits count
  cool-down  extra steps, not measured, that let the cars which arrived near the
             end of the window finish, so their whole wait is counted rather
             than only the part before the run stopped
"""
import numpy as np

from src.constants import APPROACHES, MOVES, NORTH_SOUTH


def queue_lengths(model):
    """(approaches x moves) array with the number of cars in each pocket."""
    return np.array([[np.count_nonzero(model.pockets[(a, m)]) for m in MOVES]
                     for a in APPROACHES])


def current_phase(model):
    """Index of the phase that is green, or -1 during an all-red interval."""
    return -1 if not model.green else model.controller.phase


def run_and_measure(model, warmup=500, window=2000, cooldown=300, record=False):
    """Run `model` through warm-up, measured window and cool-down; summarise it.

    Returns a dictionary of results. Rates are per 100 steps of the window and
    wait times are in steps. Waits cover every car that arrived in the window,
    including time spent queuing off the map; a car that still has not reached
    the junction after the cool-down counts with its wait so far (a lower
    bound), and how many such cars there are is reported as cars_unfinished.
    If `record` is True, the dictionary also holds the queue length of every
    pocket ("queues", shape window x 4 x 3) and the green phase ("phases", -1
    for all-red) at every step of the window.
    """
    model.run(warmup)
    first_car = model.next_id                      # cars from here on arrived in the window
    start = {name: getattr(model, name)
             for name in ("entered_junction", "blocked_arrivals", "phase_changes", "stalls")}

    queues = np.empty((window, len(APPROACHES), len(MOVES)), dtype=np.int16)
    phases = np.empty(window, dtype=np.int8)
    outside = np.empty(window)
    for t in range(window):
        model.step()
        queues[t] = queue_lengths(model)
        phases[t] = current_phase(model)
        outside[t] = model.n_waiting_outside()
    last_car = model.next_id                       # cars from here on arrived in the cool-down
    rates = {name: 100 * (getattr(model, name) - start[name]) / window for name in start}

    model.run(cooldown)

    window_cars = [car for car_id, car in {**model.agents, **model.finished}.items()
                   if first_car <= car_id < last_car]
    served = [car for car in window_cars if "entry_step" in car]   # reached the junction
    waits = np.array([car["wait_time"] for car in window_cars], dtype=float)

    def mean_wait(keep):
        chosen = [w for w, car in zip(waits, window_cars) if keep(car)]
        return float(np.mean(chosen)) if chosen else float("nan")

    pocket_means = queues.mean(axis=(0, 1))          # average cars per pocket, by move
    results = {
        "mean_wait": float(waits.mean()) if len(waits) else float("nan"),
        "wait_ns": mean_wait(lambda r: r["approach"] in NORTH_SOUTH),
        "wait_ew": mean_wait(lambda r: r["approach"] not in NORTH_SOUTH),
        "wait_left": mean_wait(lambda r: r["move"] == "left"),
        "wait_straight": mean_wait(lambda r: r["move"] == "straight"),
        "wait_right": mean_wait(lambda r: r["move"] == "right"),
        "cars_measured": len(served),
        "cars_unfinished": len(window_cars) - len(served),
        "throughput": rates["entered_junction"],
        "blocked_arrivals": rates["blocked_arrivals"],
        "phase_changes": rates["phase_changes"],
        "stalls": rates["stalls"],
        "queue_left": float(pocket_means[MOVES.index("left")]),
        "queue_straight": float(pocket_means[MOVES.index("straight")]),
        "queue_right": float(pocket_means[MOVES.index("right")]),
        "queue_outside": float(outside.mean()),    # cars waiting off the map, all approaches
    }
    if record:
        results["queues"] = queues
        results["phases"] = phases
    return results
