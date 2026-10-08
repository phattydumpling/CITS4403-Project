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
    wait times are in steps. Waits are averaged over the cars that arrived in
    the window and reached the junction by the end of the cool-down; cars that
    had not are counted in cars_unfinished. Arrivals that found the entry full
    are counted in blocked_arrivals: by default they never enter, so they have
    no wait; with queue_blocked_arrivals their time queuing off the map is part
    of their wait. If `record` is True, the dictionary also holds the queue length of every
    pocket ("queues", shape window x 4 x 3), the green phase ("phases", -1
    for all-red) and the number of cars on the approaches ("cars") at every
    step of the window, plus the wait of every measured car ("waits") and
    whether it came from the north or south ("served_ns").
    """
    model.run(warmup)
    first_car = model.next_id                      # cars from here on arrived in the window
    start = {name: getattr(model, name)
             for name in ("entered_junction", "blocked_arrivals", "phase_changes", "stalls",
                          "split_blocks", "cross_blocks")}

    queues = np.empty((window, len(APPROACHES), len(MOVES)), dtype=np.int16)
    phases = np.empty(window, dtype=np.int8)
    outside = np.empty(window)
    cars = np.empty(window, dtype=np.int16)
    for t in range(window):
        model.step()
        queues[t] = queue_lengths(model)
        phases[t] = current_phase(model)
        outside[t] = model.n_waiting_outside()
        cars[t] = model.n_cars()
    last_car = model.next_id                       # cars from here on arrived in the cool-down
    rates = {name: 100 * (getattr(model, name) - start[name]) / window for name in start}

    model.run(cooldown)

    window_cars = [car for car_id, car in {**model.agents, **model.finished}.items()
                   if first_car <= car_id < last_car]
    served = [car for car in window_cars if "entry_step" in car]   # reached the junction
    waits = np.array([car["wait_time"] for car in served], dtype=float)

    def mean_wait(keep):
        chosen = [w for w, car in zip(waits, served) if keep(car)]
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
        "split_blocks": rates["split_blocks"],    # spillback: car at the split facing a full pocket
        "cross_blocks": rates["cross_blocks"],    # ... holding up a car bound for another pocket
        "queue_left": float(pocket_means[MOVES.index("left")]),
        "queue_straight": float(pocket_means[MOVES.index("straight")]),
        "queue_right": float(pocket_means[MOVES.index("right")]),
        "queue_outside": float(outside.mean()),    # cars waiting off the map, all approaches
    }
    if record:
        results["queues"] = queues
        results["phases"] = phases
        results["cars"] = cars                     # cars on the approaches at every step
        results["waits"] = waits                   # wait of every car measured
        results["served_ns"] = np.array([car["approach"] in NORTH_SOUTH for car in served])
    return results
