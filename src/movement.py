"""Moving vehicles forward, including the exit rule of the open road."""
import numpy as np

from src.constants import EMPTY, DRIVING, WRECKED


def move_vehicles(road):
    """Move every driving vehicle forward by its speed (new_x = x + v).

    Exit rule: a vehicle whose new_x is at or past the end of the road leaves,
    i.e. it is not copied into the new grid, statuses or clearance_timers.
    There is no wrap-around. Wrecks stay where they are.
    """
    new_grid = np.full_like(road.grid, EMPTY)
    new_statuses = np.zeros_like(road.statuses)
    new_timers = np.zeros_like(road.clearance_timers)

    wrecked = road.statuses == WRECKED
    new_grid[wrecked] = road.grid[wrecked]
    new_statuses[wrecked] = WRECKED
    new_timers[wrecked] = road.clearance_timers[wrecked]

    lanes, xs = np.nonzero((road.grid != EMPTY) & (road.statuses == DRIVING))
    for lane, x in zip(lanes, xs):
        v = road.grid[lane, x]
        new_x = x + v
        if new_x >= road.length:
            road.exited += 1
            continue
        assert new_grid[lane, new_x] == EMPTY, f"two vehicles in lane {lane}, cell {new_x}"
        new_grid[lane, new_x] = v

    road.grid, road.statuses, road.clearance_timers = new_grid, new_statuses, new_timers
