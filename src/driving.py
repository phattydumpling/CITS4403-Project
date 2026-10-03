"""Speed rule for driving vehicles, including distracted drivers and crashes."""
import numpy as np

from src.constants import EMPTY, DRIVING, WRECKED
from utils.gaps import gap_ahead


def update_speeds(road, p_distracted, p_slow, clearance_time):
    """Speed rule for every driving vehicle (statuses == 0).

    1. Roll for distraction (probability p_distracted).
       - A distracted driver keeps their current speed: they neither speed up
         nor brake for the gap ahead. A stopped distracted driver therefore
         stays stopped instead of creeping into the vehicle in front.
       - An alert driver follows the Nagel-Schreckenberg rules: speed up by 1
         (up to max_v), slow down to the gap ahead (v = min(v, gap)), then
         with probability p_slow slow down by 1 more (random slowdown).
    2. Collision: if the speed is now larger than the gap, the driver hits the
       vehicle ahead. Both vehicles become wrecks with speed 0, and both get a
       clearance timer of `clearance_time` steps.

    `gap` is the number of empty cells ahead, so the vehicle that is hit sits
    at x + gap + 1. Vehicles are handled back to front within each lane, so a
    vehicle hit earlier in this loop is skipped when its own turn comes.
    """
    lanes, xs = np.nonzero((road.grid != EMPTY) & (road.statuses == DRIVING))
    for lane, x in zip(lanes, xs):
        if road.statuses[lane, x] == WRECKED:  # hit from behind earlier in this loop
            continue
        v = road.grid[lane, x]
        gap = gap_ahead(road.grid, lane, x, road.max_v)

        distracted = road.rng.random() < p_distracted
        if not distracted:
            v = min(v + 1, road.max_v)
            v = min(v, gap)
            if road.rng.random() < p_slow:
                v = max(v - 1, 0)

        if v > gap:  # only possible for a distracted driver with a vehicle in reach
            crash_x = x + gap + 1
            for cell in (x, crash_x):
                road.statuses[lane, cell] = WRECKED
                road.grid[lane, cell] = 0
                road.clearance_timers[lane, cell] = clearance_time
            road.crashes += 1
        else:
            road.grid[lane, x] = v
