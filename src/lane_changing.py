"""Lane changing, applied before speeds are updated in each time step."""
import numpy as np

from src.constants import EMPTY, DRIVING
from utils.gaps import gap_ahead, gap_behind


def change_lanes(road, crash_lookahead):
    """Let driving vehicles move sideways into a neighbouring lane.

    Incentive: a vehicle wants to change lane if the gap ahead is smaller than
    its speed, or if there is a wreck within `crash_lookahead` cells ahead in
    its own lane.

    Safety: it moves into a neighbouring lane only if
      1. the cell beside it in that lane is empty,
      2. the gap ahead in that lane is larger than its speed, and
      3. the gap behind in that lane is larger than max_v.

    Vehicles are handled one at a time in random order, and each tries its two
    neighbouring lanes in random order, so no lane or direction is favoured.
    Each move updates the grid straight away, so two vehicles can never move
    into the same cell. Wrecks never change lanes.
    """
    lanes, xs = np.nonzero((road.grid != EMPTY) & (road.statuses == DRIVING))
    for i in road.rng.permutation(len(lanes)):
        lane, x = lanes[i], xs[i]
        v = road.grid[lane, x]

        blocked = gap_ahead(road.grid, lane, x, road.max_v) < v
        wreck_ahead = road.statuses[lane, x + 1 : x + 1 + crash_lookahead].any()
        if not (blocked or wreck_ahead):
            continue

        for target in road.rng.permutation([lane - 1, lane + 1]):
            if not 0 <= target < road.lanes:
                continue
            if (road.grid[target, x] == EMPTY
                    and gap_ahead(road.grid, target, x, road.max_v + 1) > v
                    and gap_behind(road.grid, target, x, road.max_v + 1) > road.max_v):
                road.grid[target, x] = v
                road.grid[lane, x] = EMPTY
                road.lane_changes += 1
                break
