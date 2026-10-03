"""Speed rule for driving vehicles."""
import numpy as np

from src.constants import EMPTY, DRIVING
from utils.gaps import gap_ahead


def update_speeds(road):
    """Nagel-Schreckenberg speed rule for every driving vehicle:
    speed up by 1 (up to max_v), then slow down so as not to pass the gap ahead.
    """
    lanes, xs = np.nonzero((road.grid != EMPTY) & (road.statuses == DRIVING))
    for lane, x in zip(lanes, xs):
        v = min(road.grid[lane, x] + 1, road.max_v)
        gap = gap_ahead(road.grid, lane, x, road.max_v)
        road.grid[lane, x] = min(v, gap)
