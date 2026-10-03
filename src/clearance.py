"""Removing wrecks once their clearance timer runs out."""
from src.constants import EMPTY, DRIVING


def clear_wrecks(road):
    """Run at the end of every time step.

    Every clearance timer above 0 counts down by 1. When a timer reaches 0 the
    wreck in that cell is towed away: its status goes back to 0 and the cell
    becomes empty. Vehicles queued behind it start moving again on later steps
    as space opens up.
    """
    counting = road.clearance_timers > 0
    road.clearance_timers[counting] -= 1
    cleared = counting & (road.clearance_timers == 0)
    road.statuses[cleared] = DRIVING
    road.grid[cleared] = EMPTY
    road.towed += int(cleared.sum())
