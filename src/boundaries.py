"""Open road: vehicles enter at x = 0 (the exit rule is in movement.py)."""
from src.constants import EMPTY
from utils.gaps import gap_ahead


def spawn_vehicles(road, spawn_probability):
    """Entry rule, run at the start of every time step.

    For each lane, draw a random number in [0, 1). A new vehicle enters at
    x = 0, with speed 0 or 1, if the number is below spawn_probability and
    there is room: the entry cell and the max_v cells after it are all empty.
    Requiring room stops new vehicles appearing right behind a slow vehicle
    that has only just entered, which would otherwise make the entrance the
    only place where vehicles are close enough to crash.
    """
    for lane in range(road.lanes):
        has_room = (road.grid[lane, 0] == EMPTY
                    and gap_ahead(road.grid, lane, 0, road.max_v) >= road.max_v)
        if road.rng.random() < spawn_probability and has_room:
            road.grid[lane, 0] = road.rng.integers(0, 2)  # speed 0 or 1
            road.spawned += 1
