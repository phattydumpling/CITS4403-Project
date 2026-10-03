"""
Multi-lane highway cellular automaton (Nagel-Schreckenberg style).

The road is a grid of `lanes` x `length` cells, stored as three parallel arrays:
  grid             -1 if the cell is empty, otherwise the speed (0..max_v, in
                   cells per time step) of the vehicle in that cell
  statuses         0 for a driving vehicle, 1 for a vehicle frozen in a wreck
  clearance_timers time steps left until a wreck in that cell is cleared

The road is open: vehicles enter at x = 0 and leave past x = length - 1.
"""
import numpy as np

from src.constants import EMPTY
from src.boundaries import spawn_vehicles
from src.driving import update_speeds
from src.lane_changing import change_lanes
from src.movement import move_vehicles


class Highway:
    def __init__(self, lanes=3, length=500, max_v=5, spawn_probability=0.3,
                 crash_lookahead=15, seed=None):
        self.lanes = lanes
        self.length = length
        self.max_v = max_v
        self.spawn_probability = spawn_probability
        self.crash_lookahead = crash_lookahead  # cells ahead a driver looks for wrecks
        self.rng = np.random.default_rng(seed)
        self.time = 0

        self.grid = np.full((lanes, length), EMPTY, dtype=int)
        self.statuses = np.zeros((lanes, length), dtype=int)
        self.clearance_timers = np.zeros((lanes, length), dtype=int)

        # Running totals, used to check the model and measure results.
        self.spawned = 0
        self.exited = 0
        self.lane_changes = 0

    def n_vehicles(self):
        """Number of vehicles (driving or wrecked) currently on the road."""
        return int(np.sum(self.grid != EMPTY))

    def step(self):
        """Advance the simulation by one time step."""
        spawn_vehicles(self, self.spawn_probability)
        change_lanes(self, self.crash_lookahead)
        update_speeds(self)
        move_vehicles(self)
        self.time += 1

    def run(self, steps):
        for _ in range(steps):
            self.step()
