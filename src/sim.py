"""
Multi-lane highway cellular automaton (Nagel-Schreckenberg style).

The road is a grid of `lanes` x `length` cells, stored as three parallel arrays:
  grid             -1 if the cell is empty, otherwise the speed (0..max_v, in
                   cells per time step) of the vehicle in that cell
  statuses         0 for a driving vehicle, 1 for a vehicle frozen in a wreck
  clearance_timers time steps left until a wreck in that cell is cleared
"""
import numpy as np

from src.constants import EMPTY


class Highway:
    def __init__(self, lanes=3, length=500, max_v=5, seed=None):
        self.lanes = lanes
        self.length = length
        self.max_v = max_v
        self.rng = np.random.default_rng(seed)
        self.time = 0

        self.grid = np.full((lanes, length), EMPTY, dtype=int)
        self.statuses = np.zeros((lanes, length), dtype=int)
        self.clearance_timers = np.zeros((lanes, length), dtype=int)
