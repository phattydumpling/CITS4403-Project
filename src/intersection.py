"""
Four-way signalised intersection as a cellular automaton with car agents.

Each approach (N, E, S, W) is a shared feeder lane that splits into three
turning pockets (left, straight, right) ending at the stop line:

    map edge  [feeder: index feeder_length-1 ... 0]  ->  split
              [left pocket:     index pocket_length-1 ... 0]  -> stop line
              [straight pocket: index pocket_length-1 ... 0]  -> stop line
              [right pocket:    index pocket_length-1 ... 0]  -> stop line

Every lane is a 1D array of car IDs (0 = empty); cars travel towards index 0.
The details of each car are kept in the `agents` registry, keyed by car ID.
"""
import numpy as np

from src.constants import APPROACHES, EMPTY, MOVES
from src.spawning import spawn_cars


class Intersection:
    def __init__(self, demand=0.3, ns_share=0.5, left_share=0.2, right_share=0.2,
                 feeder_length=30, pocket_length=10, seed=None):
        if left_share + right_share > 1:
            raise ValueError("left_share + right_share must be at most 1")
        self.demand = demand            # average arrival probability per approach per step
        self.ns_share = ns_share        # fraction of arrivals from north and south
        self.move_shares = (left_share, 1 - left_share - right_share, right_share)
        self.rng = np.random.default_rng(seed)
        self.time = 0

        self.feeders = {a: np.full(feeder_length, EMPTY, dtype=int) for a in APPROACHES}
        self.pockets = {(a, m): np.full(pocket_length, EMPTY, dtype=int)
                        for a in APPROACHES for m in MOVES}
        self.agents = {}                # car ID -> details of that car
        self.next_id = 1

        # Running totals, used to check the model and measure results.
        self.spawned = 0
        self.blocked_arrivals = 0       # arrivals turned away because the entry cell was full

    def n_cars(self):
        """Number of cars currently in the feeder lanes and pockets."""
        lanes = list(self.feeders.values()) + list(self.pockets.values())
        return int(sum(np.count_nonzero(lane) for lane in lanes))

    def step(self):
        """Advance the simulation by one time step."""
        spawn_cars(self)
        self.time += 1

    def run(self, steps):
        for _ in range(steps):
            self.step()
