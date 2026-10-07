"""
Four-way signalised intersection as a cellular automaton with car agents.

Each approach (N, E, S, W) is a shared feeder lane that splits into three
turning pockets (left, straight, right) ending at the stop line:

    map edge  [feeder: index feeder_length-1 ... 0]  ->  split
              [left pocket:     index pocket_length-1 ... 0]  -> stop line
              [straight pocket: index pocket_length-1 ... 0]  -> stop line
              [right pocket:    index pocket_length-1 ... 0]  -> stop line

Every lane is a 1D array of car IDs (0 = empty); cars travel towards index 0.
A car that crosses its stop line on green spends `crossing_time` steps in the
junction and then leaves the model. The details of each car are kept in the
`agents` registry, keyed by car ID; cars that have left move to `finished`.
"""
import numpy as np

from src.constants import APPROACHES, EMPTY, MOVES
from src.movement import move_cars
from src.signals import AdaptiveController, FixedTimeController
from src.spawning import spawn_cars


class Intersection:
    def __init__(self, controller="fixed", demand=0.2, ns_share=0.5, turning=0.5,
                  reaction_delay=0.0, green_time=20, min_green=5,
                 max_green=40, all_red=3, feeder_length=30, pocket_length=10,
                 crossing_time=3, seed=None):
        if turning > 1:
            raise ValueError("turning must be at most 1")
        if controller == "fixed":
            self.controller = FixedTimeController(green_time, all_red)
        elif controller == "adaptive":
            self.controller = AdaptiveController(min_green, max_green, all_red)
        else:
            raise ValueError("controller must be 'fixed' or 'adaptive'")
        self.demand = demand            # average arrival probability per approach per step
        self.ns_share = ns_share        # fraction of arrivals from north and south
        self.move_shares = (turning/2, 1 - turning, turning/2)
        self.reaction_delay = reaction_delay  # chance a car starting from standstill stalls
        self.crossing_time = crossing_time  # steps a car takes to cross the junction
        # Separate random streams for arrivals and for driver behaviour, so the
        # fixed-time and adaptive runs with the same seed get identical arrivals.
        arrival_seed, driver_seed = np.random.SeedSequence(seed).spawn(2)
        self.arrival_rng = np.random.default_rng(arrival_seed)
        self.driver_rng = np.random.default_rng(driver_seed)
        self.time = 0

        self.feeders = {a: np.full(feeder_length, EMPTY, dtype=int) for a in APPROACHES}
        self.pockets = {(a, m): np.full(pocket_length, EMPTY, dtype=int)
                        for a in APPROACHES for m in MOVES}
        self.crossing = []              # (car ID, steps left) for cars inside the junction
        self.agents = {}                # car ID -> details of a car still in the model
        self.finished = {}              # car ID -> details of a car that has left
        self.next_id = 1

        # Movements that currently have a green light, as (approach, move) pairs.
        # The signal controller sets this at the start of every step.
        self.green = set()

        # Running totals, used to check the model and measure results.
        self.spawned = 0
        self.blocked_arrivals = 0       # arrivals turned away because the entry cell was full
        self.entered_junction = 0       # cars that crossed a stop line
        self.phase_changes = 0          # greens ended by the controller
        self.stalls = 0                 # times a car stalled when it could have started

    def n_cars(self):
        """Number of cars currently in the feeder lanes and pockets."""
        lanes = list(self.feeders.values()) + list(self.pockets.values())
        return int(sum(np.count_nonzero(lane) for lane in lanes))

    def step(self):
        """Advance the simulation by one time step."""
        spawn_cars(self)
        self.controller.update(self)
        move_cars(self)
        self.time += 1

    def run(self, steps):
        for _ in range(steps):
            self.step()
