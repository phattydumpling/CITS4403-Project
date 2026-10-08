"""Traffic signal phases and the two controllers that choose between them.

Vehicles drive on the left (as in Australia), so right turns cross oncoming
traffic and get their own protected phase. Only one phase is green at a time,
so conflicting movements never have green together:

    phase 0: north-south straight and left
    phase 1: north-south right
    phase 2: east-west straight and left
    phase 3: east-west right

After every green there is an all-red interval in which nothing may enter the
junction. It stands for the whole change period of a real signal (amber, then
all-red) and lasts as long as a car takes to cross the junction.
"""
import numpy as np

from src.constants import APPROACHES, EMPTY, MOVES

PHASES = (
    frozenset({("N", "straight"), ("N", "left"), ("S", "straight"), ("S", "left")}),
    frozenset({("N", "right"), ("S", "right")}),
    frozenset({("E", "straight"), ("E", "left"), ("W", "straight"), ("W", "left")}),
    frozenset({("E", "right"), ("W", "right")}),
)


def phase_queue(model, phase, cells=None):
    """Number of cars waiting in the pockets that `phase` serves. If `cells` is
    given, only the `cells` cells nearest the stop line are counted (the range
    of a detector); None counts the whole pocket."""
    return sum(int(np.count_nonzero(model.pockets[key][:cells])) for key in PHASES[phase])


def phase_wait(model, phase, cells=None):
    """Total time, in steps, that the cars in the pockets `phase` serves have
    spent stopped so far (the sum of their wait times). `cells` limits the
    count to the cells nearest the stop line, as in phase_queue()."""
    return sum(model.agents[int(car_id)]["wait_time"]
               for key in PHASES[phase] for car_id in model.pockets[key][:cells] if car_id != EMPTY)


class SignalController:
    """Behaviour shared by both controllers: one phase green at a time, and an
    all-red interval of `all_red` steps after every green. Subclasses decide
    when a green ends (should_end_green) and which phase comes next (next_phase)."""

    def __init__(self, all_red):
        self.all_red = all_red
        self.phase = 0
        self.in_all_red = False
        self.timer = 0          # steps already spent in the current green or all-red

    def update(self, model):
        """Called once per step, before cars move. Sets model.green."""
        if self.in_all_red:
            if self.timer >= self.all_red:
                self.phase = self.next_phase(model)
                self.in_all_red = False
                self.timer = 0
        elif self.should_end_green(model):
            self.in_all_red = True
            self.timer = 0
            model.phase_changes += 1
        model.green = set() if self.in_all_red else set(PHASES[self.phase])
        self.timer += 1

    def should_end_green(self, model):
        raise NotImplementedError

    def next_phase(self, model):
        raise NotImplementedError


class FixedTimeController(SignalController):
    """Phases are served in the order 0, 1, 2, 3 with fixed green times.

    `green_times` gives the green of each phase in steps; a single number gives
    every phase the same green (the untuned plan: 20 steps each).
    """

    def __init__(self, green_times=20, all_red=3):
        super().__init__(all_red)
        if np.isscalar(green_times):
            green_times = [green_times] * len(PHASES)
        self.green_times = [int(g) for g in green_times]

    def should_end_green(self, model):
        return self.timer >= self.green_times[self.phase]

    def next_phase(self, model):
        return (self.phase + 1) % len(PHASES)


def saturation_flow(reaction_delay):
    """Cars per step that a queued lane discharges on green: 1 car every two
    steps under the parallel update, slowed by the reaction delay d to
    1 / (2 + d / (1 - d)) (measured in the model)."""
    return 1 / (2 + reaction_delay / (1 - reaction_delay))


def webster_green_times(model, all_red=3, min_green=5, max_cycle=120):
    """Fixed-time plan designed from the average demand with Webster's (1958) method.

    For each phase, the flow ratio y is the arrival rate of its busiest lane
    divided by the saturation flow. With Y the sum of the four y and L the time
    lost to all-red in one cycle, the cycle length is C = (1.5 L + 5) / (1 - Y),
    limited to `max_cycle` (and used at the limit when the demand is at or
    above capacity). The green time C - L is shared between the phases in
    proportion to y, and every green is at least `min_green` steps.
    """
    from src.spawning import arrival_probability          # avoids a circular import
    s = saturation_flow(model.reaction_delay)
    lane_flow = {(a, m): arrival_probability(model, a) * share
                 for a in APPROACHES for m, share in zip(MOVES, model.move_shares)}
    y = [max(lane_flow[key] for key in phase) / s for phase in PHASES]
    total_y = sum(y)
    lost = len(PHASES) * all_red
    cycle = (1.5 * lost + 5) / (1 - total_y) if total_y < 1 else max_cycle
    cycle = min(cycle, max_cycle)
    green = cycle - lost
    if total_y == 0:
        return [min_green] * len(PHASES)
    return [max(min_green, round(green * yi / total_y)) for yi in y]


class AdaptiveController(SignalController):
    """Gives green to the phase whose waiting cars have waited longest in total.

    A phase's priority is phase_wait(): the sum of the wait times of the cars
    queued for it. It grows every step its cars sit on red (by one per waiting
    car) and falls as they leave on green, so a busy phase builds priority
    quickly but a quiet phase cannot be passed over forever. This is similar in
    spirit to Gershenson's (2005) self-organising traffic lights.

    Once the current phase has been green for at least `min_green` steps, it
    ends as soon as its own queue is empty, it reaches `max_green` steps, or
    another phase has a higher priority. After the all-red interval the next
    green goes to the phase (other than the one that just ended) with the
    highest priority; ties go to the phase that comes first in the normal order.

    `detector_range` limits what the controller can see to that many cells of
    each pocket next to the stop line (None = the whole pocket). With 0 it sees
    nothing, so every green lasts min_green and phases follow the normal order.
    """

    def __init__(self, min_green=5, max_green=40, all_red=3, detector_range=None):
        super().__init__(all_red)
        self.min_green = min_green
        self.max_green = max_green
        self.detector_range = detector_range

    def wait(self, model, phase):
        """Priority of `phase` as far as the detectors can see."""
        return phase_wait(model, phase, self.detector_range)

    def should_end_green(self, model):
        if self.timer < self.min_green:
            return False
        if phase_queue(model, self.phase, self.detector_range) == 0 or self.timer >= self.max_green:
            return True
        highest_other = max(self.wait(model, p) for p in range(len(PHASES)) if p != self.phase)
        return highest_other > self.wait(model, self.phase)

    def next_phase(self, model):
        order = [(self.phase + k) % len(PHASES) for k in range(1, len(PHASES))]
        return max(order, key=lambda p: self.wait(model, p))   # max keeps the first on ties
