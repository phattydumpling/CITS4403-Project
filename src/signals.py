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

PHASES = (
    frozenset({("N", "straight"), ("N", "left"), ("S", "straight"), ("S", "left")}),
    frozenset({("N", "right"), ("S", "right")}),
    frozenset({("E", "straight"), ("E", "left"), ("W", "straight"), ("W", "left")}),
    frozenset({("E", "right"), ("W", "right")}),
)


def phase_queue(model, phase):
    """Number of cars waiting in the pockets that `phase` serves."""
    return sum(int(np.count_nonzero(model.pockets[key])) for key in PHASES[phase])


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
    """Every phase gets `green_time` steps of green, in the order 0, 1, 2, 3."""

    def __init__(self, green_time=20, all_red=3):
        super().__init__(all_red)
        self.green_time = green_time

    def should_end_green(self, model):
        return self.timer >= self.green_time

    def next_phase(self, model):
        return (self.phase + 1) % len(PHASES)


class AdaptiveController(SignalController):
    """Gives green to the longest queue.

    Once the current phase has been green for at least `min_green` steps, it
    ends as soon as its own queue is empty, it reaches `max_green` steps, or
    another phase has a longer queue. After the all-red interval the next green
    goes to the phase (other than the one that just ended) with the longest
    queue; ties go to the phase that comes first in the normal order.
    """

    def __init__(self, min_green=5, max_green=40, all_red=3):
        super().__init__(all_red)
        self.min_green = min_green
        self.max_green = max_green

    def should_end_green(self, model):
        if self.timer < self.min_green:
            return False
        own = phase_queue(model, self.phase)
        longest_other = max(phase_queue(model, p) for p in range(len(PHASES)) if p != self.phase)
        return own == 0 or self.timer >= self.max_green or longest_other > own

    def next_phase(self, model):
        order = [(self.phase + k) % len(PHASES) for k in range(1, len(PHASES))]
        return max(order, key=lambda p: phase_queue(model, p))   # max keeps the first on ties
