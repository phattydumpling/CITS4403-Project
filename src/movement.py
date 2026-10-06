"""Moving cars along the lanes, over the stop line and across the junction.

Every car decides whether to move from the state at the start of the step
(parallel update): it advances one cell only if the cell ahead was empty when
the step began. A queue therefore starts off one car at a time, with the gap
travelling backwards through it, as at a real green light.
"""
import numpy as np

from src.constants import EMPTY


def move_cars(model):
    """Advance every car by at most one cell and record who had to wait."""
    clear_junction(model)
    start_feeders = {a: lane.copy() for a, lane in model.feeders.items()}
    start_pockets = {key: lane.copy() for key, lane in model.pockets.items()}
    moved = set()

    release_at_stop_lines(model, start_pockets, moved)
    feed_pockets(model, start_feeders, start_pockets, moved)
    for key, lane in model.pockets.items():
        advance(lane, start_pockets[key], moved)
    for approach, lane in model.feeders.items():
        advance(lane, start_feeders[approach], moved)

    count_waiting(model, moved)


def clear_junction(model):
    """Cars in the junction count down their crossing time; at 0 they leave the model."""
    still_crossing = []
    for car_id, steps_left in model.crossing:
        if steps_left > 1:
            still_crossing.append((car_id, steps_left - 1))
        else:
            record = model.agents.pop(car_id)
            record["exit_step"] = model.time
            model.finished[car_id] = record
    model.crossing = still_crossing


def release_at_stop_lines(model, start_pockets, moved):
    """A car at the stop line (index 0) enters the junction if its movement has green."""
    for key, pocket in model.pockets.items():
        car_id = int(start_pockets[key][0])
        if car_id != EMPTY and key in model.green:
            pocket[0] = EMPTY
            model.crossing.append((car_id, model.crossing_time))
            model.agents[car_id]["entry_step"] = model.time
            model.entered_junction += 1
            moved.add(car_id)


def feed_pockets(model, start_feeders, start_pockets, moved):
    """The car at the front of each feeder lane moves into the last cell of its
    pocket if that cell was empty. If the pocket is full the car waits, and so
    does every car behind it in the feeder lane, whatever their movement."""
    for approach, feeder in model.feeders.items():
        car_id = int(start_feeders[approach][0])
        if car_id == EMPTY:
            continue
        key = (approach, model.agents[car_id]["move"])
        if start_pockets[key][-1] == EMPTY:
            model.pockets[key][-1] = car_id
            feeder[0] = EMPTY
            moved.add(car_id)


def advance(lane, start, moved):
    """Move each car one cell towards index 0 if that cell was empty at the start
    of the step. Cells changed earlier in this step (the stop line and the
    feeder-to-pocket hand-over) are not affected, because they are judged on
    `start`, not on `lane`."""
    for i in np.flatnonzero((start[1:] != EMPTY) & (start[:-1] == EMPTY)) + 1:
        car_id = int(start[i])
        lane[i - 1] = car_id
        lane[i] = EMPTY
        moved.add(car_id)


def count_waiting(model, moved):
    """Every car still in a lane that did not move this step has waited one step."""
    lanes = list(model.feeders.values()) + list(model.pockets.values())
    for lane in lanes:
        for car_id in lane[lane != EMPTY]:
            if int(car_id) not in moved:
                model.agents[int(car_id)]["wait_time"] += 1
