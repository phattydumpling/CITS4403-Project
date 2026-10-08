"""Moving cars along the lanes, over the stop line and across the junction.

Every car decides whether to move from the state at the start of the step
(parallel update): it advances one cell only if the cell ahead was empty when
the step began. A queue therefore starts off one car at a time, with the gap
travelling backwards through it, as at a real green light.

Reaction delay (slow-to-start): a car that was stopped in the previous step
and could now move stays where it is with probability `model.reaction_delay`
(or its own driver's delay, if drivers differ). This applies to every car
starting from standstill, wherever it is in the queue, not only to the first
car after the light turns green.
"""
import numpy as np

from src.constants import EMPTY


def move_cars(model):
    """Advance every car by at most one cell and record who had to wait."""
    clear_junction(model)
    for record in model.agents.values():
        record["reaction_delayed"] = False
    start_feeders = {a: lane.copy() for a, lane in model.feeders.items()}
    start_pockets = {key: lane.copy() for key, lane in model.pockets.items()}
    moved = set()

    release_at_stop_lines(model, start_pockets, moved)
    feed_pockets(model, start_feeders, start_pockets, moved)
    for key, lane in model.pockets.items():
        advance(model, lane, start_pockets[key], moved)
    for approach, lane in model.feeders.items():
        advance(model, lane, start_feeders[approach], moved)

    count_waiting(model, moved)


def ready_to_move(model, car_id):
    """Slow-to-start rule. A car that was stopped last step stalls with its
    driver's reaction delay (model.reaction_delay unless drivers differ); a car
    that was already moving never does."""
    record = model.agents[car_id]
    stalls = (record["stopped"] and record["reaction_delay"] > 0
              and model.driver_rng.random() < record["reaction_delay"])
    if stalls:
        record["reaction_delayed"] = True
        model.stalls += 1
    return not stalls


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
    """A car at the stop line (index 0) enters the junction if its movement has
    green and it does not stall."""
    for key, pocket in model.pockets.items():
        car_id = int(start_pockets[key][0])
        if car_id != EMPTY and key in model.green and ready_to_move(model, car_id):
            pocket[0] = EMPTY
            model.crossing.append((car_id, model.crossing_time))
            model.agents[car_id]["entry_step"] = model.time
            model.entered_junction += 1
            moved.add(car_id)


def feed_pockets(model, start_feeders, start_pockets, moved):
    """The car at the front of each feeder lane moves into the last cell of its
    pocket if that cell was empty. If the pocket is full the car waits, and so
    does every car behind it in the feeder lane, whatever their movement.

    Spillback is counted when the pocket is completely full:
    model.full_pocket_blocks for every such step, and model.other_movement_blocks
    when the car right behind wants a different pocket (so traffic for another
    movement is held up)."""
    for approach, feeder in model.feeders.items():
        car_id = int(start_feeders[approach][0])
        if car_id == EMPTY:
            continue
        key = (approach, model.agents[car_id]["move"])
        if np.all(start_pockets[key] != EMPTY):
            model.full_pocket_blocks += 1
            follower = int(start_feeders[approach][1])
            if follower != EMPTY and model.agents[follower]["move"] != key[1]:
                model.other_movement_blocks += 1
        if start_pockets[key][-1] == EMPTY and ready_to_move(model, car_id):
            model.pockets[key][-1] = car_id
            feeder[0] = EMPTY
            moved.add(car_id)


def advance(model, lane, start, moved):
    """Move each car one cell towards index 0 if that cell was empty at the start
    of the step and the car does not stall. Cells changed earlier in this step
    (the stop line and the feeder-to-pocket hand-over) are not affected, because
    they are judged on `start`, not on `lane`."""
    for i in np.flatnonzero((start[1:] != EMPTY) & (start[:-1] == EMPTY)) + 1:
        car_id = int(start[i])
        if not ready_to_move(model, car_id):
            continue
        lane[i - 1] = car_id
        lane[i] = EMPTY
        moved.add(car_id)


def count_waiting(model, moved):
    """Every car still in a lane that did not move this step has waited one step
    and counts as stopped for the slow-to-start rule next step. Cars waiting in
    an entry queue off the map have also waited one step."""
    lanes = list(model.feeders.values()) + list(model.pockets.values())
    for lane in lanes:
        for car_id in lane[lane != EMPTY]:
            record = model.agents[int(car_id)]
            record["stopped"] = int(car_id) not in moved
            if record["stopped"]:
                record["wait_time"] += 1
    for queue in model.entry_queues.values():
        for car_id in queue:
            model.agents[car_id]["wait_time"] += 1
