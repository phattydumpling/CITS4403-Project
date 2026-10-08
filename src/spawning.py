"""New cars arriving at the outer end of each approach."""
from src.constants import APPROACHES, EMPTY, MOVES, NORTH_SOUTH


def arrival_probability(model, approach):
    """Chance that a car arrives on `approach` in one time step.

    `demand` is the average over the four approaches. `ns_share` is the
    fraction of all arrivals that come from the north and south, so 0.5 is
    balanced and 0.9 sends 90% of the traffic north-south. Probabilities are
    capped at 1 (at most one car per approach per step).
    """
    share = model.ns_share if approach in NORTH_SOUTH else 1 - model.ns_share
    return min(1.0, 2 * model.demand * share)


def spawn_cars(model):
    """Arrival rule, run at the start of every time step.

    For each approach, a car arrives with arrival_probability() and chooses its
    move (left / straight / right) from the turning shares. It enters the outer
    cell of the feeder lane if that cell is free. If the cell is occupied the
    arrival is counted in `model.blocked_arrivals` and:
      - by default it is turned away and never enters the model;
      - with model.queue_blocked_arrivals it joins the back of an entry queue
        off the map instead. The car at the front of that queue takes the outer
        cell as soon as it is free (before any new arrival), and the time spent
        queuing counts as waiting time.

    Both random numbers are drawn from model.arrival_rng on every step for
    every approach, whatever the state of the road, so two runs with the same
    seed see exactly the same arrivals even if their signals differ.
    """
    for approach in APPROACHES:
        arrives = model.arrival_rng.random() < arrival_probability(model, approach)
        move = MOVES[model.arrival_rng.choice(len(MOVES), p=model.move_shares)]
        feeder = model.feeders[approach]
        queue = model.entry_queues[approach]
        if queue and feeder[-1] == EMPTY:
            feeder[-1] = queue.popleft()
        if not arrives:
            continue
        if feeder[-1] != EMPTY:
            model.blocked_arrivals += 1
            if not model.queue_blocked_arrivals:
                continue                      # turned away: the car never enters
            car_id = new_car(model, approach, move)
            queue.append(car_id)
            model.agents[car_id]["stopped"] = True
        else:
            feeder[-1] = new_car(model, approach, move)


def new_car(model, approach, move):
    """Register a new car in the agents registry and return its ID."""
    car_id = model.next_id
    model.next_id += 1
    delay = model.reaction_delay
    if model.slow_driver_share is not None:   # different drivers (see Intersection)
        slow = model.driver_rng.random() < model.slow_driver_share
        delay = model.reaction_delay / model.slow_driver_share if slow else 0.0
    model.agents[car_id] = {
        "approach": approach,
        "move": move,
        "spawn_step": model.time,
        "reaction_delay": delay,     # chance this driver stalls when starting from standstill
        "wait_time": 0,              # steps spent not moving (including outside the map)
        "stopped": False,            # did not move last step (arrives already moving)
        "reaction_delayed": False,   # stalled this step when it could have moved
    }
    model.spawned += 1
    return car_id
