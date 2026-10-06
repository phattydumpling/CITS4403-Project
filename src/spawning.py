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

    For each approach, a car arrives with arrival_probability(). It chooses
    its move (left / straight / right) from the turning shares and enters the
    outer end of the feeder lane, but only if that cell is empty; otherwise
    the arrival is lost and counted in `model.blocked_arrivals`.

    Both random numbers are drawn from model.arrival_rng on every step for
    every approach, whatever the state of the road, so two runs with the same
    seed see exactly the same arrivals even if their signals differ.
    """
    for approach in APPROACHES:
        arrives = model.arrival_rng.random() < arrival_probability(model, approach)
        move = MOVES[model.arrival_rng.choice(len(MOVES), p=model.move_shares)]
        if not arrives:
            continue
        feeder = model.feeders[approach]
        if feeder[-1] != EMPTY:
            model.blocked_arrivals += 1
            continue
        car_id = model.next_id
        model.next_id += 1
        feeder[-1] = car_id
        model.agents[car_id] = {
            "approach": approach,
            "move": move,
            "spawn_step": model.time,
            "wait_time": 0,              # steps spent not moving
            "stopped": False,            # did not move last step (arrives already moving)
            "reaction_delayed": False,   # stalled this step when it could have moved
        }
        model.spawned += 1
