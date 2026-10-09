# Adaptive versus Fixed-Time Traffic Signal Control at a Four-Way Intersection

CITS4403 Computational Modelling project — Kaiyuan (Kevin) Tan and Josh Sporne.

## Overview

We model a four-way intersection with left-hand traffic as a cellular automaton
whose vehicles are agents, and compare a self-organising adaptive signal
controller with two fixed-time traffic signals. We experiment by varying demand,
directional imbalance, turning and driver reaction delay one at a time and
several together.

**Research question:** how does the operational efficiency of adaptive traffic
control compare with fixed-time control across varying levels of asymmetric
traffic demand, and to what extent does driver reaction delay change the
average wait time?

### The model in brief

- **Road layout.** Four approaches (N, E, S, W). Each is a 30-cell shared lane
  that splits into three 10-cell pockets (left, straight, right) ending at a
  stop line. A cell is about 7.5 m and a time step about 1 s. Vehicles drive on
  the left, as in Australia.
- **Vehicles.** Each vehicle occupies one cell and is an agent with its own
  record (approach, movement, arrival step, time spent stopped). All vehicles
  move at once (parallel update): a vehicle advances one cell if the cell ahead
  was empty at the start of the step, so a queue discharges one vehicle every
  two steps. A full pocket blocks the shared lane behind it (spillback).
- **Arrivals.** Each step a vehicle arrives on each approach at random,
  controlled by the arrival rate, the north–south probability and the left- and
  right-turn probabilities. A vehicle that finds its road full is turned away
  (or, optionally, waits in a queue off the map).
- **Reaction delay.** A vehicle starting from standstill stalls for a step with
  a given probability (a slow-to-start rule).
- **Signals.** Four protected phases (north–south straight and left,
  north–south right, east–west straight and left, east–west right), each green
  followed by a 3-step all-red. Three controllers:
  - **Fixed-time (untuned):** 20 steps per phase in a fixed order.
  - **Fixed-time (Webster):** the same order with green times designed by
    Webster's method from the average demand.
  - **Adaptive:** after a minimum green of 5 steps, the light switches to the
    phase whose queued vehicles have waited longest in total.
- **Measurement.** Each run starts empty: 500 warm-up steps, a 2,000-step
  measured window and 300 cool-down steps. The main measure is the mean number
  of steps a vehicle spends stopped; throughput, vehicles turned away, light
  changes and spillback are also recorded. For a given seed, all three
  controllers see exactly the same arriving vehicles.

## Repository structure

```
src/                      the model
  constants.py              approach and movement names
  intersection.py           Intersection class: lanes, settings, one step of the model
  spawning.py               vehicle arrivals
  movement.py               vehicle movement, stop lines, crossing the junction
  signals.py                signal phases and the three controllers
utils/                    helpers and experiment runners
  metrics.py                run_and_measure(): run one simulation and summarise it
  animation.py              draw the intersection and animate the controllers
  results_sweeps.py         baseline, one-setting sweeps, scenarios, pocket lengths
  experiments.py            main experiment (every combination) and one-factor experiments
  extensions.py             extension experiments
data/                     saved results of every experiment (CSV / NPZ)
notebooks/Results/        analysis notebooks
  Results.ipynb             animation of the model, then the main results
  Extensions.ipynb          extension experiments
  Analysis.ipynb            one-factor experiments (untuned vs adaptive)
figures/                  figures saved by the notebooks
requirements.txt          dependencies
```

## Setup

Requires **Python 3.11 or later** (needed by matplotlib 3.11; developed and tested with Python 3.12).

```bash
git clone https://github.com/phattydumpling/CITS4403-Project.git
cd CITS4403-Project

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `matplotlib` (3.11 or later) and `notebook`
(Jupyter).

## Viewing the results

All experiment results are already saved in `data/`, so the notebooks can be
run straight away without re-running any simulations:

```bash
jupyter notebook notebooks/Results/
```

| Notebook | Contents |
|---|---|
| `Results.ipynb` | Start here. An animation of the three controllers side by side on the same traffic, then the baseline comparison, each setting varied on its own, several settings at once, the effect of each setting, signals and queues over time, and pocket lengths. |
| `Extensions.ipynb` | Tipping point, self-organised signal timing, wait distributions, limited sensing, drivers who differ, a demand surge, and overload when blocked vehicles queue instead of being turned away. |
| `Analysis.ipynb` | One-factor experiments comparing the untuned and adaptive controllers. |

Run the notebooks from their own folder (Jupyter's default), as they find
`data/` and `figures/` relative to it. Building the animation in
`Results.ipynb` takes about 15 seconds; the settings at the top of that cell can
be changed to animate any other traffic.

## Running a single simulation

From the project folder:

```python
from src.intersection import Intersection
from utils.metrics import run_and_measure

model = Intersection(controller="adaptive",   # "fixed", "webster" or "adaptive"
                     demand=0.15,              # arrival rate per approach per step
                     ns_probability=0.7,       # chance an arrival comes from the north or south
                     left_probability=0.2,     # chance a vehicle turns left
                     right_probability=0.2,    # chance a vehicle turns right
                     reaction_delay=0.1,       # chance a vehicle stalls when starting from standstill
                     seed=0)
results = run_and_measure(model)               # warm-up, measured window and cool-down
print(results["mean_wait"], results["throughput"], results["blocked_arrivals"])
```

Other settings of `Intersection`, with their defaults: `green_time=20`,
`min_green=5`, `max_green=40`, `all_red=3`, `feeder_length=30`,
`pocket_length=10`, `turn_pocket_length=None` (same as `pocket_length`),
`crossing_time=3`, `queue_blocked_arrivals=False` (True: blocked vehicles wait
in a queue instead of being turned away), `detector_range=None` (cells of each
pocket the adaptive controller can see; None = all) and
`slow_driver_probability=None` (concentrate the reaction delay in a share of
drivers). `model.step()` advances the model by one step and `model.run(n)` by
`n` steps.

## Re-running the experiments

Each command runs from the project folder, uses every CPU core, and overwrites
the matching files in `data/`. Times are approximate, on a 16-core machine.

| Command | Experiment | Output | Time |
|---|---|---|---|
| `python -m utils.results_sweeps` | Baseline (30 seeds), each setting varied on its own, seven scenarios, pocket lengths | `fine_sweeps.csv`, `scenarios.csv`, `fine_pockets.csv` | ~5 min |
| `python -m utils.results_sweeps --queued` | Baseline and one-setting sweeps with blocked vehicles queuing | `fine_sweeps_queued.csv` | ~4 min |
| `python -m utils.experiments` | Main experiment: every combination of arrival rate, north–south probability, right-turn probability and reaction delay (3,240 runs) | `sweep_results.csv` | ~5 min |
| `python -m utils.experiments --one-factor` | One-factor experiments for `Analysis.ipynb` | `notebooks/Results/sweep_results.csv` | ~1 min |
| `python -m utils.extensions` | All extension experiments | `ext_*.csv`, `ext_*.npz` | ~16 min |

The extensions can also be run one at a time with `--fine`, `--sensing`,
`--drivers`, `--surge` or `--overload`, for example
`python -m utils.extensions --surge`.

After re-running an experiment, re-run the matching notebook to update its
figures. Every run is seeded, so re-running with the same package versions reproduces the saved results.
