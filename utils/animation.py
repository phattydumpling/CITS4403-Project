"""Drawing the intersection and animating several controllers side by side.

The picture is built for the north approach and rotated for the others. On the
north arm, cars drive south on the east half of the road (traffic keeps left),
in three pockets that end at the stop line y = 3: left turns next to the kerb,
then straight, then right turns next to the centre line. The shared feeder
lane continues north from the end of the pockets. The west half of each arm
carries departing traffic, which the model does not simulate, so it is drawn
lighter. The junction is the 6 x 6 square in the middle.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.constants import APPROACHES, MOVES
from src.intersection import Intersection

HALF = 3                                        # the junction spans -3..3 in x and y
LANE_X = {"left": 2.5, "straight": 1.5, "right": 0.5}   # lane centres on the north arm
QUARTER_TURNS = {"N": 0, "E": 1, "S": 2, "W": 3}         # clockwise turns from the north arm
EXIT_TURNS = {"left": 1, "straight": 2, "right": 3}      # from the arm a car arrives on to the one it leaves by
MOVE_COLOURS = {"left": "tab:blue", "straight": "0.15", "right": "tab:orange"}
LABELS = {"fixed": "Fixed-time (untuned)", "webster": "Fixed-time (Webster)", "adaptive": "Adaptive"}


def rotate(x, y, quarter_turns):
    """Rotate a point of the north arm clockwise about the centre."""
    for _ in range(quarter_turns % 4):
        x, y = y, -x
    return x, y


def arm_rectangle(ax, x0, x1, y0, y1, quarter_turns, **style):
    """Draw the rectangle x0..x1, y0..y1 of the north arm, rotated onto another arm."""
    (ax0, ay0), (ax1, ay1) = rotate(x0, y0, quarter_turns), rotate(x1, y1, quarter_turns)
    ax.add_patch(Rectangle((min(ax0, ax1), min(ay0, ay1)), abs(ax1 - ax0), abs(ay1 - ay0), **style))


def car_positions(model):
    """Position and colour of every car: in the feeder lanes and pockets, one
    per cell; inside the junction, part of the way along a straight line from
    its stop line to the lane it leaves by."""
    xy, colours = [], []
    for approach in APPROACHES:
        turns = QUARTER_TURNS[approach]
        straight_length = len(model.pockets[(approach, "straight")])
        for move in MOVES:
            for k, car_id in enumerate(model.pockets[(approach, move)]):
                if car_id:
                    xy.append(rotate(LANE_X[move], HALF + 0.5 + k, turns))
                    colours.append(MOVE_COLOURS[move])
        for i, car_id in enumerate(model.feeders[approach]):
            if car_id:
                xy.append(rotate(LANE_X["straight"], HALF + 0.5 + straight_length + i, turns))
                colours.append(MOVE_COLOURS[model.agents[car_id]["move"]])
    for car_id, steps_left in model.crossing:
        car = model.agents[car_id]
        turns = QUARTER_TURNS[car["approach"]]
        start = np.array(rotate(LANE_X[car["move"]], HALF, turns))
        end = np.array(rotate(-LANE_X[car["move"]], HALF, turns + EXIT_TURNS[car["move"]]))
        progress = (model.crossing_time - steps_left + 0.5) / model.crossing_time
        xy.append(tuple(start + progress * (end - start)))
        colours.append(MOVE_COLOURS[car["move"]])
    return np.array(xy).reshape(-1, 2), colours


class IntersectionView:
    """The drawing of one model on one axes, updated with update()."""

    def __init__(self, ax, model, title, extent=None):
        self.ax, self.model = ax, model
        pocket_length = max(len(p) for p in model.pockets.values())
        straight_length = len(model.pockets[("N", "straight")])
        road_end = HALF + straight_length + len(model.feeders["N"])
        extent = extent or road_end + 1
        road, unused = dict(color="0.72", zorder=0), dict(color="0.88", zorder=0)
        ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, color="0.72", zorder=0))
        for turns in range(4):
            arm_rectangle(ax, 0, HALF, HALF, HALF + pocket_length, turns, **road)       # pockets
            arm_rectangle(ax, 1, 2, HALF, road_end, turns, **road)                     # feeder lane
            arm_rectangle(ax, -HALF, 0, HALF, road_end, turns, **unused)               # departing side
        # white lines between the pockets and along the centre of the road
        markings = [[rotate(x, HALF, t), rotate(x, HALF + pocket_length, t)] for x in (1, 2) for t in range(4)]
        markings += [[rotate(0, HALF, t), rotate(0, road_end, t)] for t in range(4)]
        ax.add_collection(LineCollection(markings, colors="white", linewidths=0.8, zorder=1))
        # one bar at the end of each pocket, just inside the junction, coloured by that movement's light
        self.keys = [(a, m) for a in APPROACHES for m in MOVES]
        bars = [[rotate(LANE_X[m] - 0.45, HALF - 0.3, QUARTER_TURNS[a]),
                 rotate(LANE_X[m] + 0.45, HALF - 0.3, QUARTER_TURNS[a])] for a, m in self.keys]
        self.lights = LineCollection(bars, linewidths=3, zorder=2)
        ax.add_collection(self.lights)
        self.cars = ax.scatter([], [], s=12, marker="s", linewidths=0, zorder=3)
        self.info = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, va="bottom",
                            family="monospace")
        ax.set_title(title)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.axis("off")
        self.extent = extent
        self.update()

    def fit_cars(self):
        """Size the car markers to about one cell and the light bars to about
        half a cell, after the figure is laid out."""
        fig = self.ax.figure
        points_per_cell = self.ax.get_window_extent().width * 72 / fig.dpi / (2 * self.extent)
        self.cars.set_sizes([(0.85 * points_per_cell) ** 2])
        self.lights.set_linewidths(max(2, 0.6 * points_per_cell))

    def update(self):
        model = self.model
        xy, colours = car_positions(model)
        self.cars.set_offsets(xy)
        self.cars.set_facecolors(colours)
        self.lights.set_colors(["tab:green" if key in model.green else "tab:red" for key in self.keys])
        served = [car["wait_time"] for car in model.finished.values()]
        waiting = sum(1 for car in model.agents.values() if car["stopped"])
        self.info.set_text(f"step {model.time}\ncars stopped now: {waiting}\n"
                           f"cars served: {len(served)}\nmean wait so far: "
                           f"{np.mean(served) if served else 0:.1f} steps")


def legend_handles():
    """Legend entries for the car colours and the lights."""
    cars = [Line2D([], [], marker="s", linestyle="", color=c, label=f"car turning {m}" if m != "straight"
                   else "car going straight") for m, c in MOVE_COLOURS.items()]
    lights = [Line2D([], [], color="tab:green", linewidth=3, label="green light"),
              Line2D([], [], color="tab:red", linewidth=3, label="red light")]
    return cars + lights


def animate_controllers(settings, controllers=("fixed", "webster", "adaptive"), warmup=200, steps=200,
                        interval=100, extent=None, dpi=80):
    """Run one model per controller on the same settings (and so the same
    arriving cars), and return an animation of `steps` steps after `warmup`
    steps, drawn side by side, together with the models. `extent` zooms in to
    that many cells from the centre (None shows the whole road)."""
    models = [Intersection(controller=c, **settings) for c in controllers]
    for model in models:
        model.run(warmup)
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 6), dpi=dpi)
    views = [IntersectionView(ax, m, LABELS.get(c, c), extent) for ax, m, c in zip(axes, models, controllers)]
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for view in views:
        view.fit_cars()

    def draw(frame):
        for model, view in zip(models, views):
            while model.time < warmup + frame:     # safe if a frame is drawn twice
                model.step()
            view.update()

    animation = FuncAnimation(fig, draw, frames=steps, interval=interval)
    plt.close(fig)          # show the animation, not an extra still picture
    return animation, models
