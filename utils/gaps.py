"""Helpers for measuring the empty space around a cell."""
import numpy as np

from src.constants import EMPTY


def gap_ahead(grid, lane, x, limit):
    """Number of empty cells directly in front of (lane, x), counting at most `limit`.

    Cells past the end of the road count as empty, because vehicles leave there.
    Wrecks count as occupied, like any other vehicle.
    """
    cells = grid[lane, x + 1 : x + 1 + limit]
    occupied = np.flatnonzero(cells != EMPTY)
    return int(occupied[0]) if occupied.size else limit
