"""Names and values shared by every part of the intersection model."""

EMPTY = 0  # lane cell with no car in it (car IDs start at 1)

APPROACHES = ("N", "E", "S", "W")          # the four roads leading into the intersection
NORTH_SOUTH = ("N", "S")
EAST_WEST = ("E", "W")

MOVES = ("left", "straight", "right")      # one turning pocket per move on every approach
