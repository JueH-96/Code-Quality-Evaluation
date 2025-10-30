# YOUR CODE HERE
from typing import Tuple

def solve(H: int, W: int, N: int) -> None:
    # Initialize the grid with all cells as white
    grid = [['.' for _ in range(W)] for _ in range(H)]

    # Initialize Takahashi's position and direction
    x, y = 0, 0
    dx, dy = 0, 1  # facing upwards

    for _ in range(N):
        # Get the current cell color
        cell_color = grid[x][y]

        # Perform the operation
        if cell_color == '.':
            grid[x][y] = '#'
            dx, dy = -dy, dx  # rotate 90 degrees clockwise
        else:
            grid[x][y] = '.'
            dx, dy = dy, -dx  # rotate 90 degrees counterclockwise

        # Move forward one cell
        x = (x + dx) % H
        y = (y + dy) % W

    # Print the final grid
    for row in grid:
        print(''.join(row))

# Read the input
H, W, N = map(int, input().split())
solve(H, W, N)