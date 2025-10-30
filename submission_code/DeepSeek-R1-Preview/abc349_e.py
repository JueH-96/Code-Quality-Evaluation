import sys
from functools import lru_cache

# Read grid values and flatten into a single tuple of 9 elements
grid_values = []
for _ in range(3):
    row = list(map(int, sys.stdin.readline().split()))
    grid_values.extend(row)
grid_values = tuple(grid_values)

# Define all possible lines for three in a row
lines = [
    [0, 1, 2],    # Row 1
    [3, 4, 5],    # Row 2
    [6, 7, 8],    # Row 3
    [0, 3, 6],    # Column 1
    [1, 4, 7],    # Column 2
    [2, 5, 8],    # Column 3
    [0, 4, 8],    # Diagonal 1
    [2, 4, 6],    # Diagonal 2
]

def has_three(grid_state, color):
    """Check if the grid has three consecutive cells of the given color."""
    for line in lines:
        if all(grid_state[i] == color for i in line):
            return True
    return False

def calculate_sums(grid_state):
    """Calculate the sum of scores for R and B cells."""
    sum_r = 0
    sum_b = 0
    for idx in range(9):
        cell = grid_state[idx]
        val = grid_values[idx]
        if cell == 'R':
            sum_r += val
        elif cell == 'B':
            sum_b += val
    return sum_r, sum_b

@lru_cache(maxsize=None)
def can_win(grid_state, is_takahashi_turn):
    """
    Determine if the current player can force a win from the given grid state.
    grid_state is a tuple of 9 elements ('R', 'B', 'W').
    is_takahashi_turn is a boolean indicating if it's Takahashi's turn.
    Returns True if the current player can win.
    """
    current_color = 'R' if is_takahashi_turn else 'B'
    white_cells = [i for i, c in enumerate(grid_state) if c == 'W']
    
    for cell in white_cells:
        # Create new grid state after making the move
        new_grid = list(grid_state)
        new_grid[cell] = current_color
        new_grid = tuple(new_grid)
        
        # Check for three in a row
        if has_three(new_grid, current_color):
            return True
        
        # Check if grid is filled
        if 'W' not in new_grid:
            sum_r, sum_b = calculate_sums(new_grid)
            if current_color == 'R':
                return sum_r > sum_b
            else:
                return sum_b > sum_r
        
        # Recursively check if the opponent can't win from the new state
        opponent_result = can_win(new_grid, not is_takahashi_turn)
        if not opponent_result:
            return True
    
    # If no moves lead to a win, current player cannot force a win
    return False

# Initial state: all cells are white
initial_state = ('W',) * 9

# Determine if Takahashi can win from the initial state
if can_win(initial_state, True):
    print("Takahashi")
else:
    print("Aoki")