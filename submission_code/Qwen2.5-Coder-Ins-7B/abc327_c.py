# YOUR CODE HERE
import sys

def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False
    
    # Check columns
    for col in range(9):
        if sorted([grid[row][col] for row in range(9)]) != list(range(1, 10)):
            return False
    
    # Check 3x3 subgrids
    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            subgrid = []
            for row in range(3):
                for col in range(3):
                    subgrid.append(grid[start_row + row][start_col + col])
            if sorted(subgrid) != list(range(1, 10)):
                return False
    
    return True

# Read input
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check if the grid is valid
if is_valid_sudoku(grid):
    print("Yes")
else:
    print("No")