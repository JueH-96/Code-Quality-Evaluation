# YOUR CODE HERE
def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return "No"
    
    # Check columns
    for col in range(9):
        if sorted(grid[row][col] for row in range(9)) != list(range(1, 10)):
            return "No"
    
    # Check 3x3 subgrids
    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            subgrid = [grid[row][col] for row in range(start_row, start_row + 3) for col in range(start_col, start_col + 3)]
            if sorted(subgrid) != list(range(1, 10)):
                return "No"
    
    return "Yes"

import sys
input = sys.stdin.read().split()
grid = [list(map(int, input[i*9:(i+1)*9])) for i in range(9)]

print(is_valid_sudoku(grid))