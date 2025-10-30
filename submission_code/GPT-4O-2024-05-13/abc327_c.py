# YOUR CODE HERE
def is_valid_sudoku(grid):
    def is_valid_group(group):
        return sorted(group) == list(range(1, 10))
    
    # Check rows
    for row in grid:
        if not is_valid_group(row):
            return False
    
    # Check columns
    for col in range(9):
        if not is_valid_group([grid[row][col] for row in range(9)]):
            return False
    
    # Check 3x3 subgrids
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            subgrid = []
            for row in range(3):
                for col in range(3):
                    subgrid.append(grid[box_row + row][box_col + col])
            if not is_valid_group(subgrid):
                return False
    
    return True

import sys
input = sys.stdin.read
data = input().strip().split()
grid = [list(map(int, data[i*9:(i+1)*9])) for i in range(9)]

if is_valid_sudoku(grid):
    print("Yes")
else:
    print("No")