# YOUR CODE HERE
# Read the 9x9 grid
grid = []
for i in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False
    
    # Check columns
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        if sorted(column) != list(range(1, 10)):
            return False
    
    # Check 3x3 subgrids
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            subgrid = []
            for i in range(3):
                for j in range(3):
                    subgrid.append(grid[box_row + i][box_col + j])
            if sorted(subgrid) != list(range(1, 10)):
                return False
    
    return True

if is_valid_sudoku(grid):
    print("Yes")
else:
    print("No")