import sys

# Function to check if the grid is a valid Sudoku
def is_valid_sudoku(grid):
    # Check each row for uniqueness
    for row in grid:
        if len(set(row)) != 9:
            return False
    
    # Check each column for uniqueness
    for col in range(9):
        column_values = [grid[row][col] for row in range(9)]
        if len(set(column_values)) != 9:
            return False
    
    # Check each 3x3 subgrid for uniqueness
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid_values = []
            for x in range(i, i + 3):
                for y in range(j, j + 3):
                    subgrid_values.append(grid[x][y])
            if len(set(subgrid_values)) != 9:
                return False
    
    return True

# Read the grid from standard input
grid = []
for line in sys.stdin:
    row = [int(x) for x in line.split()]
    grid.append(row)

# Check if the grid is valid and print the result
if is_valid_sudoku(grid):
    print("Yes")
else:
    print("No")