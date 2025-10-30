# YOUR CODE HERE
def is_valid_sudoku(grid):
    # Check rows
    for row in grid:
        if len(set(row)) != 9:
            return "No"
    
    # Check columns
    for col in range(9):
        if len(set(grid[row][col] for row in range(9))) != 9:
            return "No"
    
    # Check 3x3 subgrids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if len(set(subgrid)) != 9:
                return "No"
    
    return "Yes"

# Read the input grid
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check if the grid is a valid Sudoku
print(is_valid_sudoku(grid))