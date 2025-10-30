import sys

# Read the grid
grid = [list(map(int, input().split())) for _ in range(9)]

# Check each row
for row in grid:
    if len(set(row)) != 9:
        print("No")
        sys.exit()

# Check each column
for col in range(9):
    column = [grid[row][col] for row in range(9)]
    if len(set(column)) != 9:
        print("No")
        sys.exit()

# Check each 3x3 subgrid
for start_row in [0, 3, 6]:
    for start_col in [0, 3, 6]:
        subgrid = []
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                subgrid.append(grid[i][j])
        if len(set(subgrid)) != 9:
            print("No")
            sys.exit()

# If all checks passed
print("Yes")