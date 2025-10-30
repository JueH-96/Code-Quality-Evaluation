# Read the grid
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check each row
for row in grid:
    if sorted(row) != list(range(1, 10)):
        print("No")
        exit()

# Check each column
for j in range(9):
    column = [grid[i][j] for i in range(9)]
    if sorted(column) != list(range(1, 10)):
        print("No")
        exit()

# Check each 3x3 subgrid
for group_row in range(3):
    for group_col in range(3):
        subgrid = []
        for i in range(3):
            for j in range(3):
                subgrid.append(grid[group_row * 3 + i][group_col * 3 + j])
        if sorted(subgrid) != list(range(1, 10)):
            print("No")
            exit()

# If all checks passed
print("Yes")