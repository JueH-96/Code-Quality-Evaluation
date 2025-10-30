# Read the 9x9 grid
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Check each row
for row in grid:
    if len(set(row)) != 9:
        print("No")
        exit()

# Check each column
for j in range(9):
    column = []
    for i in range(9):
        column.append(grid[i][j])
    if len(set(column)) != 9:
        print("No")
        exit()

# Check each 3x3 subgrid
for i in range(0, 9, 3):
    for j in range(0, 9, 3):
        subgrid = []
        for x in range(i, i + 3):
            for y in range(j, j + 3):
                subgrid.append(grid[x][y])
        if len(set(subgrid)) != 9:
            print("No")
            exit()

# If all checks passed
print("Yes")