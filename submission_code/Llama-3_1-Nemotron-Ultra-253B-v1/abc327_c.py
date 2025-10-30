# Read the grid from input
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
    column = [grid[i][j] for i in range(9)]
    if len(set(column)) != 9:
        print("No")
        exit()

# Check each 3x3 subgrid
for i in range(3):
    for j in range(3):
        subgrid = []
        for x in range(3*i, 3*i+3):
            for y in range(3*j, 3*j+3):
                subgrid.append(grid[x][y])
        if len(set(subgrid)) != 9:
            print("No")
            exit()

# All checks passed
print("Yes")