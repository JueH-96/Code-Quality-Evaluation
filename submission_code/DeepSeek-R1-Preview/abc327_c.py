# Read the grid
grid = [list(map(int, input().split())) for _ in range(9)]

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
for i_start in range(0, 9, 3):
    for j_start in range(0, 9, 3):
        subgrid = []
        for x in range(i_start, i_start + 3):
            for y in range(j_start, j_start + 3):
                subgrid.append(grid[x][y])
        if len(set(subgrid)) != 9:
            print("No")
            exit()

# If all checks passed
print("Yes")