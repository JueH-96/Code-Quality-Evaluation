import sys

grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

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
for br in (0, 3, 6):
    for bc in (0, 3, 6):
        subgrid = []
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                subgrid.append(grid[i][j])
        if len(set(subgrid)) != 9:
            print("No")
            sys.exit()

print("Yes")