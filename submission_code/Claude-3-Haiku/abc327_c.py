import sys

# Read the input grid
grid = []
for i in range(9):
    row = [int(x) for x in input().split()]
    grid.append(row)

# Check if each row contains 1-9 exactly once
for row in grid:
    if sorted(row) != list(range(1, 10)):
        print("No")
        sys.exit()

# Check if each column contains 1-9 exactly once
for j in range(9):
    col = [grid[i][j] for i in range(9)]
    if sorted(col) != list(range(1, 10)):
        print("No")
        sys.exit()

# Check if each 3x3 subgrid contains 1-9 exactly once
for i in range(0, 9, 3):
    for j in range(0, 9, 3):
        subgrid = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
        if sorted(subgrid) != list(range(1, 10)):
            print("No")
            sys.exit()

print("Yes")