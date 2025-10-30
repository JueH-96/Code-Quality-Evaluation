# Read the grid
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
for col in range(9):
    column = [grid[row][col] for row in range(9)]
    if len(set(column)) != 9:
        print("No")
        exit()

# Check each 3x3 subgrid
for i in range(3):
    for j in range(3):
        sub = []
        for x in range(3*i, 3*i + 3):
            for y in range(3*j, 3*j + 3):
                sub.append(grid[x][y])
        if len(set(sub)) != 9:
            print("No")
            exit()

# All conditions satisfied
print("Yes")