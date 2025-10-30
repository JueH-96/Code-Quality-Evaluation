# Read the 9x9 grid
grid = [list(map(int, input().split())) for _ in range(9)]

valid = True

# Check each row
for row in grid:
    if len(set(row)) != 9:
        valid = False
        break

# Check each column if rows are valid
if valid:
    for j in range(9):
        column = [grid[i][j] for i in range(9)]
        if len(set(column)) != 9:
            valid = False
            break

# Check each 3x3 subgrid if previous checks passed
if valid:
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrid = []
            for x in range(i, i+3):
                for y in range(j, j+3):
                    subgrid.append(grid[x][y])
            if len(set(subgrid)) != 9:
                valid = False
                break
        if not valid:
            break

# Output the result
print("Yes" if valid else "No")