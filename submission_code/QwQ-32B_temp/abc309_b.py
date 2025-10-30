n = int(input())
grid = []
for _ in range(n):
    line = input().strip()
    grid.append(list(line))

if n < 2:
    for row in grid:
        print(''.join(row))
    exit()

perimeter = []

# Collect top row
for j in range(n):
    perimeter.append(grid[0][j])

# Collect right column (excluding top)
for i in range(1, n):
    perimeter.append(grid[i][n-1])

# Collect bottom row (excluding last element)
for j in range(n-2, -1, -1):
    perimeter.append(grid[n-1][j])

# Collect left column (excluding top and bottom)
for i in range(n-2, 0, -1):
    perimeter.append(grid[i][0])

# Rotate the perimeter list
if len(perimeter) == 0:
    rotated = []
else:
    rotated = [perimeter[-1]] + perimeter[:-1]

# Assign back to the grid
idx = 0

# Top row
for j in range(n):
    grid[0][j] = rotated[idx]
    idx += 1

# Right column
for i in range(1, n):
    grid[i][n-1] = rotated[idx]
    idx += 1

# Bottom row
for j in range(n-2, -1, -1):
    grid[n-1][j] = rotated[idx]
    idx += 1

# Left column
for i in range(n-2, 0, -1):
    grid[i][0] = rotated[idx]
    idx += 1

# Print the result
for row in grid:
    print(''.join(row))