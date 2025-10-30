n = int(input())
grid = []
for _ in range(n):
    line = input().strip()
    grid.append([int(c) for c in line])

# Generate the coordinates of the outer layer in order
coords = []
# Top row (left to right)
for j in range(n):
    coords.append((0, j))
# Right column (top to bottom, excluding first row)
for i in range(1, n):
    coords.append((i, n-1))
# Bottom row (right to left, excluding last column)
for j in range(n-2, -1, -1):
    coords.append((n-1, j))
# Left column (bottom to top, excluding first and last rows)
for i in range(n-2, 0, -1):
    coords.append((i, 0))

# Collect the values in the current order
values = [grid[i][j] for (i, j) in coords]

# Rotate the values
if len(values) > 0:
    rotated = [values[-1]] + values[:-1]
else:
    rotated = []

# Assign rotated values back
for k in range(len(coords)):
    i, j = coords[k]
    grid[i][j] = rotated[k]

# Print the result
for row in grid:
    print(''.join(str(x) for x in row))