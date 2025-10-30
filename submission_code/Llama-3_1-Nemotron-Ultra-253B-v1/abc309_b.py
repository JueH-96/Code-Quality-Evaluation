n = int(input())
grid = [list(map(int, list(input().strip()))) for _ in range(n)]

coords = []

# Top row (i=0)
for j in range(n):
    coords.append((0, j))

# Right column (j = n-1), from i=1 to n-1
for i in range(1, n):
    coords.append((i, n-1))

# Bottom row (i=n-1), from j=n-2 downto 0
for j in range(n-2, -1, -1):
    coords.append((n-1, j))

# Left column (j=0), from i=n-2 downto 1
for i in range(n-2, 0, -1):
    coords.append((i, 0))

# Collect perimeter values
perimeter = [grid[i][j] for i, j in coords]

# Rotate the perimeter values by one (clockwise)
rotated = [perimeter[-1]] + perimeter[:-1]

# Update the grid with rotated values
for idx, (i, j) in enumerate(coords):
    grid[i][j] = rotated[idx]

# Print the result
for row in grid:
    print(''.join(map(str, row)))