n = int(input())
grid = [input().strip() for _ in range(n)]

elements = []
# Top row
for j in range(n):
    elements.append(grid[0][j])
# Right column
for i in range(1, n):
    elements.append(grid[i][n-1])
# Bottom row
for j in range(n-2, -1, -1):
    elements.append(grid[n-1][j])
# Left column
for i in range(n-2, 0, -1):
    elements.append(grid[i][0])

# Shift the elements
shifted = elements[1:] + [elements[0]] if elements else []

# Assign back to grid
# Top row
for j in range(n):
    grid[0][j] = shifted[j]

# Right column
for i in range(1, n):
    idx = n + (i-1)
    grid[i][n-1] = shifted[idx]

# Bottom row
for j in range(n-2, -1, -1):
    idx = 2 * n - 1 + (n-2 - j)
    grid[n-1][j] = shifted[idx]

# Left column
for i in range(n-2, 0, -1):
    idx = 3 * n - 3 + (i-1)
    grid[i][0] = shifted[idx]

# Print the result
for row in grid:
    print(''.join(row))