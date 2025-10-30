# YOUR CODE HERE
N = int(input())
grid = []
for i in range(N):
    grid.append(list(input().strip()))

# Create a copy of the grid for the result
result = [row[:] for row in grid]

# Extract outer ring elements in clockwise order
outer_elements = []

# Top row (left to right)
for j in range(N):
    outer_elements.append(grid[0][j])

# Right column (top to bottom, excluding corners)
for i in range(1, N):
    outer_elements.append(grid[i][N-1])

# Bottom row (right to left, excluding corners)
if N > 1:
    for j in range(N-2, -1, -1):
        outer_elements.append(grid[N-1][j])

# Left column (bottom to top, excluding corners)
if N > 1:
    for i in range(N-2, 0, -1):
        outer_elements.append(grid[i][0])

# Shift elements by 1 position clockwise
shifted_elements = [outer_elements[-1]] + outer_elements[:-1]

# Place shifted elements back
idx = 0

# Top row
for j in range(N):
    result[0][j] = shifted_elements[idx]
    idx += 1

# Right column
for i in range(1, N):
    result[i][N-1] = shifted_elements[idx]
    idx += 1

# Bottom row
if N > 1:
    for j in range(N-2, -1, -1):
        result[N-1][j] = shifted_elements[idx]
        idx += 1

# Left column
if N > 1:
    for i in range(N-2, 0, -1):
        result[i][0] = shifted_elements[idx]
        idx += 1

# Print result
for row in result:
    print(''.join(row))