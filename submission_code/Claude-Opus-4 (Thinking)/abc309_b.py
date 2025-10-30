# YOUR CODE HERE
N = int(input())
grid = []
for i in range(N):
    grid.append(list(input().strip()))

# Get outer squares in clockwise order
outer = []

# Top row (left to right)
for j in range(N):
    outer.append((0, j))

# Right column (top to bottom, excluding top-right corner)
for i in range(1, N):
    outer.append((i, N-1))

# Bottom row (right to left, excluding bottom-right corner)
for j in range(N-2, -1, -1):
    outer.append((N-1, j))

# Left column (bottom to top, excluding corners)
for i in range(N-2, 0, -1):
    outer.append((i, 0))

# Get values from these positions
values = [grid[i][j] for i, j in outer]

# Shift values clockwise by one position
shifted = [values[-1]] + values[:-1]

# Update grid with shifted values
for idx, (i, j) in enumerate(outer):
    grid[i][j] = shifted[idx]

# Print result
for row in grid:
    print(''.join(row))