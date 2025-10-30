# YOUR CODE HERE
N = int(input())
grid = []
for i in range(N):
    row = input().strip()
    grid.append([int(c) for c in row])

# Collect outer squares in clockwise order
outer_positions = []
outer_values = []

# Top row (left to right)
for j in range(N):
    outer_positions.append((0, j))
    outer_values.append(grid[0][j])

# Right column (top to bottom, excluding top-right corner)
for i in range(1, N):
    outer_positions.append((i, N-1))
    outer_values.append(grid[i][N-1])

# Bottom row (right to left, excluding bottom-right corner)
if N > 1:
    for j in range(N-2, -1, -1):
        outer_positions.append((N-1, j))
        outer_values.append(grid[N-1][j])

# Left column (bottom to top, excluding bottom-left and top-left corners)
if N > 2:
    for i in range(N-2, 0, -1):
        outer_positions.append((i, 0))
        outer_values.append(grid[i][0])

# Shift clockwise by one position
# The value at position i goes to position (i+1) % len(outer_positions)
shifted_values = [0] * len(outer_values)
for i in range(len(outer_values)):
    shifted_values[(i + 1) % len(outer_values)] = outer_values[i]

# Update the grid with shifted values
for i, (row, col) in enumerate(outer_positions):
    grid[row][col] = shifted_values[i]

# Print the result
for i in range(N):
    print(''.join(map(str, grid[i])))