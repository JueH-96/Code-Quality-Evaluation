# YOUR CODE HERE
N = int(input())
grid = [list(input()) for _ in range(N)]

def get_outer_positions(N):
    positions = []
    # Top row
    for j in range(N):
        positions.append((0, j))
    # Right column (excluding corners)
    for i in range(1, N-1):
        positions.append((i, N - 1))
    # Bottom row (if N > 1), reversed
    if N > 1:
        for j in range(N - 1, -1, -1):
            positions.append((N - 1, j))
    # Left column (excluding corners), reversed
    for i in range(N - 2, 0, -1):
        positions.append((i, 0))
    return positions

positions = get_outer_positions(N)
# Extract values
outer_values = [grid[i][j] for i, j in positions]
# Rotate the list to the right by one
outer_values = [outer_values[-1]] + outer_values[:-1]
# Assign back the rotated values
for idx, (i, j) in enumerate(positions):
    grid[i][j] = outer_values[idx]
# Print the grid
for row in grid:
    print(''.join(row))