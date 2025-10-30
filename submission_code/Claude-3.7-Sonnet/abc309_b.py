# Read the size of the grid
N = int(input())

# Read the grid
grid = []
for _ in range(N):
    row = input().strip()
    grid_row = [int(cell) for cell in row]
    grid.append(grid_row)

# Create a new grid (deep copy)
new_grid = [row[:] for row in grid]

# Define the perimeter in clockwise order
perimeter = []
# Top row
for j in range(N):
    perimeter.append((0, j))
# Right column (excluding the top corner)
for i in range(1, N):
    perimeter.append((i, N-1))
# Bottom row (excluding the right corner)
for j in range(N-2, -1, -1):
    perimeter.append((N-1, j))
# Left column (excluding the bottom corner and top corner)
for i in range(N-2, 0, -1):
    perimeter.append((i, 0))

# For each position on the perimeter, find its pre-image under the clockwise shift
for idx, (i, j) in enumerate(perimeter):
    prev_idx = (idx - 1) % len(perimeter)
    pi, pj = perimeter[prev_idx]
    new_grid[i][j] = grid[pi][pj]

# Print the resulting grid
for row in new_grid:
    print(''.join(map(str, row)))