import sys

# Read input
data = sys.stdin.read().split()
N = int(data[0])
grid = [list(row) for row in data[1:N+1]]

# Get perimeter positions in clockwise order starting from (0,0) 0-based indices
positions = []
# Top row: left to right
for c in range(N):
    positions.append((0, c))
# Right column: top to bottom, starting from row 1
for r in range(1, N):
    positions.append((r, N-1))
# Bottom row: right to left, from col N-2 to 0
for c in range(N-2, -1, -1):
    positions.append((N-1, c))
# Left column: bottom to top, from row N-2 to 1
for r in range(N-2, 0, -1):
    positions.append((r, 0))

# Extract the values in this order
perim_values = [grid[r][c] for r, c in positions]

# Perform a right rotation on the values (clockwise shift)
new_perim_values = [perim_values[-1]] + perim_values[:-1]

# Assign the new values back to the grid
for i, (r, c) in enumerate(positions):
    grid[r][c] = new_perim_values[i]

# The inner cells remain unchanged

# Output the grid
for row in grid:
    print(''.join(row))