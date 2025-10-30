n = int(input())
grid = []
for _ in range(n):
    row = input().strip()
    grid.append([int(c) for c in row])

# Extract outer squares in clockwise order
outer_values = []
outer_positions = []

# Top row
for j in range(n):
    outer_values.append(grid[0][j])
    outer_positions.append((0, j))

# Right column (excluding top corner)
for i in range(1, n):
    outer_values.append(grid[i][n-1])
    outer_positions.append((i, n-1))

# Bottom row (excluding right corner)
for j in range(n-2, -1, -1):
    outer_values.append(grid[n-1][j])
    outer_positions.append((n-1, j))

# Left column (excluding top and bottom corners)
for i in range(n-2, 0, -1):
    outer_values.append(grid[i][0])
    outer_positions.append((i, 0))

# Shift values clockwise by one position
shifted_values = [outer_values[-1]] + outer_values[:-1]

# Put shifted values back
for i, (row, col) in enumerate(outer_positions):
    grid[row][col] = shifted_values[i]

# Print the result
for row in grid:
    print(''.join(map(str, row)))