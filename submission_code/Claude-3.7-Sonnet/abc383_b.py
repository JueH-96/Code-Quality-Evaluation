import itertools

# Read input
H, W, D = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all floor cells
floor_cells = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            floor_cells.append((i, j))

# Try all possible combinations of two humidifiers
max_humidified = 0
for (i1, j1), (i2, j2) in itertools.combinations(floor_cells, 2):
    # Count how many floor cells are humidified with this configuration
    humidified = 0
    for i, j in floor_cells:
        # A cell is humidified if it's within distance D of either humidifier
        if abs(i - i1) + abs(j - j1) <= D or abs(i - i2) + abs(j - j2) <= D:
            humidified += 1
    
    # Update the maximum
    max_humidified = max(max_humidified, humidified)

# Print the result
print(max_humidified)