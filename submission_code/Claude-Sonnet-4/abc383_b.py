# YOUR CODE HERE
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def count_humidified(grid, H, W, D, hum1, hum2):
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':  # floor cell
                dist1 = manhattan_distance((i, j), hum1)
                dist2 = manhattan_distance((i, j), hum2)
                if dist1 <= D or dist2 <= D:
                    count += 1
    return count

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

# Try all pairs of floor cells
max_humidified = 0
n = len(floor_cells)

for i in range(n):
    for j in range(i + 1, n):
        hum1 = floor_cells[i]
        hum2 = floor_cells[j]
        humidified = count_humidified(grid, H, W, D, hum1, hum2)
        max_humidified = max(max_humidified, humidified)

print(max_humidified)