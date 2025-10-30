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

# Function to get humidified floor cells from a humidifier at (hi, hj)
def get_humidified(hi, hj):
    humidified = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and abs(i - hi) + abs(j - hj) <= D:
                humidified.add((i, j))
    return humidified

max_humidified = 0

# Try all pairs of floor cells
for i in range(len(floor_cells)):
    for j in range(i + 1, len(floor_cells)):
        h1 = floor_cells[i]
        h2 = floor_cells[j]
        
        humidified1 = get_humidified(h1[0], h1[1])
        humidified2 = get_humidified(h2[0], h2[1])
        
        total_humidified = humidified1.union(humidified2)
        max_humidified = max(max_humidified, len(total_humidified))

print(max_humidified)