# YOUR CODE HERE
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

# Function to calculate Manhattan distance
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Try all pairs of floor cells
max_humidified = 0
n = len(floor_cells)

for i in range(n):
    for j in range(i + 1, n):
        # Place humidifiers at floor_cells[i] and floor_cells[j]
        humidifier1 = floor_cells[i]
        humidifier2 = floor_cells[j]
        
        # Count humidified floor cells
        humidified_count = 0
        for floor_cell in floor_cells:
            # Check if this floor cell is humidified
            dist1 = manhattan_distance(floor_cell, humidifier1)
            dist2 = manhattan_distance(floor_cell, humidifier2)
            
            if dist1 <= D or dist2 <= D:
                humidified_count += 1
        
        max_humidified = max(max_humidified, humidified_count)

print(max_humidified)