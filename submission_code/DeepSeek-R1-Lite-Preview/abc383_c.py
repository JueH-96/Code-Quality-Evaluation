from collections import deque

# Read input values
H, W, D = map(int, input().split())

# Read the grid
grid = [input().strip() for _ in range(H)]

# Find all humidifier positions
humidifiers = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == 'H']

# Define a large number for unvisited cells
INF = H * W + 1

# Initialize distance matrix
distance = [[INF for _ in range(W)] for _ in range(H)]

# Set distance to 0 for all humidifiers
for i, j in humidifiers:
    distance[i][j] = 0

# Directions for movement: up, down, left, right
dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Initialize queue with all humidifiers
q = deque(humidifiers)

# Perform BFS
while q:
    i, j = q.popleft()
    for di, dj in dirs:
        ni, nj = i + di, j + dj
        # Check bounds, avoid walls, update distance if better and within D moves
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and distance[ni][nj] > distance[i][j] + 1 and distance[i][j] + 1 <= D:
            distance[ni][nj] = distance[i][j] + 1
            q.append((ni, nj))

# Count humidified floor cells
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] in '.H' and distance[i][j] <= D:
            count += 1

# Output the result
print(count)