from collections import deque

# Read input
H, W, D = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all humidifier positions
humidifiers = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            humidifiers.append((i, j))

# BFS to find all humidified cells
visited = [[False] * W for _ in range(H)]
distance = [[float('inf')] * W for _ in range(H)]
queue = deque()

# Initialize queue with all humidifiers
for i, j in humidifiers:
    queue.append((i, j, 0))
    distance[i][j] = 0
    visited[i][j] = True

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS
while queue:
    x, y, dist = queue.popleft()
    
    # If we've reached maximum distance, don't explore further
    if dist >= D:
        continue
    
    # Check all 4 directions
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        # Check bounds
        if 0 <= nx < H and 0 <= ny < W:
            # Check if it's not a wall and not visited
            if grid[nx][ny] != '#' and not visited[nx][ny]:
                visited[nx][ny] = True
                distance[nx][ny] = dist + 1
                queue.append((nx, ny, dist + 1))

# Count humidified floor cells
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] != '#' and distance[i][j] <= D:
            count += 1

print(count)