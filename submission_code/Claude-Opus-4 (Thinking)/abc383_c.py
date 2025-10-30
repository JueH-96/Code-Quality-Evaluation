from collections import deque

H, W, D = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all humidifiers
humidifiers = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            humidifiers.append((i, j))

# BFS from all humidifiers
visited = [[False] * W for _ in range(H)]
queue = deque()

# Initialize queue with all humidifiers
for hi, hj in humidifiers:
    queue.append((hi, hj, 0))
    visited[hi][hj] = True

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS
while queue:
    i, j, dist = queue.popleft()
    
    if dist < D:
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and grid[ni][nj] != '#':
                visited[ni][nj] = True
                queue.append((ni, nj, dist + 1))

# Count humidified floor cells
count = 0
for i in range(H):
    for j in range(W):
        if visited[i][j] and grid[i][j] != '#':
            count += 1

print(count)