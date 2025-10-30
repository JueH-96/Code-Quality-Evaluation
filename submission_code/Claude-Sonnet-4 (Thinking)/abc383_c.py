from collections import deque

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

# Multi-source BFS
queue = deque()
visited = set()

# Start from all humidifiers
for i, j in humidifiers:
    queue.append((i, j, 0))
    visited.add((i, j))

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

while queue:
    row, col, dist = queue.popleft()
    
    if dist < D:
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and grid[nr][nc] != '#':
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

print(len(visited))