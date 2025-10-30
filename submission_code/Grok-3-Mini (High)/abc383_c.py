import sys
from collections import deque

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
D = int(data[index])
index += 1
grid = []
for _ in range(H):
    row_str = data[index]
    grid.append(row_str)
    index += 1

# Initialize visited array
visited = [[False for _ in range(W)] for _ in range(H)]

# Initialize queue and count
queue = deque()
count = 0

# Add all humidifiers to queue with distance 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            queue.append((i, j, 0))
            visited[i][j] = True
            count += 1

# Directions for moving up, right, down, left
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

# Perform BFS
while queue:
    x, y, dist = queue.popleft()
    for k in range(4):
        nx = x + dx[k]
        ny = y + dy[k]
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and not visited[nx][ny] and dist + 1 <= D:
            queue.append((nx, ny, dist + 1))
            visited[nx][ny] = True
            count += 1

# Output the result
print(count)