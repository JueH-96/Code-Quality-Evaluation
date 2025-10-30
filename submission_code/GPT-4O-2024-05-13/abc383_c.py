# YOUR CODE HERE
import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
D = int(data[2])

grid = []
index = 3
for i in range(H):
    grid.append(data[index])
    index += 1

# Directions for up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS to find all humidified cells
def bfs(humidifiers):
    visited = [[False] * W for _ in range(H)]
    queue = deque()
    
    for (x, y) in humidifiers:
        queue.append((x, y, 0))
        visited[x][y] = True
    
    humidified_count = 0
    
    while queue:
        x, y, dist = queue.popleft()
        if grid[x][y] == '.':
            humidified_count += 1
        
        if dist < D:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] != '#':
                    visited[nx][ny] = True
                    queue.append((nx, ny, dist + 1))
    
    return humidified_count

# Find all humidifiers
humidifiers = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            humidifiers.append((i, j))

# Calculate the number of humidified cells
result = bfs(humidifiers)
print(result)