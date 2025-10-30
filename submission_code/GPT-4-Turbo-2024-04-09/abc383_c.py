import sys
from collections import deque

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
D = int(data[2])

grid = []
index = 3
for _ in range(H):
    grid.append(data[index])
    index += 1

# Directions for moving in the grid (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# To keep track of visited and humidified cells
visited = [[False] * W for _ in range(H)]
humidified = [[False] * W for _ in range(H)]

def is_valid(x, y):
    return 0 <= x < H and 0 <= y < W and grid[x][y] != '#'

def bfs(x, y):
    queue = deque([(x, y, 0)])
    visited[x][y] = True
    humidified[x][y] = True
    
    while queue:
        cx, cy, dist = queue.popleft()
        
        if dist < D:
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny) and not visited[nx][ny]:
                    visited[nx][ny] = True
                    humidified[nx][ny] = True
                    queue.append((nx, ny, dist + 1))

# Process each cell in the grid
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H' and not visited[i][j]:
            bfs(i, j)

# Count the humidified floor cells
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] != '#' and humidified[i][j]:
            count += 1

print(count)