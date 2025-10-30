import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
grid = []
for i in range(H):
    s = data[index]
    grid.append(s)
    index += 1

# Define directions for 8 adjacent cells
dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Initialize visited array
visited = [[False for _ in range(W)] for _ in range(H)]

# Count of connected components
count = 0

# Iterate through each cell
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#' and not visited[i][j]:
            count += 1
            # BFS to mark all connected cells
            queue = collections.deque()
            queue.append((i, j))
            visited[i][j] = True
            while queue:
                x, y = queue.popleft()
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and not visited[nx][ny]:
                        queue.append((nx, ny))
                        visited[nx][ny] = True

# Output the result
print(count)