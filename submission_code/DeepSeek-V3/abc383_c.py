from collections import deque

# Read input
H, W, D = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]

# Initialize visited and queue
visited = [[False for _ in range(W)] for _ in range(H)]
queue = deque()

# Find all humidifiers and mark them as visited
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            queue.append((i, j, 0))
            visited[i][j] = True

# Directions for moving up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Perform BFS
while queue:
    i, j, dist = queue.popleft()
    if dist >= D:
        continue
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            if grid[ni][nj] != '#' and not visited[ni][nj]:
                visited[ni][nj] = True
                queue.append((ni, nj, dist + 1))

# Count the number of humidified floor cells
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.' and visited[i][j]:
            count += 1
        elif grid[i][j] == 'H':
            count += 1

print(count)