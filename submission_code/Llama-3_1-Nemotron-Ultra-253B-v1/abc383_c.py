import sys
from collections import deque

h, w, d = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(h)]

dist = [[-1] * w for _ in range(h)]
q = deque()

# Initialize BFS with all humidifiers
for i in range(h):
    for j in range(w):
        if grid[i][j] == 'H':
            dist[i][j] = 0
            q.append((i, j))

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while q:
    i, j = q.popleft()
    current_dist = dist[i][j]
    if current_dist >= d:
        continue
    for dx, dy in directions:
        ni, nj = i + dx, j + dy
        if 0 <= ni < h and 0 <= nj < w:
            if grid[ni][nj] != '#' and dist[ni][nj] == -1:
                new_dist = current_dist + 1
                if new_dist > d:
                    continue
                dist[ni][nj] = new_dist
                q.append((ni, nj))

count = 0
for i in range(h):
    for j in range(w):
        if grid[i][j] != '#' and dist[i][j] != -1 and dist[i][j] <= d:
            count += 1

print(count)