import sys
from collections import deque

h, w = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(h)]

visited = [[False] * w for _ in range(h)]
directions = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

count = 0

for i in range(h):
    for j in range(w):
        if grid[i][j] == '#' and not visited[i][j]:
            count += 1
            q = deque()
            q.append((i, j))
            visited[i][j] = True
            while q:
                x, y = q.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if grid[nx][ny] == '#' and not visited[nx][ny]:
                            visited[nx][ny] = True
                            q.append((nx, ny))

print(count)