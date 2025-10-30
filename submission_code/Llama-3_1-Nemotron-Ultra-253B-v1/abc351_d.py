import sys
from collections import deque

h, w = map(int, sys.stdin.readline().split())
grid = [list(sys.stdin.readline().strip()) for _ in range(h)]

safe = [[False] * w for _ in range(h)]
for i in range(h):
    for j in range(w):
        if grid[i][j] == '#':
            continue
        is_safe = True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == '#':
                is_safe = False
                break
        safe[i][j] = is_safe

visited = [[False] * w for _ in range(h)]
max_degree = 0

for i in range(h):
    for j in range(w):
        if grid[i][j] == '#':
            continue
        if not safe[i][j]:
            max_degree = max(max_degree, 1)
            continue
        if not visited[i][j]:
            queue = deque([(i, j)])
            visited[i][j] = True
            component = [(i, j)]
            adj_set = set()
            while queue:
                x, y = queue.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == '.':
                        if not visited[nx][ny]:
                            if safe[nx][ny]:
                                visited[nx][ny] = True
                                queue.append((nx, ny))
                                component.append((nx, ny))
                            else:
                                adj_set.add((nx, ny))
            current_degree = len(component) + len(adj_set)
            max_degree = max(max_degree, current_degree)

print(max_degree)