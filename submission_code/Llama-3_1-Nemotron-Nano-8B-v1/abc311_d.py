import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
grid = []
for _ in range(n):
    s = sys.stdin.readline().strip()
    grid.append(list(s))

visited = [[False] * m for _ in range(n)]
enqueued = [[False] * m for _ in range(n)]
count = 0

start_i, start_j = 1, 1
if grid[start_i][start_j] == '.':
    visited[start_i][start_j] = True
    count += 1
    queue = deque([(start_i, start_j)])
    enqueued[start_i][start_j] = True

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while queue:
    i, j = queue.popleft()
    for di, dj in directions:
        path = []
        current_i, current_j = i, j
        path.append((current_i, current_j))
        while True:
            next_i = current_i + di
            next_j = current_j + dj
            if 0 <= next_i < n and 0 <= next_j < m and grid[next_i][next_j] == '.':
                current_i, current_j = next_i, next_j
                path.append((current_i, current_j))
            else:
                break
        for x, y in path:
            if not visited[x][y]:
                visited[x][y] = True
                count += 1
        endpoint_i, endpoint_j = path[-1]
        if not enqueued[endpoint_i][endpoint_j]:
            enqueued[endpoint_i][endpoint_j] = True
            queue.append((endpoint_i, endpoint_j))

print(count)