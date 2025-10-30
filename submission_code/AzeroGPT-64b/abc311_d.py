import sys
from collections import deque

def dfs(graph, n, m):
    visited = [[False] * m for _ in range(n)]
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    queue = deque([(1, 1)]);
    total = 0

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            while 0 < nx < n and 0 < ny < m and graph[nx][ny] != '#':
                visited[nx][ny] = True
                total += 1
                nx, ny = nx + dx, ny + dy
                if graph[nx][ny] == '#':
                    graph[nx - dx][ny - dy] = '-'
                    break
            else:
                if 0 < nx < n and 0 < ny < m and not visited[nx][ny] and graph[nx][ny] != '#':
                    queue.append((nx, ny))

    return total

n, m = [int(num) for num in sys.stdin.readline().split()]
grid = []
for _ in range(n):
    grid.append(list(sys.stdin.readline().strip()))

result = dfs(grid, n, m)
print(result)