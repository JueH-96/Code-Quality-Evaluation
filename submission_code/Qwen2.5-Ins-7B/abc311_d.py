# YOUR CODE HERE
from collections import deque

def can_reach(grid, i, j):
    N, M = len(grid), len(grid[0])
    visited = [[False] * M for _ in range(N)]
    visited[i][j] = True
    queue = deque([(i, j)])
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    reachable = 0

    while queue:
        x, y = queue.popleft()
        if grid[x][y] == '.':
            reachable += 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < N - 1 and 0 < ny < M - 1 and not visited[nx][ny] and grid[nx][ny] == '.':
                visited[nx][ny] = True
                queue.append((nx, ny))
    return reachable

N, M = map(int, input().split())
grid = [input() for _ in range(N)]
print(can_reach(grid, 1, 1))