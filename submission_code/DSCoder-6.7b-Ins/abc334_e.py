import sys
from collections import deque

MOD = 998244353
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

def bfs(x, y, grid, visited):
    q = deque([(x, y)])
    visited[x][y] = True
    cnt = 1
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not visited[nx][ny] and grid[nx][ny] == '#':
                visited[nx][ny] = True
                q.append((nx, ny))
                cnt += 1
    return cnt

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    reds = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    visited = [[False]*W for _ in range(H)]
    green_components = sum(row.count('#') for row in grid)
    total_red_cells = len(reds)
    expected_value = sum(green_components - bfs(x, y, grid, visited) + 1 for x, y in reds)
    expected_value *= pow(total_red_cells, MOD-2, MOD)
    print(expected_value % MOD)

solve()