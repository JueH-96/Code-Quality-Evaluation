import sys
from collections import deque

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    degrees = [[0]*W for _ in range(H)]
    magnets = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '#']

    for mx, my in magnets:
        visited = [[False]*W for _ in range(H)]
        visited[mx][my] = True
        queue = deque([(mx, my, 0)])

        while queue:
            x, y, d = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == '.':
                    visited[nx][ny] = True
                    degrees[nx][ny] = max(degrees[nx][ny], d + 1)
                    queue.append((nx, ny, d + 1))

    print(max(max(row) for row in degrees))

solve()