from sys import stdin, stdout
from collections import deque

H, W = map(int, stdin.readline().split())
grid = [list(stdin.readline().strip()) for _ in range(H)]
visited = [[False]*W for _ in range(H)]
dx = [-1, 0, 1, 0, -1, -1, 1, 1]
dy = [0, 1, 0, -1, -1, 1, -1, 1]

def bfs(i, j):
    queue = deque([(i, j)])
    visited[i][j] = True
    while queue:
        x, y = queue.popleft()
        for k in range(8):
            nx, ny = x + dx[k], y + dy[k]
            if nx < 0 or nx >= H or ny < 0 or ny >= W:
                continue
            if grid[nx][ny] == '#' and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append((nx, ny))

count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#' and not visited[i][j]:
            bfs(i, j)
            count += 1

stdout.write(str(count) + '
')