# YOUR CODE HERE

import sys
from collections import deque

H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

def bfs(x, y):
    visited = [[False]*W for _ in range(H)]
    visited[x][y] = True
    queue = deque([(x, y)])
    count = 1
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == '.':
                visited[nx][ny] = True
                queue.append((nx, ny))
                count += 1
    return count

max_freedom = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            max_freedom = max(max_freedom, bfs(i, j))

print(max_freedom)