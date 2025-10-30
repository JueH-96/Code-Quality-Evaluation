import sys
from collections import deque

H, W = map(int, input().split())
S = [list(input()) for _ in range(H)]
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

def bfs(sx, sy):
    visited = [[0]*W for _ in range(H)]
    visited[sx][sy] = 1
    queue = deque([(sx, sy)])
    cnt = 0
    while queue:
        x, y = queue.popleft()
        cnt += 1
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if nx < 0 or nx >= H or ny < 0 or ny >= W:
                continue
            if S[nx][ny] == '#' or visited[nx][ny]:
                continue
            visited[nx][ny] = 1
            queue.append((nx, ny))
    return cnt

res = 0
for i in range(H):
    for j in range(W):
        if S[i][j] == '.':
            res = max(res, bfs(i, j))
print(res)