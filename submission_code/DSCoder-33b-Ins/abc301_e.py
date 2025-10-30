from collections import deque

H, W, T = map(int, input().split())
grid = [list(input()) for _ in range(H)]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

def bfs(x, y, t):
    visited = [[False]*W for _ in range(H)]
    visited[x][y] = True
    queue = deque([(x, y, t, 0)])
    while queue:
        x, y, t, c = queue.popleft()
        if (x, y) == goal:
            return c
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] != '#':
                visited[nx][ny] = True
                if grid[nx][ny] == 'o':
                    queue.append((nx, ny, t-1, c+1))
                else:
                    queue.append((nx, ny, t-1, c))
    return -1

print(bfs(start[0], start[1], T))