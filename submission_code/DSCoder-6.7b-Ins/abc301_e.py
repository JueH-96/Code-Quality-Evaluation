from collections import deque

def solve():
    H, W, T = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    visited = [[0]*W for _ in range(H)]
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                sx, sy = i, j
            elif grid[i][j] == 'G':
                gx, gy = i, j

    q = deque([(sx, sy, 0, 0)])
    visited[sx][sy] = 1
    candies = 0

    while q:
        x, y, t, c = q.popleft()
        if (x, y) == (gx, gy) and t <= T:
            candies = c
            break
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and visited[nx][ny] == 0 and t+1 <= T:
                if grid[nx][ny] == 'o':
                    q.append((nx, ny, t+1, c+1))
                    visited[nx][ny] = 1
                else:
                    q.append((nx, ny, t+1, c))
                    visited[nx][ny] = 1

    if (gx, gy) not in q:
        print(-1)
    else:
        print(candies)

solve()