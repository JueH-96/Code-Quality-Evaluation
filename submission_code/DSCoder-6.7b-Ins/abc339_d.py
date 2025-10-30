from collections import deque

def solve():
    N = int(input())
    grid = [list(input()) for _ in range(N)]
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
                grid[i][j] = '.'
    d = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dist = [[[-1, -1] for _ in range(N)] for _ in range(N)]
    q = deque([(*players[0], 0), (*players[1], 1)])
    dist[players[0][0]][players[0][1]][0] = 0
    dist[players[1][0]][players[1][1]][1] = 0
    while q:
        i, j, p = q.popleft()
        for di, dj in d:
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N and grid[ni][nj] != '#' and dist[ni][nj][1-p] == -1:
                dist[ni][nj][1-p] = dist[i][j][p] + 1
                q.append((ni, nj, 1-p))
    if min(dist[players[1][0]][players[1][1]]) == -1:
        print(-1)
    else:
        print(max(dist[players[1][0]][players[1][1]]))

solve()