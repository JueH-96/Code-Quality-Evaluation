import sys
from heapq import heappop, heappush

H, W, Y = map(int, sys.stdin.readline().split())
A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

INF = 10**9+7
dist = [[INF]*W for _ in range(H)]
Q = []
for i in range(H):
    for j in range(W):
        if i == 0 or i == H-1 or j == 0 or j == W-1:
            heappush(Q, (A[i][j], i, j))
            dist[i][j] = A[i][j]

cnt = [0]*(10**5+10)
while Q:
    d, x, y = heappop(Q)
    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        if nx < 0 or H <= nx or ny < 0 or W <= ny:
            continue
        nd = max(d, A[nx][ny])
        if dist[nx][ny] > nd:
            dist[nx][ny] = nd
            heappush(Q, (nd, nx, ny))

for i in range(H):
    for j in range(W):
        cnt[dist[i][j]] += 1

for i in range(10**5, 0, -1):
    cnt[i-1] += cnt[i]

for i in range(1, Y+1):
    print(cnt[i])