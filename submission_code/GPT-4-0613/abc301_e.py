from heapq import heappop, heappush
from itertools import product

H, W, T = map(int, input().split())
grid = [list(input()) for _ in range(H)]
start, goal, candies = None, None, []
for i, j in product(range(H), range(W)):
    if grid[i][j] == 'S':
        start = (i, j)
    elif grid[i][j] == 'G':
        goal = (i, j)
    elif grid[i][j] == 'o':
        candies.append((i, j))
candies.append(goal)

INF = float('inf')
dist = [[INF]*W for _ in range(H)]
dist[start[0]][start[1]] = 0
heap = [(0, start)]
dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
while heap:
    d, (x, y) = heappop(heap)
    if d > dist[x][y]:
        continue
    for dx, dy in dx_dy:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < H and 0 <= ny < W) or grid[nx][ny] == '#':
            continue
        nd = d + 1
        if nd < dist[nx][ny]:
            dist[nx][ny] = nd
            heappush(heap, (nd, (nx, ny)))

dp = [[INF]*(len(candies)+1) for _ in range(1<<len(candies))]
dp[0][0] = 0
for i in range(1<<len(candies)):
    for j in range(len(candies)):
        if dp[i][j] == INF:
            continue
        for k in range(len(candies)):
            if (i>>k)&1:
                continue
            dp[i|(1<<k)][k] = min(dp[i|(1<<k)][k], dp[i][j] + abs(candies[j][0]-candies[k][0]) + abs(candies[j][1]-candies[k][1]))

ans = -1
for i in range(1<<len(candies)):
    for j in range(len(candies)):
        if dp[i][j] + dist[candies[j][0]][candies[j][1]] <= T:
            ans = max(ans, bin(i).count('1')-1)
print(ans)