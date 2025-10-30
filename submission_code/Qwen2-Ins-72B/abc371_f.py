from bisect import bisect_left, bisect_right
from collections import deque

N = int(input())
X = list(map(int, input().split()))
Q = int(input())
TG = [list(map(int, input().split())) for _ in range(Q)]

X = [0] + X
TG = [TG[i-1] + [i] for i in range(1, Q+1)]
TG.sort(key=lambda x: x[1])

ans = 0
tasks = deque(TG)
pos = X[:]
dp = [0] * (N+1)
for i in range(1, N+1):
    dp[i] = dp[i-1] + pos[i] - pos[i-1]

while tasks:
    while tasks and tasks[0][1] <= pos[tasks[0][0]]:
        t, g, i = tasks.popleft()
        ans += abs(pos[t] - g) - abs(X[t] - g)
        pos[t] = g
        dp[t] = dp[t-1] + pos[t] - pos[t-1]
    t, g, i = tasks[0]
    l = bisect_left(pos, g)
    r = bisect_right(pos, g)
    if l == r:
        r += 1
    if l == 0:
        tasks[0][1] = pos[l]
        ans += dp[r-1] - dp[t] + pos[r] - pos[l]
    else:
        tasks[0][1] = pos[l-1]
        ans += dp[r-1] - dp[l-1] - (dp[t-1] - dp[l-1]) + pos[r] - pos[l] - (pos[l-1] - pos[l-2])

print(ans)