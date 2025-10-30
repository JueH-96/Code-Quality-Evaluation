import sys

N = int(sys.stdin.readline().strip())

edges = []
for _ in range(N-1):
    u, v, l = map(int, sys.stdin.readline().strip().split())
    edges.append((l, u, v))

edges.sort()

dp = [0] * (N+1)
dp2 = [0] * (N+1)
dp[1] = edges[0][0]
dp2[1] = dp[1]

for i in range(2, N+1):
    dp[i] = min(dp[i-1] + edges[i-2][0], dp2[i-1] + edges[i-2][1] + edges[i-2][2])
    dp2[i] = max(dp2[i-1], dp[i] + edges[i-2][2])

for i in range(1, N+1):
    print(dp2[i])