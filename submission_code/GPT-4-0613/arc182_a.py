# YOUR CODE HERE
import sys
input = sys.stdin.readline
mod = 998244353
N, Q = map(int, input().split())
PV = [list(map(int, input().split())) for _ in range(Q)]
PV.sort(key=lambda x: (-x[1], x[0]))
dp = [[0]*(N+1) for _ in range(Q+1)]
dp[0][0] = 1
for i in range(Q):
    P, V = PV[i]
    dp[i+1] = dp[i][:]
    for j in range(N, -1, -1):
        if j < P:
            dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j] * (P-j)) % mod
        else:
            dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j] * (N-j+1)) % mod
        dp[i+1][j] = dp[i+1][j] * j % mod
print(sum(dp[Q][i] * dp[Q][i] % mod for i in range(N+1)) % mod)