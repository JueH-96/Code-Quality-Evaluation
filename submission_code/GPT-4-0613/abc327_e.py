import sys
import math
from operator import itemgetter

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    P = sorted([(p, i) for i, p in enumerate(P)], key=itemgetter(0), reverse=True)
    dp = [[-float('inf')] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    for i in range(N):
        for j in range(i + 1):
            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j] + P[i][0] * 0.9 ** (N - 1 - j))
            dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + P[i][0] * 0.9 ** (N - 1 - j))
    ans = max(dp[N][k] - 1200 * math.sqrt(k) for k in range(N + 1))
    print(ans)

solve()