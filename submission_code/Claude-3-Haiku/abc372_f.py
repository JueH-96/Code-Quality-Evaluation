from collections import defaultdict

MOD = 998244353

def solve():
    N, M, K = map(int, input().split())
    edges = defaultdict(list)
    for _ in range(M):
        x, y = map(int, input().split())
        edges[x].append(y)

    dp = [[0] * (K + 1) for _ in range(N + 1)]
    dp[1][0] = 1

    for i in range(1, N + 1):
        for j in range(1, K + 1):
            for v in edges[i]:
                dp[v][j] = (dp[v][j] + dp[i][j - 1]) % MOD

    return sum(dp[1][K]) % MOD

print(solve())