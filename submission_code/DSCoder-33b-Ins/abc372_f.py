import sys

MOD = 998244353

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Create adjacency list
    adj = [[] for _ in range(N+1)]
    for i in range(1, N):
        adj[i].append(i+1)
    for u, v in edges:
        adj[u].append(v)

    # Initialize dp table
    dp = [[0]*(K+1) for _ in range(N+1)]
    dp[1][0] = 1

    # Dynamic programming
    for k in range(1, K+1):
        for v in range(1, N+1):
            for u in adj[v]:
                dp[u][k] = (dp[u][k] + dp[v][k-1]) % MOD

    print(dp[1][K])

solve()