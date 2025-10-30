from collections import defaultdict
from itertools import accumulate

def solve(N, M, LRX):
    MOD = 998244353
    LRX = sorted(LRX, key=lambda x: x[2])
    LRX = [(l-1, r-1, x-1) for l, r, x in LRX]

    # dp[i][j] = number of ways to fill the first i+1 slots with j as the max
    dp = [[0] * N for _ in range(N)]
    dp[0][0] = 1
    for i in range(1, N):
        dp[i][i] = 1
        for j in range(i):
            dp[i][j] = dp[i-1][j] * (N-i) % MOD
            dp[i][i] += dp[i-1][j] * (i-j) % MOD
            dp[i][i] %= MOD

    # build a graph of constraints
    graph = defaultdict(list)
    for l, r, x in LRX:
        graph[x].append((l, r))

    # dfs through the graph to calculate the number of ways to fill the slots
    ans = 0
    visited = [False] * N
    stack = [(N-1, N-1)]
    while stack:
        x, max_val = stack.pop()
        if visited[x]:
            continue
        visited[x] = True
        ways = dp[max_val][x]
        for l, r in graph[x]:
            if l <= max_val:
                ways -= dp[r][l] * (max_val-l+1) % MOD
                ways %= MOD
        ans += ways * (N-x-1) % MOD
        ans %= MOD
        for l, r in graph[x]:
            stack.append((l, r))
    return ans

# read inputs
N, M = map(int, input().split())
LRX = [tuple(map(int, input().split())) for _ in range(M)]

# solve and print answer
print(solve(N, M, LRX))