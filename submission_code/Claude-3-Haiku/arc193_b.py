import sys
from collections import defaultdict

MOD = 998244353

def solve():
    N = int(sys.stdin.readline().strip())
    s = sys.stdin.readline().strip()

    # Build the graph
    graph = defaultdict(list)
    for i in range(N):
        graph[i].append((i + 1) % N)
        if s[i] == '1':
            graph[i].append(N)

    # Count the number of possible in-degree sequences
    dp = [0] * (N + 1)
    dp[0] = 1
    for i in range(N):
        new_dp = [0] * (N + 1)
        for j in range(N + 1):
            for neighbor in graph[i]:
                new_dp[neighbor] = (new_dp[neighbor] + dp[j]) % MOD
        dp = new_dp

    print(sum(dp) % MOD)

solve()