MOD = 998244353

N, M, K = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(N + 1)]

# Add cycle edges
for i in range(1, N + 1):
    next_vertex = i + 1 if i < N else 1
    adj[i].append(next_vertex)

# Add additional edges
for _ in range(M):
    x, y = map(int, input().split())
    adj[x].append(y)

# DP: dp[i] = number of ways to reach vertex i after current number of steps
dp = [0] * (N + 1)
dp[1] = 1

for step in range(K):
    new_dp = [0] * (N + 1)
    for v in range(1, N + 1):
        if dp[v] > 0:
            for next_v in adj[v]:
                new_dp[next_v] = (new_dp[next_v] + dp[v]) % MOD
    dp = new_dp

print(sum(dp[1:]) % MOD)