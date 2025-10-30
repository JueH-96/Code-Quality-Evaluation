n, m, k = map(int, input().split())

# Build adjacency list (0-indexed internally for efficiency)
adj = [[] for _ in range(n)]

# First N edges: i -> (i+1) % n
for i in range(n):
    adj[i].append((i + 1) % n)

# Next M edges
for _ in range(m):
    x, y = map(int, input().split())
    adj[x - 1].append(y - 1)  # Convert to 0-indexed

mod = 998244353

# DP: dp[v] = number of ways to reach vertex v
dp = [0] * n
dp[0] = 1  # Start at vertex 0 (which is vertex 1 in 1-indexed)

# Perform K moves
for _ in range(k):
    new_dp = [0] * n
    for u in range(n):
        if dp[u]:  # Only process vertices that are reachable
            for v in adj[u]:
                new_dp[v] = (new_dp[v] + dp[u]) % mod
    dp = new_dp

# Sum all dp values to get total number of paths
print(sum(dp) % mod)