def can_achieve_ratio(edges, n, ratio):
    # Create adjacency list with modified weights
    adj = [[] for _ in range(n + 1)]
    for u, v, b, c in edges:
        weight = b - ratio * c
        adj[u].append((v, weight))
    
    # DP to find maximum weight path from 1 to N
    dp = [-float('inf')] * (n + 1)
    dp[1] = 0
    
    for u in range(1, n + 1):
        if dp[u] == -float('inf'):
            continue
        for v, weight in adj[u]:
            dp[v] = max(dp[v], dp[u] + weight)
    
    return dp[n] >= -1e-9  # Allow for small numerical errors

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, b, c = map(int, input().split())
    edges.append((u, v, b, c))

# Binary search on the answer
left, right = 0.0, 10000.0
for _ in range(100):  # Enough iterations for precision
    mid = (left + right) / 2
    if can_achieve_ratio(edges, n, mid):
        left = mid
    else:
        right = mid

print(left)