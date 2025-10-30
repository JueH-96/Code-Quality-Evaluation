def can_achieve_ratio(n, edges, ratio):
    # Create adjacency list
    adj = [[] for _ in range(n + 1)]
    for u, v, b, c in edges:
        adj[u].append((v, b - ratio * c))
    
    # DP: dp[v] = maximum weight to reach vertex v
    dp = [-float('inf')] * (n + 1)
    dp[1] = 0
    
    # Process vertices in order (since u < v, we can process in order)
    for u in range(1, n):
        if dp[u] == -float('inf'):
            continue
        for v, weight in adj[u]:
            dp[v] = max(dp[v], dp[u] + weight)
    
    return dp[n] >= 0

def solve():
    n, m = map(int, input().split())
    edges = []
    
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))
    
    # Binary search on the ratio
    left, right = 0.0, 10000.0  # Maximum possible ratio is max_beauty/min_cost = 10000/1
    
    # Binary search with 100 iterations for precision
    for _ in range(100):
        mid = (left + right) / 2
        if can_achieve_ratio(n, edges, mid):
            left = mid
        else:
            right = mid
    
    print(f"{left:.16f}")

solve()