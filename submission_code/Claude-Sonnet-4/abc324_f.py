from collections import defaultdict
import sys

def can_achieve_ratio(edges, n, ratio):
    # Transform edges: new_weight = beauty - ratio * cost
    # Check if there's a path from 1 to n with non-negative total weight
    
    # Build adjacency list with transformed weights
    graph = defaultdict(list)
    for u, v, b, c in edges:
        weight = b - ratio * c
        graph[u].append((v, weight))
    
    # DP: dp[i] = maximum weight to reach vertex i
    dp = [-float('inf')] * (n + 1)
    dp[1] = 0
    
    # Process vertices in topological order (1 to n)
    for u in range(1, n + 1):
        if dp[u] == -float('inf'):
            continue
        for v, weight in graph[u]:
            dp[v] = max(dp[v], dp[u] + weight)
    
    return dp[n] >= -1e-9  # Allow small numerical errors

def solve():
    n, m = map(int, input().split())
    edges = []
    
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))
    
    # Binary search on the ratio
    left, right = 0.0, 10000.0  # Maximum possible ratio is max(b)/min(c)
    
    for _ in range(100):  # Sufficient iterations for required precision
        mid = (left + right) / 2
        if can_achieve_ratio(edges, n, mid):
            left = mid
        else:
            right = mid
    
    print(f"{left:.16f}")

solve()