import sys
from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u-1].append((v-1, w))  # Convert to 0-indexed
    
    # DP state: dp[mask][v] = minimum cost to visit vertices in mask and end at vertex v
    INF = float('inf')
    dp = [[INF] * N for _ in range(1 << N)]
    
    # Initialize: start at each vertex
    for start in range(N):
        dp[1 << start][start] = 0
    
    # Fill DP table
    for mask in range(1 << N):
        for v in range(N):
            if dp[mask][v] == INF:
                continue
            
            # Try going to each neighbor
            for next_v, weight in graph[v]:
                new_mask = mask | (1 << next_v)
                dp[new_mask][next_v] = min(dp[new_mask][next_v], dp[mask][v] + weight)
    
    # Find minimum cost to visit all vertices
    full_mask = (1 << N) - 1
    result = INF
    
    for v in range(N):
        result = min(result, dp[full_mask][v])
    
    if result == INF:
        print("No")
    else:
        print(result)

solve()