from collections import defaultdict
from math import inf

def max_beauty_cost_ratio(N, M, edges):
    # Create a graph representation
    graph = defaultdict(list)
    for u, v, b, c in edges:
        graph[u].append((v, b, c))

    # Initialize the DP table
    dp = [[0, 0] for _ in range(N+1)]  # (total_beauty, total_cost)

    # Perform dynamic programming
    for u in range(1, N+1):
        max_ratio = 0
        for v, b, c in graph[u]:
            total_beauty, total_cost = dp[v]
            ratio = (total_beauty + b) / (total_cost + c)
            if ratio > max_ratio:
                max_ratio = ratio
                dp[u][0] = total_beauty + b
                dp[u][1] = total_cost + c

    # Return the maximum ratio
    return dp[N][0] / dp[N][1]

# Read the input
N, M = map(int, input().split())
edges = []
for _ in range(M):
    u, v, b, c = map(int, input().split())
    edges.append((u, v, b, c))

# Solve the problem
print(max_beauty_cost_ratio(N, M, edges))