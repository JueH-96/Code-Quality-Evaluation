def count_ways(N, M, K, edges):
    MOD = 998244353
    
    # Create adjacency list for the graph
    graph = [[] for _ in range(N + 1)]
    
    # Add the edges from 1 to N (circular)
    for i in range(1, N + 1):
        graph[i].append(i % N + 1)
    
    # Add the additional edges from the input
    for x, y in edges:
        graph[x].append(y)
    
    # Initialize dp array
    dp = [0] * (N + 1)
    dp[1] = 1  # Starting at vertex 1
    
    # Iterate K times to calculate the number of ways
    for _ in range(K):
        new_dp = [0] * (N + 1)
        for v in range(1, N + 1):
            for next_v in graph[v]:
                new_dp[next_v] = (new_dp[next_v] + dp[v]) % MOD
        dp = new_dp
    
    return dp[1]  # Return the number of ways to return to vertex 1 after K moves

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N, M, K
N, M, K = map(int, data[0].split())
edges = [tuple(map(int, line.split())) for line in data[1:M + 1]]

# Get the result
result = count_ways(N, M, K, edges)

# Print the result
print(result)