# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def solve(N, M, K, edges):
    # Initialize the adjacency list
    adj = defaultdict(list)
    for i in range(N):
        adj[i].append((i + 1) % N)
    for x, y in edges:
        adj[x - 1].append(y - 1)
    
    # Initialize the dp array
    dp = [0] * N
    dp[0] = 1
    
    # Perform the dynamic programming
    for _ in range(K):
        new_dp = [0] * N
        for u in range(N):
            for v in adj[u]:
                new_dp[v] = (new_dp[v] + dp[u]) % MOD
        dp = new_dp
    
    return sum(dp) % MOD

# Read input
N, M, K = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]

# Solve and print the result
print(solve(N, M, K, edges))