# YOUR CODE HERE
from collections import defaultdict
import sys

def dfs(node, parent, adj, required, dp):
    children_sum = 0
    for child in adj[node]:
        if child != parent:
            children_sum += dfs(child, node, adj, required, dp)
    
    if node in required or children_sum > 0:
        dp[node] = 1 + children_sum
    else:
        dp[node] = 0
    
    return dp[node]

def solve():
    N, K = map(int, input().split())
    adj = defaultdict(list)
    
    for _ in range(N-1):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
    
    required = set(map(int, input().split()))
    
    dp = [0] * (N + 1)
    dfs(1, 0, adj, required, dp)
    
    print(max(dp))

solve()