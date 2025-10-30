# YOUR CODE HERE
from collections import defaultdict

def dfs(node, parent, dist, graph, max_dist, min_dist):
    for neighbor, weight in graph[node]:
        if neighbor != parent:
            max_dist[neighbor] = max(max_dist[node] + weight, max_dist[neighbor])
            min_dist[neighbor] = min(min_dist[node] + weight, min_dist[neighbor])
            dfs(neighbor, node, dist, graph, max_dist, min_dist)

def solve(N, edges):
    graph = defaultdict(list)
    for u, v, l in edges:
        graph[u-1].append((v-1, l))
        graph[v-1].append((u-1, l))
    
    max_dist = [0] * N
    min_dist = [0] * N
    dfs(0, -1, 0, graph, max_dist, min_dist)
    
    for i in range(N):
        max_dist[i] += min_dist[i]
    
    dp = [0] * N
    dp[0] = max_dist[0]
    for i in range(1, N):
        dp[i] = max(dp[i-1] + max_dist[i], dp[i-1])
    
    ans = [0] * (N+1)
    for i in range(N-1, -1, -1):
        ans[i] = dp[i-1] + max_dist[i] if i > 0 else dp[0]
    
    return ans

N = int(input())
edges = []
for _ in range(N-1):
    u, v, l = map(int, input().split())
    edges.append((u, v, l))

result = solve(N, edges)
for k in range(1, N+1):
    print(result[k-1])