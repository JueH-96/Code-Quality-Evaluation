from itertools import combinations
from collections import deque
import sys

def shortest_path(adj, N):
    dist = [float('inf')] * (N+1)
    dist[1] = 0
    queue = deque([1])
    
    while queue:
        v = queue.popleft()
        for u, w in adj[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                queue.append(u)
    
    return dist[N]

def solve():
    # Read input
    N, M, K = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    # Try all possible combinations of K edges to set to weight 1
    max_dist = 0
    for selected in combinations(range(M), K):
        # Create adjacency list with selected edges having weight 1
        adj = [[] for _ in range(N+1)]
        for i in range(M):
            u, v = edges[i]
            w = 1 if i in selected else 0
            adj[u].append((v, w))
        
        # Find shortest path with current configuration
        dist = shortest_path(adj, N)
        max_dist = max(max_dist, dist)
    
    print(max_dist)

if __name__ == "__main__":
    solve()