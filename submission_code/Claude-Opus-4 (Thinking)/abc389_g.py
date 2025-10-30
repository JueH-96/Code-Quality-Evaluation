n, p = map(int, input().split())

from itertools import combinations
from collections import deque

# Precompute all possible edges
all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

results = []

for m in range(n - 1, n * (n - 1) // 2 + 1):
    count = 0
    
    for edge_set in combinations(all_edges, m):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from vertex 0 (represents vertex 1 in problem)
        dist = [-1] * n
        dist[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        # Check if connected and count even distances
        if -1 not in dist:
            even_count = sum(1 for d in dist if d % 2 == 0)
            if even_count == n // 2:
                count = (count + 1) % p
    
    results.append(count)

print(' '.join(map(str, results)))