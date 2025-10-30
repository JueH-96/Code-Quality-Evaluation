from collections import defaultdict
from math import inf

def find_min_weight_walk(N, M, edges):
    # Create adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    # Initialize distances and visited set
    dist = [inf] * (N + 1)
    dist[1] = 0
    visited = set()

    # Perform Bellman-Ford algorithm
    for _ in range(N - 1):
        for u in range(1, N + 1):
            for v, w in graph[u]:
                dist[v] = min(dist[v], dist[u] + w)

    # Check for negative cycles
    for u in range(1, N + 1):
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                return "No"

    # Find the minimum weight walk
    for i in range(1, N + 1):
        visited.add(i)
        if len(visited) == N:
            return dist[i]

    return "No"

# Read input
N, M = map(int, input().split())
edges = []
for _ in range(M):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

# Solve the problem
print(find_min_weight_walk(N, M, edges))