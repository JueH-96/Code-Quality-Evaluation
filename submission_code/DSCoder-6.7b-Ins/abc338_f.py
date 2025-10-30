import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * (N+1)
    dist[1] = 0

    for _ in range(N-1):
        for u in range(1, N+1):
            if dist[u] != float('inf'):
                for v, w in graph[u]:
                    dist[v] = min(dist[v], dist[u] + w)

    for u in range(1, N+1):
        if dist[u] != float('inf'):
            for v, w in graph[u]:
                if dist[v] > dist[u] + w:
                    return 'No'

    return min(dist[1:])

print(solve())