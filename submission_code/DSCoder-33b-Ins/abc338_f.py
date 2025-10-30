# YOUR CODE HERE
import sys

def bellman_ford(graph, source):
    dist = [float('inf')] * len(graph)
    dist[source] = 0

    for _ in range(len(graph) - 1):
        for u in range(len(graph)):
            for v, w in graph[u]:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

    for u in range(len(graph)):
        for v, w in graph[u]:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return None

    return dist

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(N)]

    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        graph[u - 1].append((v - 1, w))

    min_dist = float('inf')
    for source in range(N):
        dist = bellman_ford(graph, source)
        if dist is not None:
            min_dist = min(min_dist, sum(dist))

    if min_dist == float('inf'):
        print('No')
    else:
        print(min_dist)

solve()