# YOUR CODE HERE

import sys
from heapq import heappop, heappush
from collections import defaultdict

def solve(n, edges):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    def dijkstra(start):
        dist = [float('inf')] * (n + 1)
        dist[start] = 0
        heap = [(0, start)]
        while heap:
            d, u = heappop(heap)
            if d != dist[u]: continue
            for v, w in graph[u]:
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heappush(heap, (dist[v], v))
        return dist

    dist1 = dijkstra(1)
    distN = dijkstra(n)
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = max(dist1[i], distN[i])
    return ans

n = int(input())
edges = []
for _ in range(n - 1):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

ans = solve(n, edges)
for i in range(1, n + 1):
    print(ans[i])