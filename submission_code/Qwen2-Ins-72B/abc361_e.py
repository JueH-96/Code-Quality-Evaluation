from heapq import heappush, heappop
from collections import defaultdict

def solve():
    N = int(input())
    edges = defaultdict(list)
    for _ in range(N-1):
        A, B, C = map(int, input().split())
        edges[A-1].append((B-1, C))
        edges[B-1].append((A-1, C))

    def dijkstra(start):
        dist = [float('inf')] * N
        dist[start] = 0
        heap = [(0, start)]
        while heap:
            d, v = heappop(heap)
            if d > dist[v]:
                continue
            for u, w in edges[v]:
                if dist[u] > dist[v] + w:
                    dist[u] = dist[v] + w
                    heappush(heap, (dist[u], u))
        return dist

    min_dist = float('inf')
    for i in range(N):
        dist = dijkstra(i)
        min_dist = min(min_dist, sum(dist))

    print(min_dist)

solve()