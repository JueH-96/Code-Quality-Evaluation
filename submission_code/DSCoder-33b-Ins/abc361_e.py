import sys
from heapq import heappop, heappush

def solve():
    read = sys.stdin.readline
    write = sys.stdout.write

    N = int(read())
    graph = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b, c = map(int, read().split())
        graph[a].append((b, c))
        graph[b].append((a, c))

    def dijkstra(start):
        dist = [float('inf')] * (N+1)
        dist[start] = 0
        heap = [(0, start)]
        while heap:
            d, node = heappop(heap)
            if d > dist[node]:
                continue
            for neighbor, cost in graph[node]:
                if d + cost < dist[neighbor]:
                    dist[neighbor] = d + cost
                    heappush(heap, (dist[neighbor], neighbor))
        return dist

    dist1 = dijkstra(1)
    max_dist = max(enumerate(dist1), key=lambda x: x[1])[0]
    dist2 = dijkstra(max_dist)
    write(str(max(dist2)))

solve()