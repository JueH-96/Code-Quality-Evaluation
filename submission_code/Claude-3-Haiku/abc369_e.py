from collections import defaultdict
import heapq

def dijkstra(graph, start, end, bridges):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start - 1] = 0
    heap = [(0, start)]
    used_bridges = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u == end:
            return d, used_bridges
        if d > dist[u - 1]:
            continue
        for v, w in graph[u]:
            if (min(u, v), max(u, v)) in bridges:
                used_bridges.add((min(u, v), max(u, v)))
            new_dist = d + w
            if new_dist < dist[v - 1]:
                dist[v - 1] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return float('inf'), used_bridges

def solve():
    n, m = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(m):
        u, v, t = map(int, input().split())
        graph[u].append((v, t))
        graph[v].append((u, t))

    q = int(input())
    for _ in range(q):
        k = int(input())
        bridges = set()
        for _ in range(k):
            b = int(input())
            u, v = map(int, input().split())
            bridges.add((min(u, v), max(u, v)))
        time, used_bridges = dijkstra(graph, 1, n, bridges)
        print(time if used_bridges == bridges else -1)

solve()