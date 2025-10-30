import sys
from heapq import heappop, heappush
from collections import defaultdict

def dijkstra(graph, start, end=None):
    queue = [(0, start)]
    distances = {start: 0}
    while queue:
        dist, node = heappop(queue)
        if node == end:
            return dist
        if node not in distances:
            distances[node] = dist
            for next_node, next_dist in graph[node].items():
                if next_node not in distances:
                    heappush(queue, (dist + next_dist, next_node))
    return float('inf')

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    graph = defaultdict(dict)
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        graph[a][b] = graph[b][a] = 1
    guards = []
    for _ in range(K):
        p, h = map(int, sys.stdin.readline().split())
        guards.append((p, h))
    guarded = set()
    for p, h in guards:
        for node in range(1, N+1):
            if dijkstra(graph, node, p) <= h:
                guarded.add(node)
    guarded = sorted(list(guarded))
    print(len(guarded))
    print(' '.join(map(str, guarded)))

solve()