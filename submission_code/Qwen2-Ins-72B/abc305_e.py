from collections import defaultdict, deque
from heapq import heappush, heappop

N, M, K = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(M)]
guards = [list(map(int, input().split())) for _ in range(K)]

graph = defaultdict(list)
for a, b in edges:
    graph[a].append(b)
    graph[b].append(a)

def dijkstra(start):
    dist = {start: 0}
    queue = [(0, start)]
    while queue:
        d, node = heappop(queue)
        if d > dist[node]:
            continue
        for neighbor in graph[node]:
            if neighbor not in dist or dist[neighbor] > d + 1:
                dist[neighbor] = d + 1
                heappush(queue, (d + 1, neighbor))
    return dist

guarded = set()
for p, h in guards:
    dist = dijkstra(p)
    for node, d in dist.items():
        if d <= h:
            guarded.add(node)

print(len(guarded))
print(' '.join(map(str, sorted(guarded))))