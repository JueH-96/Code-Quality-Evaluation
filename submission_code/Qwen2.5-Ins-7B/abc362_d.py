# YOUR CODE HERE
import heapq
from collections import defaultdict

N, M = map(int, input().split())
A = list(map(int, input().split()))
edges = defaultdict(list)
for _ in range(M):
    U, V, B = map(int, input().split())
    edges[U-1].append((V-1, B))
    edges[V-1].append((U-1, B))

pq = [(A[0], 0)]
dist = [float('inf')] * N
dist[0] = A[0]

while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]:
        continue
    for v, b in edges[u]:
        new_d = d + b + A[v]
        if new_d < dist[v]:
            dist[v] = new_d
            heapq.heappush(pq, (new_d, v))

print(*dist[1:])