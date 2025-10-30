# YOUR CODE HERE
from collections import defaultdict
import heapq

N, M, X1 = map(int, input().split())
edges = defaultdict(list)
for _ in range(M):
    A, B, S, T = map(int, input().split())
    edges[A].append((B, T - S))
X = [0] * M

# Dijkstra's algorithm to find the minimum delay
dist = [float('inf')] * (N + 1)
dist[1] = X1
pq = [(X1, 1)]
while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]:
        continue
    for v, w in edges[u]:
        if dist[v] > d + w:
            dist[v] = d + w
            heapq.heappush(pq, (dist[v], v))

# Calculate the minimum delays for each train
for i in range(M):
    A, B, S, T = map(int, input().split())
    X[i] = max(0, dist[B] - (T - S))

print(*X[1:])