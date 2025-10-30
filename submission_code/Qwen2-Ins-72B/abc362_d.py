from heapq import heappush, heappop

N, M = map(int, input().split())
A = list(map(int, input().split()))
edges = [list(map(int, input().split())) for _ in range(M)]

graph = [[] for _ in range(N)]
for u, v, w in edges:
    graph[u-1].append((v-1, w))
    graph[v-1].append((u-1, w))

dist = [float('inf')] * N
dist[0] = A[0]
pq = [(dist[0], 0)]

while pq:
    d, u = heappop(pq)
    if d > dist[u]:
        continue
    for v, w in graph[u]:
        if dist[u] + A[v] + w < dist[v]:
            dist[v] = dist[u] + A[v] + w
            heappush(pq, (dist[v], v))

print(' '.join(map(str, dist[1:])))