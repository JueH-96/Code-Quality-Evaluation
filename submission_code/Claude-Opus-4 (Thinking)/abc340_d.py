import heapq

n = int(input())
adj = [[] for _ in range(n+1)]

for i in range(1, n):
    a, b, x = map(int, input().split())
    adj[i].append((i+1, a))
    adj[i].append((x, b))

# Dijkstra's algorithm
dist = [float('inf')] * (n+1)
dist[1] = 0
pq = [(0, 1)]

while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]:
        continue
    for v, w in adj[u]:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            heapq.heappush(pq, (dist[v], v))

print(dist[n])