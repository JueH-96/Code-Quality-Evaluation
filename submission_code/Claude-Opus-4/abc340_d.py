import heapq

N = int(input())

# Build adjacency list
adj = [[] for _ in range(N + 1)]

for i in range(1, N):
    A, B, X = map(int, input().split())
    # From stage i, we can go to stage i+1 with cost A
    adj[i].append((i + 1, A))
    # From stage i, we can go to stage X with cost B
    adj[i].append((X, B))

# Dijkstra's algorithm
dist = [float('inf')] * (N + 1)
dist[1] = 0
pq = [(0, 1)]  # (distance, node)

while pq:
    d, u = heapq.heappop(pq)
    
    if d > dist[u]:
        continue
    
    for v, w in adj[u]:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            heapq.heappush(pq, (dist[v], v))

print(dist[N])