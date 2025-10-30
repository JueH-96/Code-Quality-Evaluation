import heapq

N, M = map(int, input().split())
A = list(map(int, input().split()))

# Build adjacency list
graph = [[] for _ in range(N)]
for _ in range(M):
    u, v, b = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1  # Convert to 0-indexed
    graph[u].append((v, b))
    graph[v].append((u, b))

# Dijkstra's algorithm
dist = [float('inf')] * N
dist[0] = A[0]  # Starting vertex has its own weight
pq = [(A[0], 0)]  # (distance, vertex)

while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]:
        continue
    
    for v, edge_weight in graph[u]:
        new_dist = dist[u] + edge_weight + A[v]
        if new_dist < dist[v]:
            dist[v] = new_dist
            heapq.heappush(pq, (new_dist, v))

# Output distances to vertices 2, 3, ..., N (1-indexed)
result = []
for i in range(1, N):
    result.append(str(dist[i]))

print(' '.join(result))