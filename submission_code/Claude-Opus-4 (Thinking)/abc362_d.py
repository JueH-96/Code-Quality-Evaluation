import heapq

# Read input
N, M = map(int, input().split())
A = [0] + list(map(int, input().split()))  # 1-indexed

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    u, v, b = map(int, input().split())
    adj[u].append((v, b))
    adj[v].append((u, b))

# Dijkstra's algorithm
dist = [float('inf')] * (N + 1)
dist[1] = A[1]  # Starting from vertex 1, initial weight is A_1

pq = [(A[1], 1)]  # (distance, vertex)
visited = [False] * (N + 1)

while pq:
    d, u = heapq.heappop(pq)
    
    if visited[u]:
        continue
    
    visited[u] = True
    
    for v, edge_weight in adj[u]:
        new_dist = d + edge_weight + A[v]
        if new_dist < dist[v]:
            dist[v] = new_dist
            heapq.heappush(pq, (new_dist, v))

# Output results for vertices 2 to N
result = []
for i in range(2, N + 1):
    result.append(str(dist[i]))

print(' '.join(result))