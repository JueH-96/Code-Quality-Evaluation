from collections import deque

# Read input
N, M = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    a, b = map(int, input().split())
    adj[a].append(b)

# BFS from vertex 1 to find shortest distances
dist = [-1] * (N + 1)
dist[1] = 0
queue = deque([1])

while queue:
    u = queue.popleft()
    for v in adj[u]:
        if dist[v] == -1:
            dist[v] = dist[u] + 1
            queue.append(v)

# Find minimum cycle containing vertex 1
min_cycle = float('inf')
for u in range(1, N + 1):
    if 1 in adj[u] and dist[u] != -1:
        # There's an edge from u to 1, and 1 can reach u
        min_cycle = min(min_cycle, dist[u] + 1)

if min_cycle == float('inf'):
    print(-1)
else:
    print(min_cycle)