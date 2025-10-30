from collections import deque

n, m = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(n + 1)]
for _ in range(m):
    a, b = map(int, input().split())
    adj[a].append(b)

# BFS from vertex 1
dist = [-1] * (n + 1)
dist[1] = 0
queue = deque([1])
min_cycle = float('inf')

while queue:
    u = queue.popleft()
    for v in adj[u]:
        if v == 1:
            # Found a cycle
            min_cycle = min(min_cycle, dist[u] + 1)
        elif dist[v] == -1:
            dist[v] = dist[u] + 1
            queue.append(v)

if min_cycle == float('inf'):
    print(-1)
else:
    print(min_cycle)