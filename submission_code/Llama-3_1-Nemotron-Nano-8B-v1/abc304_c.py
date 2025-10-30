import sys
from collections import deque

n, d = map(int, sys.stdin.readline().split())
points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
d_sq = d * d

# Build adjacency list
adj = [[] for _ in range(n)]
for i in range(n):
    x1, y1 = points[i]
    for j in range(n):
        if i == j:
            continue
        x2, y2 = points[j]
        dx = x1 - x2
        dy = y1 - y2
        if dx * dx + dy * dy <= d_sq:
            adj[i].append(j)

# BFS to find reachable nodes
visited = [False] * n
visited[0] = True
q = deque([0])

while q:
    current = q.popleft()
    for neighbor in adj[current]:
        if not visited[neighbor]:
            visited[neighbor] = True
            q.append(neighbor)

# Output results
for status in visited:
    print("Yes" if status else "No")