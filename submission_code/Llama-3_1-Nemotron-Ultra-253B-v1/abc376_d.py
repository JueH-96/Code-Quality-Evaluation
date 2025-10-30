import sys
from collections import deque

n, m = map(int, sys.stdin.readline().split())
edges = []
adj = [[] for _ in range(n + 1)]

for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    edges.append((a, b))
    adj[a].append(b)

distance = [-1] * (n + 1)
distance[1] = 0
q = deque([1])

while q:
    u = q.popleft()
    for v in adj[u]:
        if distance[v] == -1:
            distance[v] = distance[u] + 1
            q.append(v)

min_cycle = float('inf')
for a, b in edges:
    if b == 1 and distance[a] != -1:
        min_cycle = min(min_cycle, distance[a] + 1)

print(min_cycle if min_cycle != float('inf') else -1)