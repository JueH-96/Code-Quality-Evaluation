# YOUR CODE HERE
import sys
from collections import deque

input = sys.stdin.read
N1, N2, M = map(int, input().split())
edges = [[] for _ in range(N1 + N2 + 1)]
for _ in range(M):
    a, b = map(int, input().split())
    edges[a].append(b)
    edges[b].append(a)

def bfs(start):
    visited = [False] * (N1 + N2 + 1)
    queue = deque([start])
    visited[start] = True
    dist = [0] * (N1 + N2 + 1)
    while queue:
        node = queue.popleft()
        for neighbor in edges[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist

dist1 = bfs(1)
dist2 = bfs(N1 + N2)

max_dist = max(dist1[N1 + N2], dist2[1])
for i in range(2, N1 + 1):
    for j in range(N1 + 1, N1 + N2 + 1):
        if i in edges[j]:
            max_dist = max(max_dist, dist1[i] + dist2[j])
            break

print(max_dist)