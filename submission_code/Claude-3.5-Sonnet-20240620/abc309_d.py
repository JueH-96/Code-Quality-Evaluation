# YOUR CODE HERE
from collections import deque

def bfs(graph, start, end):
    queue = deque([(start, 0)])
    visited = set([start])
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

N1, N2, M = map(int, input().split())
N = N1 + N2

graph = [[] for _ in range(N + 1)]

for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

max_d = 0

for u in range(1, N1 + 1):
    dist_u_1 = bfs(graph, u, 1)
    for v in range(N1 + 1, N + 1):
        dist_v_N = bfs(graph, v, N)
        d = dist_u_1 + 1 + dist_v_N
        max_d = max(max_d, d)

print(max_d)