import sys
from collections import defaultdict, deque

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
    return -1

def solve():
    N1, N2, M = map(int, sys.stdin.readline().split())
    graph = defaultdict(list)
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        graph[a].append(b)
        graph[b].append(a)
    max_dist = -1
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N1 + N2 + 1):
            dist = bfs(graph, u, v)
            if dist > max_dist:
                max_dist = dist
    print(max_dist)

solve()