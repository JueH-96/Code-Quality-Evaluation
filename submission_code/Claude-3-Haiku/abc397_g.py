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

def solve():
    N, M, K = map(int, input().split())
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
    
    max_dist = float('inf')
    for edges in range(1 << M):
        new_graph = [g[:] for g in graph]
        for i in range(M):
            if edges & (1 << i):
                u, v = map(int, input().split())
                new_graph[u].remove(v)
                new_graph[u].append(v)
        dist = bfs(new_graph, 1, N)
        max_dist = min(max_dist, dist)
    
    print(max_dist)

solve()