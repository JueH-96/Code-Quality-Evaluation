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
    N1, N2, M = map(int, input().split())
    graph = [[] for _ in range(N1 + N2 + 1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    max_dist = 0
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N1 + N2 + 1):
            graph[u].append(v)
            graph[v].append(u)
            dist = bfs(graph, 1, N1 + N2)
            graph[u].remove(v)
            graph[v].remove(u)
            max_dist = max(max_dist, dist)
    
    return max_dist

print(solve())