from collections import deque

def bfs(graph, start, n):
    dist = [-1] * (n+1)
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        v = queue.popleft()
        for u in graph[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                queue.append(u)
                
    return dist

def solve():
    N1, N2, M = map(int, input().split())
    N = N1 + N2
    
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    # Get distances from vertex 1 to all vertices in first component
    dist1 = bfs(graph, 1, N)
    
    # Get distances from vertex N to all vertices in second component
    distN = bfs(graph, N, N)
    
    max_dist = 0
    # Try all possible edges between components
    for u in range(1, N1+1):
        if dist1[u] == -1:
            continue
        for v in range(N1+1, N+1):
            if distN[v] == -1:
                continue
            # Path length will be dist1[u] + 1 + distN[v]
            max_dist = max(max_dist, dist1[u] + 1 + distN[v])
            
    print(max_dist)

solve()