import sys

def dijkstra(graph, start, end):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    queue = [start]
    
    while queue:
        u = queue.pop(0)
        if u == end:
            return dist[u]
        
        for v in range(n):
            if graph[u][v] != float('inf'):
                new_dist = dist[u] + graph[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    queue.append(v)
    
    return float('inf')

def solve():
    N, A, B, C = map(int, input().split())
    
    # Create the graph
    graph = [[float('inf')] * N for _ in range(N)]
    for i in range(N):
        row = list(map(int, input().split()))
        for j in range(N):
            graph[i][j] = row[j] * A
            graph[i][j] = min(graph[i][j], row[j] * B + C)
    
    # Find the minimum time
    return dijkstra(graph, 0, N-1)

print(solve())