import sys
from collections import defaultdict, deque

def read_ints(): return map(int, sys.stdin.readline().strip().split())

def find_min_weight_walk(N, M, edges):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # Check if there's a vertex with no outgoing edges
    for i in range(1, N + 1):
        if not graph[i]:
            return "No"
    
    # Check if the graph is strongly connected
    visited = [False] * (N + 1)
    def dfs(v):
        visited[v] = True
        for u, _ in graph[v]:
            if not visited[u]:
                dfs(u)
    
    dfs(1)
    if not all(visited[1:]):
        return "No"
    
    # Find the minimum weight walk
    min_weight = 0
    for u in range(1, N + 1):
        for v, w in graph[u]:
            min_weight += w
    
    return min_weight

N, M = read_ints()
edges = [read_ints() for _ in range(M)]
print(find_min_weight_walk(N, M, edges))