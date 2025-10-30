from itertools import combinations
from collections import defaultdict, deque
import sys

def shortest_path(graph, start, end, n):
    # BFS to find shortest path with 0-1 weights
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                if weight == 0:
                    queue.appendleft(v)  # 0-weight edges go to front
                else:
                    queue.append(v)      # 1-weight edges go to back
    
    return dist[end] if dist[end] != float('inf') else -1

def solve():
    n, m, k = map(int, input().split())
    edges = []
    
    for i in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    max_distance = 0
    
    # Try all combinations of k edges to set weight to 1
    for selected_edges in combinations(range(m), k):
        # Build graph with selected edges having weight 1, others weight 0
        graph = defaultdict(list)
        
        for i, (u, v) in enumerate(edges):
            weight = 1 if i in selected_edges else 0
            graph[u].append((v, weight))
        
        # Find shortest path
        distance = shortest_path(graph, 1, n, n)
        if distance != -1:
            max_distance = max(max_distance, distance)
    
    print(max_distance)

solve()