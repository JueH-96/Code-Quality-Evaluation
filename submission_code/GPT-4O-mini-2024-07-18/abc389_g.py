from itertools import combinations
from collections import defaultdict
import sys

def is_connected(graph, n):
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
                count += 1
                
    return count == n

def count_even_odd_distance_graphs(n, m):
    vertices = list(range(n))
    edges = list(combinations(vertices, 2))
    total_graphs = 0
    
    for edge_combination in combinations(edges, m):
        graph = defaultdict(list)
        for u, v in edge_combination:
            graph[u].append(v)
            graph[v].append(u)
        
        if is_connected(graph, n):
            # BFS to determine distances from vertex 1 (index 0)
            distances = [-1] * n
            distances[0] = 0
            queue = [0]
            while queue:
                node = queue.pop(0)
                for neighbor in graph[node]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            
            even_count = sum(1 for d in distances if d % 2 == 0)
            odd_count = n - even_count
            
            if even_count == odd_count:
                total_graphs += 1
    
    return total_graphs

def solve():
    input_data = sys.stdin.read().strip()
    n, p = map(int, input_data.split())
    
    results = []
    
    for m in range(n - 1, (n * (n - 1)) // 2 + 1):
        count = count_even_odd_distance_graphs(n, m)
        results.append(count % p)
    
    print(" ".join(map(str, results)))