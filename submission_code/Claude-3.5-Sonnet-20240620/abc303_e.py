# YOUR CODE HERE
from collections import defaultdict

def find_stars(n, edges):
    # Create adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Find leaf nodes
    leaves = [node for node in range(1, n+1) if len(graph[node]) == 1]
    
    # Find centers of stars
    centers = []
    for node in range(1, n+1):
        if len(graph[node]) > 1 and all(len(graph[neighbor]) == 1 for neighbor in graph[node] if neighbor != node):
            centers.append(node)
    
    # Calculate levels of stars
    levels = [len(graph[center]) for center in centers]
    
    # Sort levels in ascending order
    levels.sort()
    
    return levels

# Read input
n = int(input())
edges = [list(map(int, input().split())) for _ in range(n-1)]

# Find and print star levels
result = find_stars(n, edges)
print(*result)