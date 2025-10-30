import sys
from collections import defaultdict, deque

def min_vertices_in_tree(N, K, edges, vertices):
    # Build the adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Mark the specified vertices
    marked = set(vertices)
    
    # Function to perform BFS and find the minimum subtree
    def bfs(start):
        queue = deque([start])
        visited = set([start])
        count = 0
        
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    if v in marked:
                        queue.append(v)
        
        return count
    
    # Find the minimum subtree that includes all marked vertices
    min_vertices = float('inf')
    for v in vertices:
        min_vertices = min(min_vertices, bfs(v))
    
    return min_vertices

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
edges = [(int(data[i]), int(data[i+1])) for i in range(2, 2*(N-1)+2, 2)]
vertices = [int(data[i]) for i in range(2*(N-1)+4, 2*(N-1)+4+K)]

# Solve the problem
result = min_vertices_in_tree(N, K, edges, vertices)

# Print the result
print(result)