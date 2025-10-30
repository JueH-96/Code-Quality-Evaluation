import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]

# Create adjacency list for the tree
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# To find the levels of the stars, we need to determine the degree of each vertex
degree = [0] * (N + 1)
for u, v in edges:
    degree[u] += 1
    degree[v] += 1

# We need to find the number of leaves (degree 1 vertices) connected to each non-leaf vertex
# This will help us determine the original stars
def find_star_levels():
    # We will use a queue to perform a BFS and count leaves
    queue = deque()
    visited = [False] * (N + 1)
    star_levels = []
    
    # Start with all leaves
    for node in range(1, N + 1):
        if degree[node] == 1:
            queue.append(node)
            visited[node] = True
    
    # BFS to find and count star centers
    while queue:
        node = queue.popleft()
        
        # Check all adjacent nodes
        for neighbor in graph[node]:
            if not visited[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    queue.append(neighbor)
                    visited[neighbor] = True
                elif degree[neighbor] == 0:
                    # This was a star center, count its leaves
                    star_levels.append(len(graph[neighbor]))
    
    # Sort the star levels in ascending order
    star_levels.sort()
    return star_levels

# Get the star levels from the tree
result = find_star_levels()

# Print the result
print(" ".join(map(str, result)))