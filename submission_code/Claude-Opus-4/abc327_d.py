# YOUR CODE HERE
from collections import deque

def is_bipartite(n, edges):
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    
    for a, b in edges:
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    
    # Color array: -1 means uncolored, 0 and 1 are the two colors
    color = [-1] * (n + 1)
    
    # Check each connected component
    for start in range(1, n + 1):
        if color[start] == -1:
            # BFS to color the component
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in adj[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
    
    return True

# Read input
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Create edges
edges = [(a[i], b[i]) for i in range(m)]

# Check if bipartite
if is_bipartite(n, edges):
    print("Yes")
else:
    print("No")