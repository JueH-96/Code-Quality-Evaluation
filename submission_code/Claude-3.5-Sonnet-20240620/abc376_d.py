# YOUR CODE HERE
from collections import deque

def bfs(graph, start):
    queue = deque([(start, 0)])
    visited = set([start])
    parent = {start: None}
    
    while queue:
        node, distance = queue.popleft()
        
        if node == 1 and distance > 0:
            return distance
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append((neighbor, distance + 1))
            elif neighbor == 1 and parent[node] != 1:
                return distance + 1
    
    return -1

# Read input
N, M = map(int, input().split())
graph = {i: [] for i in range(1, N+1)}

for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)

# Find the minimum cycle length containing vertex 1
result = bfs(graph, 1)

# Print the result
print(result)