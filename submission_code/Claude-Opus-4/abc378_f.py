# YOUR CODE HERE
from collections import defaultdict, deque

def find_path(graph, start, end, n):
    # BFS to find path from start to end
    parent = [-1] * (n + 1)
    visited = [False] * (n + 1)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)
    
    # Reconstruct path
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path

n = int(input())
graph = defaultdict(list)
degree = [0] * (n + 1)

# Build the tree
for _ in range(n - 1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)
    degree[u] += 1
    degree[v] += 1

count = 0

# Try all possible edges
for u in range(1, n + 1):
    for v in range(u + 1, n + 1):
        # Skip if edge already exists
        if v in graph[u]:
            continue
        
        # Find path from u to v
        path = find_path(graph, u, v, n)
        
        # Check conditions
        valid = True
        
        # Check endpoints (u and v) - they need degree 2 in original tree
        if degree[u] != 2 or degree[v] != 2:
            valid = False
        
        # Check internal vertices - they need degree 3 in original tree
        if valid:
            for i in range(1, len(path) - 1):
                if degree[path[i]] != 3:
                    valid = False
                    break
        
        if valid:
            count += 1

print(count)