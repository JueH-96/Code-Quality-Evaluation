from collections import deque

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a].append(b)
    
    # BFS to find shortest cycle containing vertex 1
    queue = deque()
    visited = [False] * (N + 1)
    
    # Start from all neighbors of vertex 1
    for neighbor in graph[1]:
        queue.append((neighbor, 1))
    
    visited[1] = True  # Mark vertex 1 as visited after adding its neighbors
    
    while queue:
        vertex, dist = queue.popleft()
        
        if vertex == 1:
            # Found a cycle back to vertex 1
            return dist
        
        if not visited[vertex]:
            visited[vertex] = True
            
            # Add all neighbors to queue
            for neighbor in graph[vertex]:
                if not visited[neighbor] or neighbor == 1:
                    queue.append((neighbor, dist + 1))
    
    return -1

print(solve())