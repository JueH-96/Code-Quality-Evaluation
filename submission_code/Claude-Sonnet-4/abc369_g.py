import sys
from collections import defaultdict, deque

def solve():
    n = int(input())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        graph[u].append((v, l))
        graph[v].append((u, l))
    
    # Calculate distances from vertex 1 to all other vertices
    distances = [0] * (n + 1)
    visited = [False] * (n + 1)
    
    # BFS from vertex 1
    queue = deque([1])
    visited[1] = True
    
    while queue:
        curr = queue.popleft()
        for neighbor, length in graph[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distances[neighbor] = distances[curr] + length
                queue.append(neighbor)
    
    # Get distances to all vertices except vertex 1, and sort in descending order
    vertex_distances = []
    for i in range(2, n + 1):
        vertex_distances.append(distances[i])
    
    vertex_distances.sort(reverse=True)
    
    # For each K, calculate the answer
    for k in range(1, n + 1):
        if k == 1:
            # Aoki chooses the farthest vertex from vertex 1
            score = 2 * vertex_distances[0]
        else:
            # Aoki chooses the k farthest vertices
            # If k <= n-1, she chooses k vertices excluding vertex 1
            # If k == n, she must include vertex 1, so she chooses the k-1 farthest vertices plus vertex 1
            if k <= n - 1:
                score = 2 * sum(vertex_distances[:k])
            else:  # k == n
                score = 2 * sum(vertex_distances)
        
        print(score)

solve()