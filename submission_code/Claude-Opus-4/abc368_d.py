# YOUR CODE HERE
from collections import defaultdict, deque

def find_path(graph, start, end, n):
    """Find path between two vertices using BFS"""
    if start == end:
        return [start]
    
    visited = [False] * (n + 1)
    parent = [-1] * (n + 1)
    queue = deque([start])
    visited[start] = True
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                queue.append(neighbor)
                if neighbor == end:
                    # Reconstruct path
                    path = []
                    current = end
                    while current != -1:
                        path.append(current)
                        current = parent[current]
                    return path[::-1]
    return []

def solve():
    # Read input
    n, k = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(n - 1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    # Read specified vertices
    specified = list(map(int, input().split()))
    
    if k == 1:
        print(1)
        return
    
    # Find all vertices that must be included
    included = set()
    
    # Add all paths between consecutive pairs of specified vertices
    for i in range(len(specified)):
        for j in range(i + 1, len(specified)):
            path = find_path(graph, specified[i], specified[j], n)
            included.update(path)
    
    print(len(included))

solve()