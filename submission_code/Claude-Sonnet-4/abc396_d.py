import sys
from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    min_xor = float('inf')
    
    def dfs(current, target, visited, current_xor):
        nonlocal min_xor
        
        if current == target:
            min_xor = min(min_xor, current_xor)
            return
        
        for neighbor, weight in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, target, visited, current_xor ^ weight)
                visited.remove(neighbor)
    
    # Start DFS from vertex 1
    visited = {1}
    dfs(1, N, visited, 0)
    
    print(min_xor)

solve()